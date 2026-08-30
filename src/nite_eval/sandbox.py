"""Real execution environment for coding tasks.

Replaces the mock tools with a container the model can actually write to and
run commands in. The mocks reported success unconditionally — `run_tests`
returned `{passed: 4, failed: 0}` before the model had written anything, and
`run_code` returned empty stdout with exit 0 — so a model that emitted garbage
and one that emitted a correct implementation received identical feedback, and
models burned their turn budgets retrying for output that never came.

`SandboxToolEnv` exposes the same `call(tool_name, arguments)` interface as
`MockToolEnv`, so `run_conversation` does not know the difference.

Security: this executes model-generated code. Containers run with no network,
a non-root user, a read-only root filesystem, capped memory/CPU/PIDs, and hard
timeouts. Nothing is mounted from the host — files are streamed in over stdin,
never a bind mount, so a symlink in model output cannot reach host paths.
"""

from __future__ import annotations

import logging
import subprocess
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

logger = logging.getLogger(__name__)

# Resource ceilings for model-generated code.
# Sized for compiling, not just running. `go test -race` forks a compiler and
# a vet process per package and writes a large build cache; at 256 PIDs and a
# 256m /tmp it failed with "resource temporarily unavailable" and "no space
# left on device", which reads exactly like a failing race check.
DEFAULT_MEMORY = "2g"
DEFAULT_CPUS = "2"
DEFAULT_PIDS = 1024
DEFAULT_WORKSPACE_SIZE = "1g"
DEFAULT_TMP_SIZE = "2g"
DEFAULT_COMMAND_TIMEOUT = 120
CONTAINER_START_TIMEOUT = 60
MAX_OUTPUT_CHARS = 8000

# Every sandbox carries this label so orphans can be found and removed. A
# process killed mid-task (a CI timeout, Ctrl-C) never runs stop(), and the
# container then idles indefinitely holding memory.
SANDBOX_LABEL = "nite-eval-sandbox"


class SandboxError(RuntimeError):
    """Raised when the sandbox itself fails, as distinct from the code in it."""


@dataclass
class ExecResult:
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool = False


@dataclass
class SandboxSpec:
    """Execution environment declared by a task's `environment:` block."""

    image: str
    workdir: str = "/workspace"
    test_cmd: str = ""
    # Command used for scoring. `test_cmd` collects the whole workspace, which
    # includes the model's own tests — scoring on that lets a model that writes
    # many trivial passing tests dilute the hidden suite toward 1.0. Falls back
    # to test_cmd when a task has no separate scoring command.
    hidden_test_cmd: str = ""
    setup_cmd: str = ""
    memory: str = DEFAULT_MEMORY
    cpus: str = DEFAULT_CPUS
    command_timeout: int = DEFAULT_COMMAND_TIMEOUT
    # Docker network mode. Defaults to "none": the sandbox runs model-generated
    # code, and a test that reaches a live service is both an egress path and a
    # source of failures that look like model regressions. Opt in per task only
    # where a stub cannot test the same thing.
    network: str = "none"
    pids: int = DEFAULT_PIDS
    workspace_size: str = DEFAULT_WORKSPACE_SIZE
    tmp_size: str = DEFAULT_TMP_SIZE

    @classmethod
    def from_task_yaml(cls, data: dict | None) -> SandboxSpec | None:
        if not data or not data.get("image"):
            return None
        return cls(
            image=data["image"],
            workdir=data.get("workdir", "/workspace"),
            test_cmd=data.get("test_cmd", ""),
            hidden_test_cmd=data.get("hidden_test_cmd", ""),
            setup_cmd=data.get("setup_cmd", ""),
            memory=data.get("memory", DEFAULT_MEMORY),
            network=data.get("network", "none"),
            pids=int(data.get("pids", DEFAULT_PIDS)),
            workspace_size=data.get("workspace_size", DEFAULT_WORKSPACE_SIZE),
            tmp_size=data.get("tmp_size", DEFAULT_TMP_SIZE),
            cpus=str(data.get("cpus", DEFAULT_CPUS)),
            command_timeout=int(data.get("command_timeout", DEFAULT_COMMAND_TIMEOUT)),
        )


def _truncate(text: str) -> str:
    if len(text) <= MAX_OUTPUT_CHARS:
        return text
    return text[:MAX_OUTPUT_CHARS] + f"\n[... {len(text) - MAX_OUTPUT_CHARS} more chars truncated ...]"


def _docker(args: list[str], timeout: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["docker", *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def docker_available() -> bool:
    try:
        return _docker(["info", "--format", "{{.ServerVersion}}"], timeout=15).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def reap_orphans(older_than_seconds: int = 0, label: str = SANDBOX_LABEL) -> list[str]:
    """Remove sandbox containers left behind by an interrupted run.

    Returns the ids removed. Only touches containers carrying `label`, so it
    cannot disturb anything else on the host.

    `older_than_seconds` guards against reaping a live run. Reaping purely by
    label destroyed an in-flight evaluation once: the test suite exercises this
    function, and running the tests while an eval was going removed that eval's
    container mid-task, which surfaced as "No such container" and an unscored
    criterion. Callers that might overlap a run should pass a threshold; tests
    should pass their own `label`.
    """
    listing = _docker(["ps", "--quiet", "--filter", f"label={label}=1"], timeout=15)
    if listing.returncode != 0:
        return []

    removed = []
    for container_id in listing.stdout.split():
        if older_than_seconds:
            started = _docker(["inspect", "-f", "{{.State.StartedAt}}", container_id], timeout=15)
            if started.returncode != 0:
                continue
            try:
                began = datetime.fromisoformat(started.stdout.strip().replace("Z", "+00:00"))
            except ValueError:
                continue
            if (datetime.now(UTC) - began).total_seconds() < older_than_seconds:
                continue
        if _docker(["rm", "--force", container_id], timeout=30).returncode == 0:
            removed.append(container_id)
            logger.warning("Reaped orphaned sandbox %s", container_id[:12])
    return removed


@dataclass
class SandboxToolEnv:
    """A running container the model writes files into and executes commands in."""

    spec: SandboxSpec
    container_id: str = ""
    call_log: list[dict] = field(default_factory=list)
    files_written: list[str] = field(default_factory=list)

    def start(self) -> None:
        """Launch the container. It idles until commands are exec'd into it."""
        name = f"nite-eval-{uuid.uuid4().hex[:12]}"
        result = _docker(
            [
                "run",
                "--detach",
                "--name",
                name,
                "--label",
                f"{SANDBOX_LABEL}=1",
                "--network",
                self.spec.network,
                "--memory",
                self.spec.memory,
                "--memory-swap",
                self.spec.memory,  # equal to memory disables swap
                "--cpus",
                self.spec.cpus,
                "--pids-limit",
                str(self.spec.pids),
                "--read-only",
                "--tmpfs",
                f"{self.spec.workdir}:rw,exec,size={self.spec.workspace_size}",
                "--tmpfs",
                f"/tmp:rw,exec,size={self.spec.tmp_size}",
                "--security-opt",
                "no-new-privileges",
                "--cap-drop",
                "ALL",
                "--workdir",
                self.spec.workdir,
                self.spec.image,
                "sleep",
                "infinity",
            ],
            timeout=CONTAINER_START_TIMEOUT,
        )
        if result.returncode != 0:
            raise SandboxError(f"failed to start sandbox from {self.spec.image}: {result.stderr.strip()}")

        self.container_id = result.stdout.strip()
        logger.info("Sandbox %s started from %s", self.container_id[:12], self.spec.image)
        if self.spec.network != "none":
            logger.warning(
                "Sandbox %s has network access (%s) — model-generated code can reach the network, "
                "and live-service failures will look like model regressions",
                self.container_id[:12],
                self.spec.network,
            )

        if self.spec.setup_cmd:
            setup = self.exec(self.spec.setup_cmd)
            if setup.exit_code != 0:
                logger.warning("Sandbox setup_cmd failed (%d): %s", setup.exit_code, setup.stderr[:200])

    def stop(self) -> None:
        if not self.container_id:
            return
        _docker(["rm", "--force", self.container_id], timeout=30)
        logger.info("Sandbox %s removed", self.container_id[:12])
        self.container_id = ""

    def exec(self, command: str, timeout: int | None = None) -> ExecResult:
        """Run a shell command inside the sandbox."""
        if not self.container_id:
            raise SandboxError("sandbox is not running")

        limit = timeout or self.spec.command_timeout
        try:
            # Nothing from `command` is interpolated into the docker argv; it is
            # passed as a single argument to the container's own shell.
            result = _docker(
                ["exec", "--workdir", self.spec.workdir, self.container_id, "sh", "-c", command],
                timeout=limit,
            )
        except subprocess.TimeoutExpired:
            return ExecResult(
                exit_code=124,
                stdout="",
                stderr=f"command exceeded {limit}s and was killed",
                timed_out=True,
            )
        return ExecResult(
            exit_code=result.returncode,
            stdout=_truncate(result.stdout),
            stderr=_truncate(result.stderr),
        )

    def write_file(self, path: str, content: str) -> dict:
        """Write a file into the sandbox by streaming it to `cat` over stdin.

        Not `docker cp`: the daemon refuses to copy into a container whose
        rootfs is read-only, even when the destination is a writable tmpfs, and
        dropping --read-only to satisfy it would trade isolation for
        convenience. Not a bind mount either, so a symlink in model output
        cannot reach host paths. Content travels on stdin rather than inside
        the command string, so quoting in generated source cannot break out.
        """
        if not self.container_id:
            raise SandboxError("sandbox is not running")

        dest = path if path.startswith("/") else f"{self.spec.workdir}/{path}"
        resolved = PurePosixPath(dest)
        allowed = PurePosixPath(self.spec.workdir)
        if not (resolved == allowed or allowed in resolved.parents) or ".." in resolved.parts:
            return {"error": f"refusing to write outside the workspace: {path}"}

        try:
            result = subprocess.run(
                [
                    "docker",
                    "exec",
                    "-i",
                    "--workdir",
                    self.spec.workdir,
                    self.container_id,
                    "sh",
                    "-c",
                    'mkdir -p "$(dirname "$1")" && cat > "$1"',
                    "sh",
                    dest,
                ],
                input=content,
                capture_output=True,
                text=True,
                timeout=self.spec.command_timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return {"error": f"write timed out after {self.spec.command_timeout}s"}

        if result.returncode != 0:
            return {"error": f"write failed: {result.stderr.strip()}"}

        self.files_written.append(dest)
        return {"status": "written", "path": dest, "bytes": len(content.encode())}

    def read_file(self, path: str) -> dict:
        dest = path if path.startswith("/") else f"{self.spec.workdir}/{path}"
        result = self.exec(f"cat {dest}")
        if result.exit_code != 0:
            return {"error": f"could not read {dest}: {result.stderr.strip()}"}
        return {"content": result.stdout}

    def run_tests(self, directory: str = "", command: str = "") -> dict:
        chosen = command or self.spec.test_cmd
        if not chosen:
            return {"error": "no test command configured for this task"}
        cmd = f"cd {directory} && {chosen}" if directory else chosen
        result = self.exec(cmd)
        return {
            "exit_code": result.exit_code,
            "passed": result.exit_code == 0,
            "output": result.stdout or result.stderr,
            "timed_out": result.timed_out,
        }

    # --- MockToolEnv-compatible interface ---

    def call(self, tool_name: str, arguments: dict) -> dict:
        """Dispatch a tool call. Mirrors MockToolEnv.call's contract."""
        self.call_log.append({"name": tool_name, "arguments": arguments})

        try:
            if tool_name == "write_file":
                return {"content": self.write_file(arguments.get("path", ""), arguments.get("content", ""))}
            if tool_name == "read_file":
                return {"content": self.read_file(arguments.get("path", ""))}
            if tool_name == "run_tests":
                return {"content": self.run_tests(arguments.get("directory", ""))}
            if tool_name == "run_code":
                result = self.exec(arguments.get("command", ""))
                return {
                    "content": {
                        "exit_code": result.exit_code,
                        "stdout": result.stdout,
                        "stderr": result.stderr,
                        "timed_out": result.timed_out,
                    }
                }
        except SandboxError as e:
            return {"error": f"sandbox error: {e}"}

        return {"error": f"tool '{tool_name}' is not available in the execution environment"}

    def run_hidden_suite(self, suite_dir: Path) -> dict:
        """Copy our test files in and run them against the model's code.

        The model's own tests stay visible to it for iteration, but they do not
        decide the score — a model that writes trivial tests would otherwise
        grade itself. Files are copied after the conversation ends so the model
        never sees them.
        """
        if not suite_dir.exists():
            return {"error": f"hidden suite not found: {suite_dir}"}

        copied = []
        for path in sorted(suite_dir.rglob("*")):
            if path.is_dir():
                continue
            rel = path.relative_to(suite_dir).as_posix()
            result = self.write_file(rel, path.read_text())
            if "error" in result:
                return {"error": f"could not install {rel}: {result['error']}"}
            copied.append(rel)

        # Scored with hidden_test_cmd, which is scoped to our files. The model
        # keeps seeing its own tests through its own run_tests calls.
        outcome = self.run_tests(command=self.spec.hidden_test_cmd or self.spec.test_cmd)
        outcome["hidden_files"] = copied
        outcome["scoring_command"] = self.spec.hidden_test_cmd or self.spec.test_cmd
        return outcome

    def get_call_log(self) -> list[dict]:
        return self.call_log

    def __enter__(self) -> SandboxToolEnv:
        self.start()
        return self

    def __exit__(self, *args: object) -> None:
        self.stop()
