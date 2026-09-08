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

import json
import logging
import re
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
    # Shell globs for the model's own test files. They are moved aside before
    # the hidden suite runs: a helper the model happens to name the same as one
    # of ours collides at compile time and zeroes the whole package, which is a
    # harness artefact rather than a fact about the model's code.
    isolate_globs: str = ""
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
            isolate_globs=data.get("isolate_globs", ""),
            setup_cmd=data.get("setup_cmd", ""),
            memory=data.get("memory", DEFAULT_MEMORY),
            network=data.get("network", "none"),
            pids=int(data.get("pids", DEFAULT_PIDS)),
            workspace_size=data.get("workspace_size", DEFAULT_WORKSPACE_SIZE),
            tmp_size=data.get("tmp_size", DEFAULT_TMP_SIZE),
            cpus=str(data.get("cpus", DEFAULT_CPUS)),
            command_timeout=int(data.get("command_timeout", DEFAULT_COMMAND_TIMEOUT)),
        )


# An `ls -l` entry: mode bits, link count, owner, group, size, then the date.
# Anchored to the start of a line and required to carry the whole prefix, so a
# date inside file contents cannot match — every shell result passes through
# here, including `cat` behind read_file, and rewriting source the model is
# about to copy would be far worse than the non-determinism being fixed.
# `find -ls`, `ls -li` and `ls -ls` prefix each entry with an inode and/or a
# block count, so the mode bits are not at the line start and the date was
# leaking through. The prefix is optional and purely numeric; the whole
# mode/links/owner/group/size run is still required, so the safety property
# above is unchanged.
_LS_ENTRY = r"^(\s*(?:\d+\s+){0,2}[bcdlps-][rwxsStT-]{9}[.+]?\s+\d+\s+\S+\s+\S+\s+\d+\s+)"
_LS_DATE_RE = re.compile(_LS_ENTRY + r"(\w{3}\s+\d{1,2}\s+(?:\d{2}:\d{2}|\d{4}))", re.MULTILINE)
_LS_ISO_RE = re.compile(
    _LS_ENTRY + r"(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:\s+[+-]\d{4})?)",
    re.MULTILINE,
)
_LS_DATE_PLACEHOLDER = "Jan  1 00:00"
_LS_ISO_PLACEHOLDER = "1970-01-01 00:00:00.000000000 +0000"

# CPython's default repr carries the object's heap address, which ASLR moves on
# every container start: `<horizons_client.HorizonsClient object at 0x7038...>`.
# It reaches the model through pytest failure output. Anchored on both sides —
# ` at ` before, the closing `>` after — so a hex literal in source or in a test
# assertion is never touched.
_PY_ADDR_RE = re.compile(r"( at )0x[0-9a-fA-F]{4,}(?=>)")
_PY_ADDR_PLACEHOLDER = "0x0000000000"

# `go test` prints wall-clock elapsed per package and per test, and two runs of
# identical code differ in the last millisecond (`0.005s` vs `0.006s`). Anchored
# to the status line's own shape — a leading `ok`/`FAIL` plus the package, or a
# `--- PASS:` header — so a duration a program prints itself survives. Neither
# is anchored to end-of-line, so `coverage:` trailers are preserved.
_GO_PKG_TIME_RE = re.compile(r"^(ok|FAIL)([ \t]+\S+[ \t]+)\d+\.\d+s", re.MULTILINE)
_GO_TEST_TIME_RE = re.compile(r"^([ \t]*--- (?:PASS|FAIL|SKIP): \S+ \()\d+\.\d+s\)", re.MULTILINE)
_GO_PKG_TIME_PLACEHOLDER = "0.000s"
_GO_TEST_TIME_PLACEHOLDER = "0.00s)"

# pytest's summary line, the third runtime's version of the same leak:
# `1 failed, 3 passed, 1 skipped, 1 warning in 0.02s`, optionally padded with
# `=` when the task does not pass -q. Anchored at BOTH ends of the line — it
# must be nothing but pytest's own `<n> <word>` run followed by the duration —
# so `cache warmed: 2 passed in 0.5s` and `2 failed in 0.34s window` survive.
# A line that is exactly `2 passed in 0.5s` and came from the model's own code
# is indistinguishable from pytest's and is rewritten; nothing can separate them.
# The `in 125.34s (0:02:05)` form pytest uses past a minute is deliberately not
# matched — unobserved here, and failing to match only leaves it alone.
_PYTEST_SUMMARY_RE = re.compile(
    r"^(=*[ ]*\d+ [a-z]+(?:, \d+ [a-z]+)* in )\d+\.\d+s(?=[ ]*=*$)",
    re.MULTILINE,
)
_PYTEST_SUMMARY_PLACEHOLDER = "0.00s"

# `deno test` does the same at millisecond resolution: `... ok (5ms)` per case
# and `ok | 3 passed | 0 failed (12ms)` in the summary. Anchored on deno's own
# `... ok` marker and on the summary's `N failed` field.
_DENO_CASE_TIME_RE = re.compile(r"(\.{3} (?:ok|FAILED|ignored|cancelled)) \(\d+(?:\.\d+)?m?s\)")
_DENO_SUMMARY_TIME_RE = re.compile(r"^((?:ok|FAILED) \|.*\d+ failed[^(\n]*)\(\d+(?:\.\d+)?m?s\)", re.MULTILINE)
_DENO_TIME_PLACEHOLDER = "(0ms)"

# Wall clock printed by code the MODEL wrote. Only two shapes are covered: Go's
# `log` default prefix (`Ldate|Ltime`, optionally `Lmicroseconds`) at the start
# of a line, and `log/slog`'s text handler, anchored on its literal `time=` key.
# Line-start plus a full date-and-time is the tightest anchor available here.
_GO_LOG_TIME_RE = re.compile(r"^\d{4}/\d{2}/\d{2} \d{2}:\d{2}:\d{2}(?:\.\d+)?", re.MULTILINE)
_SLOG_TIME_RE = re.compile(r"\btime=\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?")
_GO_LOG_TIME_PLACEHOLDER = "1970/01/01 00:00:00"
_SLOG_TIME_PLACEHOLDER = "time=1970-01-01T00:00:00Z"

# Docker names a container after the first 12 hex of its id, and the model sees
# that whenever it runs `env`. Anchored on the variable name and a whole line,
# because a bare `hostname` call emits the same 12 characters with no context to
# separate them from a digest — that call stays unhandled rather than guessed at.
_HOSTNAME_RE = re.compile(r"^(HOSTNAME=)[0-9a-f]{12}$", re.MULTILINE)
_HOSTNAME_PLACEHOLDER = "sandbox"


def _normalize_volatile(text: str) -> str:
    """Blank out per-container variation, so two runs see byte-identical output.

    A directory's mtime is the container's creation time, so `ls -la` on turn 1
    differs between two runs of the same model and the conversation diverges
    from turn 2 onward — at temperature 0, producing different code and
    different automated scores. Measuring repeat runs found the first differing
    tool-call argument always landed immediately after the first differing tool
    result, on one of seven surfaces: `ls` mtimes, ASLR heap addresses in Python
    reprs, elapsed times from all three test runners the sandbox images provide
    (`pytest`, `go test`, `deno test`), a wall clock printed by the model's own
    generated code, and the container hostname.

    Every pattern is anchored to the context that identifies it, never to the
    bare volatile token, because every shell result passes through here —
    including `cat` behind read_file. Rewriting source the model is about to
    copy would be worse than the non-determinism being fixed.

    **Model-authored output is only partly controllable.** The harness does not
    choose the log format of code it did not write, so only Go's `log` default
    prefix and `log/slog`'s `time=` key are handled. A model that formats its own
    timestamp, prints a duration, or seeds from the clock still diverges.

    **An eighth surface is out of reach of substitution entirely: Go randomizes
    map iteration order per process**, so a table-driven test backed by a map
    emits its subtest blocks in a different order each run (measured on
    coding_mcp_easy_01, ornith-1.5-35b-a3b, two runs of TestLoadInvalidURL).
    That is the order of lines, not a token within one, and normalising it would
    mean sorting output — which mangles interleaved writers and is not this
    function's job. See test_sandbox_normalization.py.

    So this makes coding markedly less non-deterministic, not reproducible.
    """
    text = _LS_ISO_RE.sub(lambda m: m.group(1) + _LS_ISO_PLACEHOLDER, text)
    text = _LS_DATE_RE.sub(lambda m: m.group(1) + _LS_DATE_PLACEHOLDER, text)
    text = _PY_ADDR_RE.sub(lambda m: m.group(1) + _PY_ADDR_PLACEHOLDER, text)
    text = _PYTEST_SUMMARY_RE.sub(lambda m: m.group(1) + _PYTEST_SUMMARY_PLACEHOLDER, text)
    text = _GO_PKG_TIME_RE.sub(lambda m: m.group(1) + m.group(2) + _GO_PKG_TIME_PLACEHOLDER, text)
    text = _GO_TEST_TIME_RE.sub(lambda m: m.group(1) + _GO_TEST_TIME_PLACEHOLDER, text)
    text = _DENO_CASE_TIME_RE.sub(lambda m: m.group(1) + " " + _DENO_TIME_PLACEHOLDER, text)
    text = _DENO_SUMMARY_TIME_RE.sub(lambda m: m.group(1) + _DENO_TIME_PLACEHOLDER, text)
    text = _GO_LOG_TIME_RE.sub(_GO_LOG_TIME_PLACEHOLDER, text)
    text = _SLOG_TIME_RE.sub(_SLOG_TIME_PLACEHOLDER, text)
    return _HOSTNAME_RE.sub(lambda m: m.group(1) + _HOSTNAME_PLACEHOLDER, text)


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
    """Whether a Docker daemon is reachable.

    Honours DOCKER_HOST, so an orchestrator running in a pod can drive a daemon
    elsewhere rather than needing a socket mounted into it.
    """
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
            stdout=_truncate(_normalize_volatile(result.stdout)),
            stderr=_truncate(_normalize_volatile(result.stderr)),
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

    @staticmethod
    def _as_text(value: object) -> str:
        """Coerce a model-supplied argument to text.

        Tool arguments are whatever the model put in its JSON. A model writing a
        config file may pass `content` as an object rather than a string, and
        that object went straight into subprocess(input=...), which raised
        'dict' object has no attribute 'encode' and killed the task. Structured
        values are serialised as JSON, which is what the model meant by them.
        """
        if isinstance(value, str):
            return value
        if isinstance(value, (dict, list)):
            return json.dumps(value, indent=2)
        if value is None:
            return ""
        return str(value)

    def call(self, tool_name: str, arguments: dict) -> dict:
        """Dispatch a tool call. Mirrors MockToolEnv.call's contract."""
        self.call_log.append({"name": tool_name, "arguments": arguments})

        try:
            if tool_name == "write_file":
                return {
                    "content": self.write_file(
                        self._as_text(arguments.get("path", "")),
                        self._as_text(arguments.get("content", "")),
                    )
                }
            if tool_name == "read_file":
                return {"content": self.read_file(self._as_text(arguments.get("path", "")))}
            if tool_name == "run_tests":
                return {"content": self.run_tests(self._as_text(arguments.get("directory", "")))}
            if tool_name == "run_code":
                result = self.exec(self._as_text(arguments.get("command", "")))
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

        # Move the model's own tests out of the way first. coding_mcp_hard_01
        # declared a `upstream(...)` helper, as did our hidden suite; Go saw
        # "upstream redeclared in this block", the package failed to compile,
        # and both test_pass_rate and race_detector_clean scored 0 without a
        # single test running.
        if self.spec.isolate_globs:
            moved = self.exec(
                f"mkdir -p /tmp/model_tests && for f in {self.spec.isolate_globs}; do "
                '[ -e "$f" ] && mv "$f" /tmp/model_tests/ || true; done; true'
            )
            if moved.exit_code != 0:
                logger.warning("Could not isolate the model's tests: %s", moved.stderr[:200])

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
