"""Sandbox execution environment.

Split into two groups: pure logic that runs anywhere, and integration tests
that need a Docker daemon and are skipped without one.
"""

from unittest.mock import patch

import pytest

from nite_eval.sandbox import SandboxSpec, SandboxToolEnv, docker_available

pytestmark_docker = pytest.mark.skipif(not docker_available(), reason="requires a Docker daemon")


# --- spec parsing (no Docker) ---


def test_spec_absent_without_an_image():
    assert SandboxSpec.from_task_yaml(None) is None
    assert SandboxSpec.from_task_yaml({}) is None
    assert SandboxSpec.from_task_yaml({"workdir": "/app"}) is None


def test_spec_defaults_and_overrides():
    spec = SandboxSpec.from_task_yaml({"image": "golang:1.23-alpine"})
    assert spec is not None
    assert spec.workdir == "/workspace"
    assert spec.memory == "2g"  # sized for compiling, not just running

    spec = SandboxSpec.from_task_yaml(
        {"image": "node:20-alpine", "workdir": "/app", "test_cmd": "npm test", "cpus": 4, "command_timeout": 300}
    )
    assert spec is not None
    assert spec.workdir == "/app"
    assert spec.test_cmd == "npm test"
    assert spec.cpus == "4"  # normalised to str for the docker argv
    assert spec.command_timeout == 300


# --- path containment (no Docker) ---


@pytest.mark.parametrize(
    "path",
    ["../../etc/passwd", "/etc/passwd", "../outside.txt", "/workspace/../../root/.ssh/id_rsa"],
)
def test_write_refuses_paths_outside_the_workspace(path):
    env = SandboxToolEnv(SandboxSpec(image="python:3.12-alpine"), container_id="fake")
    result = env.write_file(path, "x")
    assert "error" in result
    assert "outside the workspace" in result["error"]


@pytest.mark.parametrize("path", ["a.py", "pkg/sub/mod.py", "/workspace/nested/file.go"])
def test_write_allows_paths_inside_the_workspace(path):
    """Containment must not reject legitimate writes."""
    env = SandboxToolEnv(SandboxSpec(image="python:3.12-alpine"), container_id="fake")
    with patch("nite_eval.sandbox.subprocess.run") as run:
        run.return_value.returncode = 0
        run.return_value.stderr = ""
        result = env.write_file(path, "content")
    assert result.get("status") == "written"


def test_unknown_tool_is_reported_not_silently_succeeded():
    env = SandboxToolEnv(SandboxSpec(image="python:3.12-alpine"), container_id="fake")
    result = env.call("send_email", {})
    assert "error" in result


def test_run_tests_without_a_configured_command_is_an_error():
    env = SandboxToolEnv(SandboxSpec(image="python:3.12-alpine"), container_id="fake")
    assert "error" in env.run_tests()


# --- integration (needs Docker) ---


@pytestmark_docker
def test_written_code_actually_executes():
    spec = SandboxSpec(image="python:3.12-alpine", command_timeout=30)
    with SandboxToolEnv(spec) as sb:
        sb.call("write_file", {"path": "hello.py", "content": "def add(a, b):\n    return a + b\n"})
        result = sb.call("run_code", {"command": "python -c 'import hello; print(hello.add(2, 3))'"})
    assert result["content"]["exit_code"] == 0
    assert result["content"]["stdout"].strip() == "5"


@pytestmark_docker
def test_file_content_survives_shell_metacharacters():
    """Model-written Go and TypeScript is full of quotes and backslashes."""
    tricky = """s = "it's $HOME `whoami` \\\\ end"\nprint(len(s))\n"""
    spec = SandboxSpec(image="python:3.12-alpine", command_timeout=30)
    with SandboxToolEnv(spec) as sb:
        sb.call("write_file", {"path": "q.py", "content": tricky})
        assert sb.read_file("q.py")["content"] == tricky


@pytestmark_docker
def test_network_is_unavailable():
    spec = SandboxSpec(image="python:3.12-alpine", command_timeout=10)
    with SandboxToolEnv(spec) as sb:
        result = sb.call("run_code", {"command": "wget -q -T2 -O- http://example.com; echo rc=$?"})
    out = result["content"]
    assert out["timed_out"] or "rc=0" not in out["stdout"]


@pytestmark_docker
def test_root_filesystem_is_read_only():
    spec = SandboxSpec(image="python:3.12-alpine", command_timeout=15)
    with SandboxToolEnv(spec) as sb:
        result = sb.call("run_code", {"command": "touch /usr/local/x"})
    assert result["content"]["exit_code"] != 0


@pytestmark_docker
def test_runaway_command_is_killed():
    spec = SandboxSpec(image="python:3.12-alpine", command_timeout=3)
    with SandboxToolEnv(spec) as sb:
        result = sb.call("run_code", {"command": "sleep 60"})
    assert result["content"]["timed_out"] is True
    assert result["content"]["exit_code"] == 124


@pytestmark_docker
def test_failing_tests_report_failure():
    """The whole point: a wrong implementation must not report success."""
    spec = SandboxSpec(
        image="python:3.12-alpine",
        test_cmd="python -m unittest discover -p 'test_*.py' 2>&1",
        command_timeout=60,
    )
    with SandboxToolEnv(spec) as sb:
        sb.call("write_file", {"path": "impl.py", "content": "def add(a, b):\n    return a - b\n"})
        sb.call(
            "write_file",
            {
                "path": "test_impl.py",
                "content": (
                    "import unittest\n"
                    "from impl import add\n\n\n"
                    "class T(unittest.TestCase):\n"
                    "    def test_add(self):\n"
                    "        self.assertEqual(add(2, 3), 5)\n"
                ),
            },
        )
        result = sb.call("run_tests", {})
    assert result["content"]["passed"] is False
    assert result["content"]["exit_code"] != 0


@pytestmark_docker
def test_container_is_removed_on_exit():
    spec = SandboxSpec(image="python:3.12-alpine")
    with SandboxToolEnv(spec) as sb:
        container_id = sb.container_id
        assert container_id
    assert sb.container_id == ""
    from nite_eval.sandbox import _docker

    check = _docker(["inspect", container_id], timeout=15)
    assert check.returncode != 0, "container outlived the context manager"


def test_network_defaults_to_none():
    """Egress must be opt-in, never the default."""
    spec = SandboxSpec.from_task_yaml({"image": "python:3.12-alpine"})
    assert spec is not None
    assert spec.network == "none"


def test_network_can_be_enabled_per_task():
    spec = SandboxSpec.from_task_yaml({"image": "python:3.12-alpine", "network": "bridge"})
    assert spec is not None
    assert spec.network == "bridge"


@pytestmark_docker
def test_orphaned_sandboxes_are_reapable():
    """A killed process never runs stop(), leaving the container idling.

    Uses a test-only label. Reaping by the production label would remove any
    sandbox belonging to an evaluation running on this host — which is exactly
    what happened once, destroying a live run's container mid-task.
    """
    from nite_eval.sandbox import _docker, reap_orphans

    test_label = "nite-eval-sandbox-test"
    orphan = SandboxToolEnv(SandboxSpec(image="python:3.12-alpine"))
    with patch("nite_eval.sandbox.SANDBOX_LABEL", test_label):
        orphan.start()
    container_id = orphan.container_id
    assert container_id
    # Simulate the process dying: never call stop().

    removed = reap_orphans(label=test_label)
    assert any(container_id.startswith(r) or r.startswith(container_id[:12]) for r in removed)
    assert _docker(["inspect", container_id], timeout=15).returncode != 0


def test_reaper_can_spare_recently_started_containers():
    """The age guard is what keeps a live run's sandbox safe."""
    from nite_eval.sandbox import reap_orphans

    with patch("nite_eval.sandbox._docker") as docker:
        docker.return_value.returncode = 0
        docker.return_value.stdout = ""
        assert reap_orphans(older_than_seconds=3600) == []


def test_hidden_test_cmd_defaults_to_test_cmd():
    spec = SandboxSpec.from_task_yaml({"image": "x", "test_cmd": "go test ./..."})
    assert spec is not None
    assert spec.hidden_test_cmd == ""  # falls back at call time


def test_hidden_test_cmd_is_separate_from_test_cmd():
    """Scoring must not run the model's own tests; the model still may."""
    spec = SandboxSpec.from_task_yaml(
        {"image": "x", "test_cmd": "go test ./...", "hidden_test_cmd": "go test -run '^TestHidden' ./..."}
    )
    assert spec is not None
    assert spec.test_cmd == "go test ./..."
    assert spec.hidden_test_cmd == "go test -run '^TestHidden' ./..."


@pytestmark_docker
def test_scoring_ignores_the_models_own_tests():
    """Regression: `go test ./...` counted 20 tests where the hidden suite has 8.

    A model that writes many trivial passing tests would otherwise dilute the
    hidden suite toward 1.0 regardless of whether its code is correct.
    """
    spec = SandboxSpec(
        image="python:3.12-alpine",
        test_cmd="python -m unittest discover -p '*_test.py' 2>&1",
        hidden_test_cmd="python -m unittest hidden_check 2>&1",
        command_timeout=60,
    )
    hidden = (
        "import unittest\n\n\nclass H(unittest.TestCase):\n"
        "    def test_real(self):\n        self.assertEqual(1, 2)\n"  # fails
    )
    model_own = "import unittest\n\n\nclass M(unittest.TestCase):\n" + "".join(
        f"    def test_trivial_{i}(self):\n        self.assertTrue(True)\n" for i in range(10)
    )
    with SandboxToolEnv(spec) as sb:
        sb.call("write_file", {"path": "hidden_check.py", "content": hidden})
        sb.call("write_file", {"path": "model_own_test.py", "content": model_own})

        scored = sb.run_tests(command=spec.hidden_test_cmd)
        assert scored["passed"] is False, "the failing hidden test must decide the score"

        # The model's own view still includes its tests.
        own = sb.run_tests()
        assert "10" in own["output"] or "test" in own["output"]


@pytestmark_docker
def test_sandbox_is_removed_even_when_the_task_raises():
    """A run that dies mid-task must not leave a container holding memory."""
    from nite_eval.sandbox import _docker

    sb = SandboxToolEnv(SandboxSpec(image="python:3.12-alpine"))
    sb.start()
    container_id = sb.container_id
    try:
        raise RuntimeError("task blew up")
    except RuntimeError:
        sb.stop()

    assert _docker(["inspect", container_id], timeout=15).returncode != 0


def test_structured_arguments_are_serialised_not_crashed_on():
    """A model may pass `content` as an object rather than a string.

    It did, during the six-model baseline: the dict went into
    subprocess(input=...) and raised "'dict' object has no attribute 'encode'",
    killing the task.
    """
    env = SandboxToolEnv(SandboxSpec(image="python:3.12-alpine"), container_id="fake")
    captured = {}

    def fake_write(path, content):
        captured["path"], captured["content"] = path, content
        return {"status": "written"}

    env.write_file = fake_write  # type: ignore[method-assign]
    env.call("write_file", {"path": "/a.json", "content": {"servers": [{"name": "notion"}]}})

    assert isinstance(captured["content"], str)
    assert '"name": "notion"' in captured["content"]


def test_non_string_scalars_are_coerced():
    env = SandboxToolEnv(SandboxSpec(image="python:3.12-alpine"), container_id="fake")
    assert env._as_text(42) == "42"
    assert env._as_text(None) == ""
    assert env._as_text("already text") == "already text"
    assert env._as_text(["a", "b"]).startswith("[")
