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
    assert spec.memory == "1g"

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
