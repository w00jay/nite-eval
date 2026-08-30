"""Score coding tasks by running their code.

`method: automated` criteria (`test_pass_rate`, `go_vet_clean`,
`race_detector_clean`) returned a hardcoded 0.0 at 40-70% of each coding task's
weight, so a coding score could not respond to the code the model wrote. This
runs the checks for real inside the task's sandbox.

Test counts are parsed from the toolchain's own output rather than trusting an
exit code, so a suite where 3 of 4 tests pass scores 0.75 instead of 0.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from nite_eval.sandbox import SandboxToolEnv

logger = logging.getLogger(__name__)

# `go test -v` emits one --- PASS:/--- FAIL: line per test.
GO_TEST_RESULT_RE = re.compile(r"^\s*--- (PASS|FAIL|SKIP): (\S+)", re.MULTILINE)
# pytest's terminal summary: "3 passed, 1 failed in 0.12s"
PYTEST_COUNT_RE = re.compile(r"(\d+) (passed|failed|error|errors)")
# node:test / jest style "Tests:  1 failed, 3 passed", and `deno test`'s
# "ok | 11 passed | 1 failed".
JS_COUNT_RE = re.compile(r"(\d+) (passing|failing|passed|failed)")


def _parse_go(output: str) -> tuple[int, int]:
    results = GO_TEST_RESULT_RE.findall(output)
    passed = sum(1 for kind, _ in results if kind == "PASS")
    total = sum(1 for kind, _ in results if kind in ("PASS", "FAIL"))
    return passed, total


def _parse_pytest(output: str) -> tuple[int, int]:
    passed = failed = 0
    for count, kind in PYTEST_COUNT_RE.findall(output):
        if kind == "passed":
            passed += int(count)
        else:
            failed += int(count)
    return passed, passed + failed


def _parse_js(output: str) -> tuple[int, int]:
    passed = failed = 0
    for count, kind in JS_COUNT_RE.findall(output):
        if kind in ("passing", "passed"):
            passed += int(count)
        else:
            failed += int(count)
    return passed, passed + failed


def parse_test_output(language: str, output: str, exit_code: int) -> tuple[float, dict]:
    """Return (pass_fraction, details) from a test runner's output.

    Falls back to the exit code when no counts can be parsed — a suite that
    fails to compile produces no per-test lines, and that is a genuine 0.
    """
    parsers = {
        "go": _parse_go,
        "python": _parse_pytest,
        "typescript": _parse_js,
        "javascript": _parse_js,
        "deno": _parse_js,
    }
    parser = parsers.get(language.lower())

    passed, total = parser(output) if parser else (0, 0)
    if total:
        return passed / total, {"passed": passed, "total": total, "exit_code": exit_code}

    # No parseable results. Compilation failure, harness error, or an unknown
    # runner — all of which mean nothing was demonstrated to work.
    return (1.0 if exit_code == 0 else 0.0), {
        "passed": None,
        "total": None,
        "exit_code": exit_code,
        "note": "no per-test results parsed; scored from exit code",
    }


def run_automated_checks(
    sandbox: SandboxToolEnv,
    task_id: str,
    scoring: dict,
    suite_root: Path,
    language: str,
    checks: dict[str, str],
) -> dict[str, tuple[float, dict]]:
    """Run every `automated` criterion the task declares.

    `checks` maps a criterion name to the shell command that decides it, e.g.
    `{"go_vet_clean": "go vet ./..."}`. `test_pass_rate` is special-cased: it
    installs the hidden suite and parses per-test results.

    Returns {criterion: (score, details)}. A criterion with no configured
    command is omitted, so the caller excludes it rather than inventing a score.
    """
    results: dict[str, tuple[float, dict]] = {}
    automated = [name for name, cfg in scoring.items() if cfg.get("method") == "automated"]
    if not automated:
        return results

    for name in automated:
        if name == "test_pass_rate":
            suite_dir = suite_root / task_id
            outcome = sandbox.run_hidden_suite(suite_dir)
            if "error" in outcome:
                logger.warning("Hidden suite for %s unavailable: %s", task_id, outcome["error"])
                continue
            score, details = parse_test_output(language, outcome.get("output", ""), outcome.get("exit_code", 1))
            details["hidden_files"] = outcome.get("hidden_files", [])
            results[name] = (score, details)
            continue

        command = checks.get(name)
        if not command:
            logger.warning("No command configured for automated criterion %s/%s", task_id, name)
            continue

        exec_result = sandbox.exec(command)
        results[name] = (
            1.0 if exec_result.exit_code == 0 else 0.0,
            {
                "command": command,
                "exit_code": exec_result.exit_code,
                "output": (exec_result.stdout or exec_result.stderr)[:2000],
                "timed_out": exec_result.timed_out,
            },
        )

    return results
