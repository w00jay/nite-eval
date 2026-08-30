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

# Which tests failed, per runner. A partial score says how many failed; the
# names say which, and that is what makes it actionable.
FAILED_NAME_RES = {
    "go": re.compile(r"^\s*--- FAIL: (\S+)", re.MULTILINE),
    "python": re.compile(r"^(?:FAILED|ERROR) (\S+)", re.MULTILINE),
    "typescript": re.compile(r"^(.+?) (?:\.{3} )?FAILED", re.MULTILINE),
}
FAILED_NAME_RES["javascript"] = FAILED_NAME_RES["typescript"]
FAILED_NAME_RES["deno"] = FAILED_NAME_RES["typescript"]


# What a detector actually printing a finding looks like, so a check that fails
# for an unrelated reason can be told apart from one that found something.
DETECTOR_SIGNALS = {
    "race_detector_clean": ("WARNING: DATA RACE", "race detected"),
    "go_vet_clean": ("vet: ", ".go:"),
}


def _failed_test_names(language: str, output: str) -> list[str]:
    pattern = FAILED_NAME_RES.get(language.lower())
    if not pattern:
        return []
    seen: list[str] = []
    for name in pattern.findall(output):
        cleaned = name.strip()
        if cleaned and cleaned not in seen:
            seen.append(cleaned)
    return seen[:20]


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
        details: dict[str, object] = {"passed": passed, "total": total, "exit_code": exit_code}
        if passed < total:
            # A partial score alone is not actionable: artemis scored 13/14 and
            # nothing recorded which test failed. Keep the names, and a slice of
            # output for the assertion behind them.
            details["failed_tests"] = _failed_test_names(language, output)
            details["output"] = output[-4000:]
        return passed / total, details

    # No parseable results. Compilation failure, harness error, or an unknown
    # runner — all of which mean nothing was demonstrated to work.
    #
    # The output is retained here because a 0.0 from this branch is ambiguous:
    # the model's code may be wrong, or it may be correct but not match the
    # contract the hidden suite compiles against. Without the compiler's message
    # the two are indistinguishable after the sandbox is gone.
    return (1.0 if exit_code == 0 else 0.0), {
        "passed": None,
        "total": None,
        "exit_code": exit_code,
        "note": "no per-test results parsed; scored from exit code",
        "output": output[-4000:],
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
            details["scoring_command"] = outcome.get("scoring_command", "")
            results[name] = (score, details)
            continue

        command = checks.get(name)
        if not command:
            logger.warning("No command configured for automated criterion %s/%s", task_id, name)
            continue

        exec_result = sandbox.exec(command)
        output = exec_result.stdout or exec_result.stderr

        # A check that runs the test suite (`go test -race`) fails for two very
        # different reasons: the thing it checks for, or the tests failing. Only
        # the first is what the criterion measures. Without separating them, a
        # single unrelated test failure scores the race detector 0 and reads as
        # a data race.
        score = 1.0 if exec_result.exit_code == 0 else 0.0
        details = {
            "command": command,
            "exit_code": exec_result.exit_code,
            "output": output[:2000],
            "timed_out": exec_result.timed_out,
        }
        if exec_result.exit_code != 0:
            signals = DETECTOR_SIGNALS.get(name, ())
            if signals:
                hit = [sig for sig in signals if sig in output]
                details["detector_fired"] = bool(hit)
                details["matched"] = hit
                if not hit:
                    details["note"] = (
                        f"{name} command failed but its detector never fired; "
                        "scored as inconclusive rather than as a positive finding"
                    )
                    score = 0.0
        results[name] = (score, details)

    return results
