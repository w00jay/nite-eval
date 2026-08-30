"""Automated scoring from real test runs.

Replaces the hardcoded 0.0 that `method: automated` returned at 40-70% of each
coding task's weight.
"""

from pathlib import Path
from unittest.mock import MagicMock

from nite_eval.automated_scoring import parse_test_output, run_automated_checks

GO_OUTPUT = """=== RUN   TestHiddenLoadValidConfig
--- PASS: TestHiddenLoadValidConfig (0.00s)
=== RUN   TestHiddenInvalidURLRejected
--- FAIL: TestHiddenInvalidURLRejected (0.00s)
    config_hidden_test.go:60: expected a validation error
=== RUN   TestHiddenEnabledServersFilters
--- PASS: TestHiddenEnabledServersFilters (0.00s)
FAIL
"""


def test_go_partial_credit():
    """A suite where 2 of 3 pass must not score 0 just because it exited 1."""
    score, details = parse_test_output("go", GO_OUTPUT, exit_code=1)
    assert score == 2 / 3
    assert details["passed"] == 2
    assert details["total"] == 3


def test_pytest_counts():
    score, details = parse_test_output("python", "== 3 passed, 1 failed in 0.12s ==", exit_code=1)
    assert score == 0.75
    assert details["passed"] == 3


def test_js_counts():
    score, _ = parse_test_output("typescript", "Tests:  1 failed, 3 passed, 4 total", exit_code=1)
    assert score == 0.75


def test_compilation_failure_scores_zero():
    """No per-test lines and a non-zero exit means nothing was shown to work."""
    score, details = parse_test_output("go", "config.go:8:2: undefined: yaml", exit_code=2)
    assert score == 0.0
    assert details["passed"] is None
    assert "exit code" in details["note"]


def test_unparseable_but_successful_run_is_trusted():
    score, _ = parse_test_output("go", "ok  \tmcpconfig\t0.003s", exit_code=0)
    assert score == 1.0


def test_unknown_language_falls_back_to_exit_code():
    assert parse_test_output("rust", "test result: ok", exit_code=0)[0] == 1.0
    assert parse_test_output("rust", "test result: FAILED", exit_code=1)[0] == 0.0


def test_missing_hidden_suite_is_omitted_not_scored():
    """An absent suite must leave the criterion unscored, never invent a 0."""
    sandbox = MagicMock()
    sandbox.run_hidden_suite.return_value = {"error": "hidden suite not found: /nope"}
    results = run_automated_checks(
        sandbox=sandbox,
        task_id="coding_x",
        scoring={"test_pass_rate": {"method": "automated", "weight": 0.5}},
        suite_root=Path("/nope"),
        language="go",
        checks={},
    )
    assert results == {}


def test_check_without_a_command_is_omitted():
    sandbox = MagicMock()
    results = run_automated_checks(
        sandbox=sandbox,
        task_id="coding_x",
        scoring={"race_detector_clean": {"method": "automated", "weight": 0.15}},
        suite_root=Path("/tmp"),
        language="go",
        checks={},
    )
    assert results == {}
    sandbox.exec.assert_not_called()


def test_configured_check_runs_and_scores_binary():
    sandbox = MagicMock()
    sandbox.exec.return_value = MagicMock(exit_code=0, stdout="", stderr="", timed_out=False)
    results = run_automated_checks(
        sandbox=sandbox,
        task_id="coding_x",
        scoring={"go_vet_clean": {"method": "automated", "weight": 0.2}},
        suite_root=Path("/tmp"),
        language="go",
        checks={"go_vet_clean": "go vet ./..."},
    )
    assert results["go_vet_clean"][0] == 1.0
    sandbox.exec.assert_called_once_with("go vet ./...")

    sandbox.exec.return_value = MagicMock(exit_code=1, stdout="vet: bad", stderr="", timed_out=False)
    results = run_automated_checks(
        sandbox=sandbox,
        task_id="coding_x",
        scoring={"go_vet_clean": {"method": "automated", "weight": 0.2}},
        suite_root=Path("/tmp"),
        language="go",
        checks={"go_vet_clean": "go vet ./..."},
    )
    assert results["go_vet_clean"][0] == 0.0


def test_non_automated_criteria_are_ignored():
    sandbox = MagicMock()
    results = run_automated_checks(
        sandbox=sandbox,
        task_id="coding_x",
        scoring={"code_quality": {"method": "judge_rubric", "weight": 0.3}},
        suite_root=Path("/tmp"),
        language="go",
        checks={},
    )
    assert results == {}
    sandbox.run_hidden_suite.assert_not_called()


def test_unparseable_output_is_retained_for_diagnosis():
    """A 0.0 from the fallback is ambiguous without the compiler's message.

    coding_mcp_hard_01 scored test_pass_rate 0.0 with exit_code 1 and nothing
    else, leaving no way to tell a wrong implementation from one that simply
    did not match the contract the hidden suite compiles against.
    """
    _, details = parse_test_output("go", "config.go:12:6: undefined: Gateway", exit_code=2)
    assert details["output"] == "config.go:12:6: undefined: Gateway"


def test_retained_output_is_bounded():
    _, details = parse_test_output("go", "x" * 20000, exit_code=1)
    assert len(details["output"]) == 4000


def test_partial_failure_records_which_tests_failed():
    """13/14 tells you one failed; the name tells you which."""
    score, details = parse_test_output("go", GO_OUTPUT, exit_code=1)
    assert score == 2 / 3
    assert details["failed_tests"] == ["TestHiddenInvalidURLRejected"]
    assert "expected a validation error" in details["output"]


def test_pytest_failure_names_are_captured():
    output = (
        "FAILED test_hidden_horizons.py::test_midpoint_is_linearly_interpolated - assert 1 == 2\n"
        "== 13 passed, 1 failed in 0.4s ==\n"
    )
    score, details = parse_test_output("python", output, exit_code=1)
    assert score == 13 / 14
    assert details["failed_tests"] == ["test_hidden_horizons.py::test_midpoint_is_linearly_interpolated"]


def test_a_full_pass_carries_no_failure_noise():
    _, details = parse_test_output("go", "--- PASS: TestA (0.00s)\n--- PASS: TestB (0.00s)\nPASS\n", exit_code=0)
    assert details["passed"] == 2
    assert "failed_tests" not in details
    assert "output" not in details


def test_race_failure_records_whether_the_detector_actually_fired():
    """`go test -race` failing is not the same as a data race being found."""
    sandbox = MagicMock()
    sandbox.exec.return_value = MagicMock(
        exit_code=1, stdout="--- FAIL: TestSomethingUnrelated\nFAIL", stderr="", timed_out=False
    )
    results = run_automated_checks(
        sandbox=sandbox,
        task_id="coding_x",
        scoring={"race_detector_clean": {"method": "automated", "weight": 0.15}},
        suite_root=Path("/tmp"),
        language="go",
        checks={"race_detector_clean": "go test -race ."},
    )
    _, details = results["race_detector_clean"]
    assert details["detector_fired"] is False
    assert "detector never fired" in details["note"]


def test_a_real_data_race_is_recorded_as_one():
    sandbox = MagicMock()
    sandbox.exec.return_value = MagicMock(
        exit_code=66, stdout="WARNING: DATA RACE\nWrite at 0x00c000 by goroutine 7:", stderr="", timed_out=False
    )
    results = run_automated_checks(
        sandbox=sandbox,
        task_id="coding_x",
        scoring={"race_detector_clean": {"method": "automated", "weight": 0.15}},
        suite_root=Path("/tmp"),
        language="go",
        checks={"race_detector_clean": "go test -race ."},
    )
    score, details = results["race_detector_clean"]
    assert score == 0.0
    assert details["detector_fired"] is True
    assert "WARNING: DATA RACE" in details["matched"]


def test_a_clean_detector_run_needs_no_extra_fields():
    sandbox = MagicMock()
    sandbox.exec.return_value = MagicMock(exit_code=0, stdout="ok", stderr="", timed_out=False)
    results = run_automated_checks(
        sandbox=sandbox,
        task_id="coding_x",
        scoring={"race_detector_clean": {"method": "automated", "weight": 0.15}},
        suite_root=Path("/tmp"),
        language="go",
        checks={"race_detector_clean": "go test -race ."},
    )
    score, details = results["race_detector_clean"]
    assert score == 1.0
    assert "detector_fired" not in details
