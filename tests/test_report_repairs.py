"""Report surfacing of malformed tool-call repairs.

A repaired call is invisible in the scores by design — the parser salvages it,
so the model is credited with the work. That makes the repair count the only
place a model's malformed-JSON rate shows up, and it has to reach the report.
"""

import tempfile

from nite_eval.report import generate_report
from nite_eval.results_db import ResultsDB


def _db_with_result(repaired: int, tool_calls: int = 28, unscored: float = 0.0) -> ResultsDB:
    db = ResultsDB(tempfile.mktemp(suffix=".db"))
    db.create_run("run-001", ["model-a"])
    db.register_tasks("run-001", ["model-a"], [("coding_mcp_hard_01", "coding", "hard")])
    db.mark_task_running("run-001", "model-a", "coding_mcp_hard_01")
    db.save_task_result(
        run_id="run-001",
        model_name="model-a",
        task_id="coding_mcp_hard_01",
        final_response="done",
        total_turns=29,
        total_tool_calls=tool_calls,
        total_latency_ms=736634,
        reached_max_turns=True,
        weighted_score=0.5,
        repaired_tool_calls=repaired,
        unscored_weight=unscored,
    )
    return db


def test_repair_rate_appears_in_report():
    with _db_with_result(repaired=23, tool_calls=28) as db:
        report = generate_report(db, "run-001")

    assert "Malformed Tool Calls" in report
    assert "| model-a | 23 | 28 | 82% |" in report


def test_no_repair_section_when_nothing_was_repaired():
    """Don't clutter reports for models that emit valid JSON."""
    with _db_with_result(repaired=0) as db:
        report = generate_report(db, "run-001")

    assert "Malformed Tool Calls" not in report


def test_per_task_table_has_repair_column():
    with _db_with_result(repaired=23) as db:
        report = generate_report(db, "run-001")

    assert "| Task | Diff | model-a | Turns | TCs | Rep | Unscored |" in report
    # 29 turns, 28 tool calls, 23 repaired, 0% unscored
    assert "| 29 | 28 | 23 | 0% |" in report


def test_legacy_rows_without_the_column_default_to_zero():
    """Historical rows predate repaired_tool_calls; they must not break reports."""
    db = ResultsDB(tempfile.mktemp(suffix=".db"))
    db.create_run("run-001", ["model-a"])
    db.register_tasks("run-001", ["model-a"], [("t1", "coding", "hard")])
    db.mark_task_running("run-001", "model-a", "t1")
    db.save_task_result(
        run_id="run-001",
        model_name="model-a",
        task_id="t1",
        final_response="x",
        total_turns=1,
        total_tool_calls=1,
        total_latency_ms=1.0,
        reached_max_turns=False,
        weighted_score=0.1,
    )
    # Simulate a pre-migration row.
    db._conn.execute("UPDATE task_results SET repaired_tool_calls = NULL")
    report = generate_report(db, "run-001")
    db.close()

    assert "Malformed Tool Calls" not in report
    assert "| 1 | 1 | 0 | 0% |" in report


def test_partial_scoring_is_flagged_in_the_report():
    """A score over half its criteria must not read like a complete one."""
    with _db_with_result(repaired=0, unscored=0.65) as db:
        report = generate_report(db, "run-001")

    assert "Partially Scored Dimensions" in report
    assert "not comparable" in report
    assert "| model-a | coding | 65% |" in report


def test_no_partial_section_when_everything_was_scored():
    with _db_with_result(repaired=0, unscored=0.0) as db:
        report = generate_report(db, "run-001")

    assert "Partially Scored Dimensions" not in report
