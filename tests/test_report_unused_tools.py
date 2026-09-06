"""A task that offered tools and used none is a measurement, not a result.

`coding_wine_medium_01` in run-20260905-235130: 1 turn, 0 tool calls, 1311ms,
a 22-character response, and it still scored 0.45. The existing "Tasks That
Produced No Answer" section detects the opposite failure — every turn ending in
a tool call and no answer ever emitted — so nothing covered this.

The broader case is not a failure at all but a limit on what the score means:
across every run since the 2026-08-30 boundary, planning tasks completed with
zero tool calls 22.9% of the time and scored 0.75 against 0.76 for the ones
that used tools. Those numbers cannot distinguish grounded work from recalled
work, and the report should say so rather than leaving a reader to assume the
tools were exercised.
"""

from nite_eval.report import generate_report
from nite_eval.results_db import ResultsDB


def _db(tmp_path):
    db = ResultsDB(tmp_path / "t.db")
    db.create_run("run-x", ["m1"])
    return db


def _task(db, task_id, dimension, *, turns, tool_calls, tools_declared, score, response="ok"):
    db.register_tasks("run-x", ["m1"], [(task_id, dimension, "medium")])
    db.save_task_result(
        run_id="run-x",
        model_name="m1",
        task_id=task_id,
        final_response=response,
        total_turns=turns,
        total_tool_calls=tool_calls,
        total_latency_ms=1000.0,
        reached_max_turns=False,
        weighted_score=score,
        tools_declared=tools_declared,
    )


def test_task_that_declared_tools_and_used_none_is_flagged(tmp_path):
    db = _db(tmp_path)
    _task(
        db,
        "coding_wine_medium_01",
        "coding",
        turns=1,
        tool_calls=0,
        tools_declared=4,
        score=0.45,
        response="/app/scanlabel/scan_label.ts",
    )
    report = generate_report(db, "run-x")
    assert "Declared Tools, Used None" in report
    assert "coding_wine_medium_01" in report
    db.close()


def test_task_that_used_its_tools_is_not_flagged(tmp_path):
    db = _db(tmp_path)
    _task(db, "coding_mcp_easy_01", "coding", turns=8, tool_calls=10, tools_declared=3, score=0.9)
    report = generate_report(db, "run-x")
    assert "Declared Tools, Used None" not in report
    db.close()


def test_task_with_no_tools_on_offer_is_not_flagged(tmp_path):
    """Using no tools is only notable when there were tools to use."""
    db = _db(tmp_path)
    _task(db, "prose_task", "research", turns=1, tool_calls=0, tools_declared=0, score=0.8)
    report = generate_report(db, "run-x")
    assert "Declared Tools, Used None" not in report
    db.close()


def test_unknown_tool_count_does_not_flag(tmp_path):
    """Runs predating the column must not be retro-flagged on a NULL."""
    db = _db(tmp_path)
    _task(db, "old_task", "planning", turns=1, tool_calls=0, tools_declared=None, score=0.76)
    report = generate_report(db, "run-x")
    assert "Declared Tools, Used None" not in report
    db.close()


def test_section_reports_the_response_length(tmp_path):
    """A 22-char answer and a 4593-char answer are different problems."""
    db = _db(tmp_path)
    _task(
        db,
        "coding_wine_medium_01",
        "coding",
        turns=1,
        tool_calls=0,
        tools_declared=4,
        score=0.45,
        response="/app/scanlabel/scan_label.ts",
    )
    report = generate_report(db, "run-x")
    assert "28" in report  # len("/app/scanlabel/scan_label.ts") == 28
    db.close()
