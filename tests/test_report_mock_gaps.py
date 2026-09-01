"""Fixture gaps have to reach the report, not just the run log.

A score cannot distinguish "the model answered badly" from "the fixture had no
answer to give". The report is where that distinction has to be made, because
that is what someone reads before saying one model beats another.
"""

import tempfile

from nite_eval.report import generate_report
from nite_eval.results_db import ResultsDB


def _db(unmatched_for_a: int) -> ResultsDB:
    db = ResultsDB(tempfile.mktemp(suffix=".db"))
    db.create_run("run-001", ["model-a", "model-b"])
    db.register_tasks("run-001", ["model-a", "model-b"], [("t1", "agentic", "medium")])
    for model, misses in (("model-a", unmatched_for_a), ("model-b", 0)):
        db.save_task_result(
            run_id="run-001",
            model_name=model,
            task_id="t1",
            final_response="r",
            total_turns=2,
            total_tool_calls=6,
            total_latency_ms=100.0,
            reached_max_turns=False,
            weighted_score=0.6,
            unmatched_mock_calls=misses,
        )
    return db


def test_mock_gaps_are_reported():
    with _db(5) as db:
        report = generate_report(db, "run-001", {"agentic": 1.0})
    assert "Unanswered Tool Calls" in report
    assert "model-a" in report
    assert "5" in report


def test_clean_run_says_nothing():
    """No section at all when every call was answered."""
    with _db(0) as db:
        report = generate_report(db, "run-001", {"agentic": 1.0})
    assert "Unanswered Tool Calls" not in report
