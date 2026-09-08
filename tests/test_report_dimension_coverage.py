"""A dimension that never ran must not print as 0.00.

In run-20260906-181136 (`--dimension coding`) the summary table showed
Research, Planning and Agentic as 0.00 for every model and then a Composite
column, so qwen3.8's "Composite 0.78" was its coding score wearing a label that
implies four dimensions.

The arithmetic was never wrong: get_dimension_averages omits a dimension with
nothing terminal in it, and compute_composite renormalises over what remains.
Only the display conflated "did not run" with "ran and scored zero".
"""

import pathlib
import tempfile

from nite_eval.report import generate_report
from nite_eval.results_db import ResultsDB

WEIGHTS = {"research": 0.25, "planning": 0.25, "coding": 0.25, "agentic": 0.25}


def _db(tasks):
    db = ResultsDB(pathlib.Path(tempfile.mkdtemp()) / "t.db")
    db.create_run("run-x", ["m1"])
    db.register_tasks("run-x", ["m1"], [(t, d, "medium") for t, d, _ in tasks])
    for task_id, _dim, score in tasks:
        db.save_task_result(
            run_id="run-x",
            model_name="m1",
            task_id=task_id,
            final_response="r",
            total_turns=1,
            total_tool_calls=1,
            total_latency_ms=100.0,
            reached_max_turns=False,
            weighted_score=score,
            tools_declared=2,
        )
    return db


def test_dimension_that_never_ran_prints_a_dash_not_zero():
    db = _db([("coding_a", "coding", 0.78)])
    report = generate_report(db, "run-x", WEIGHTS)
    header_row = next(ln for ln in report.splitlines() if ln.startswith("| m1 |"))
    assert "0.00" not in header_row, header_row
    assert "—" in header_row, header_row
    db.close()


def test_dimension_that_ran_and_scored_zero_still_prints_zero():
    """A real zero and an absent dimension must stay distinguishable."""
    db = _db([("coding_a", "coding", 0.0)])
    report = generate_report(db, "run-x", WEIGHTS)
    header_row = next(ln for ln in report.splitlines() if ln.startswith("| m1 |"))
    assert "0.00" in header_row, header_row
    db.close()


def test_composite_says_which_dimensions_it_covers():
    """0.78 over one dimension must not read as a four-dimension composite."""
    db = _db([("coding_a", "coding", 0.78)])
    report = generate_report(db, "run-x", WEIGHTS)
    assert "coding" in report.lower()
    assert "1 of 4 dimensions" in report or "coding only" in report.lower()
    db.close()


def test_full_run_composite_carries_no_partial_caveat():
    db = _db([("r", "research", 0.8), ("p", "planning", 0.7), ("c", "coding", 0.6), ("a", "agentic", 0.9)])
    report = generate_report(db, "run-x", WEIGHTS)
    row = next(ln for ln in report.splitlines() if ln.startswith("| m1 |"))
    assert "—" not in row
    assert "of 4 dimensions" not in report
    db.close()
