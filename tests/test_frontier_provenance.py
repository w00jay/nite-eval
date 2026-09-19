"""Provider/format/cost provenance: storage, migration, and how the report reads it."""

import sqlite3
import tempfile

from nite_eval.report import generate_report
from nite_eval.results_db import ResultsDB


def _db() -> ResultsDB:
    return ResultsDB(tempfile.mktemp(suffix=".db"))


def _record(db, run_id, model, *, provider, native, cost, score):
    db.save_task_result(
        run_id=run_id,
        model_name=model,
        task_id="agentic_t1",
        final_response="answer",
        total_turns=2,
        total_tool_calls=1,
        total_latency_ms=1000.0,
        reached_max_turns=False,
        weighted_score=score,
        provider=provider,
        native_tools=native,
        cost_usd=cost,
    )
    db.save_score(
        run_id=run_id,
        model_name=model,
        task_id="agentic_t1",
        dimension="agentic",
        method="subset_match",
        raw_score=score,
        normalized=score,
        weight=1.0,
    )


def _mixed_run() -> ResultsDB:
    db = _db()
    models = ["qwen3.6-35b-a3b", "claude-opus-5"]
    db.create_run("run-1", models)
    db.register_tasks("run-1", models, [("agentic_t1", "agentic", "easy")])
    _record(db, "run-1", "qwen3.6-35b-a3b", provider="local", native=False, cost=0.0, score=0.6)
    _record(db, "run-1", "claude-opus-5", provider="anthropic", native=True, cost=0.045, score=0.95)
    return db


def test_provenance_round_trips_through_the_summary():
    db = _mixed_run()
    summary = db.get_run_summary("run-1")
    assert summary["claude-opus-5"]["provider"] == "anthropic"
    assert summary["claude-opus-5"]["native_tools"] == 1
    assert summary["claude-opus-5"]["cost_usd"] == 0.045
    assert summary["qwen3.6-35b-a3b"]["native_tools"] == 0
    db.finish_run("run-1", total_cost_usd=0.045)
    assert db._conn.execute("SELECT total_cost_usd FROM eval_runs WHERE run_id='run-1'").fetchone()[0] == 0.045
    db.close()


def test_report_flags_mixed_tool_formats_as_a_confound():
    db = _mixed_run()
    report = generate_report(db, "run-1")
    assert "Comparability caveat" in report
    assert "hermes" in report and "native" in report
    db.close()


def test_report_shows_provider_and_cost():
    db = _mixed_run()
    report = generate_report(db, "run-1")
    assert "## Provider & Cost" in report
    assert "| claude-opus-5 | anthropic | native | $0.0450 |" in report
    assert "| qwen3.6-35b-a3b | local | hermes | free |" in report
    db.close()


def test_report_does_not_claim_a_format_for_pre_frontier_runs():
    """A run recorded before provenance existed is unknown, not local."""
    db = _db()
    db.create_run("run-old", ["qwen3.6-35b-a3b"])
    db.register_tasks("run-old", ["qwen3.6-35b-a3b"], [("agentic_t1", "agentic", "easy")])
    db.save_task_result(
        run_id="run-old",
        model_name="qwen3.6-35b-a3b",
        task_id="agentic_t1",
        final_response="x",
        total_turns=1,
        total_tool_calls=0,
        total_latency_ms=100.0,
        reached_max_turns=False,
        weighted_score=0.5,
    )
    report = generate_report(db, "run-old")
    assert "## Provider & Cost" not in report
    assert "Comparability caveat" not in report
    assert "| qwen3.6-35b-a3b | ? |" in report
    db.close()


def test_database_without_provenance_columns_migrates_in_place():
    """An existing results DB gains the columns without losing its history."""
    path = tempfile.mktemp(suffix=".db")
    with ResultsDB(path) as db:
        db.create_run("run-old", ["qwen3.6-35b-a3b"])
        db.register_tasks("run-old", ["qwen3.6-35b-a3b"], [("agentic_t1", "agentic", "easy")])
        _record(db, "run-old", "qwen3.6-35b-a3b", provider=None, native=None, cost=None, score=0.75)

    # Simulate the pre-frontier schema by dropping the new columns
    conn = sqlite3.connect(path)
    for column in ("provider", "native_tools", "cost_usd"):
        conn.execute(f"ALTER TABLE task_results DROP COLUMN {column}")
    conn.execute("ALTER TABLE eval_runs DROP COLUMN total_cost_usd")
    conn.commit()
    conn.close()

    with ResultsDB(path) as db:
        columns = {r[1] for r in db._conn.execute("PRAGMA table_info(task_results)")}
        assert {"provider", "native_tools", "cost_usd"} <= columns
        assert "total_cost_usd" in {r[1] for r in db._conn.execute("PRAGMA table_info(eval_runs)")}
        summary = db.get_run_summary("run-old")["qwen3.6-35b-a3b"]
        assert summary["avg_score"] == 0.75  # history intact
        assert summary["provider"] is None  # and honestly unknown
