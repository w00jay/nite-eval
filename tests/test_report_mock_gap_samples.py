"""Recorded arguments for calls the mocks could not answer.

The count alone told readers to check the fixture, which is right when a mock
was too narrow and actively misleading when the model emitted a call nothing
could match. Both were seen in run-20260901-043322: sparkling wine against a
seven-mock fixture with no catch-all was the fixture's fault, and gemma4
nesting `server` one level too deep inside call_mcp_tool was the model's.

Distinguishing them automatically is not possible, so the arguments are stored
and the report hands them to a person.
"""

import json
import tempfile

from nite_eval.mock_tools import MAX_UNMATCHED_SAMPLES, MockToolEnv, summarise_unmatched
from nite_eval.report import generate_report
from nite_eval.results_db import ResultsDB


def _env() -> MockToolEnv:
    return MockToolEnv.from_task_yaml(
        {"query_inventory": [{"match": {"filters": {"wine_type": "white"}}, "response": {"wines": []}}]}
    )


# --- capture ---


def test_nothing_recorded_when_every_call_matched():
    env = _env()
    env.call("query_inventory", {"filters": {"wine_type": "white"}})
    assert summarise_unmatched(env.unmatched_calls) is None


def test_unmatched_arguments_are_captured_verbatim():
    env = _env()
    env.call("query_inventory", {"filters": {"wine_type": "sparkling"}})
    payload = json.loads(summarise_unmatched(env.unmatched_calls))
    assert payload["total"] == 1
    assert payload["sample"][0]["name"] == "query_inventory"
    assert "sparkling" in payload["sample"][0]["arguments"]
    assert payload["sample"][0]["reason"] == "no_matching_mock"


def test_undeclared_tool_is_recorded_with_its_own_reason():
    """The two reasons need opposite fixes: broaden a mock, or drop the tool."""
    env = _env()
    env.call("fetch_url", {"url": "https://example.com"})
    payload = json.loads(summarise_unmatched(env.unmatched_calls))
    assert payload["sample"][0]["reason"] == "no_mock_for_tool"


def test_sample_is_bounded_but_the_total_is_not():
    """A model retrying one rejected call twenty times is one fact, not twenty."""
    env = _env()
    for _ in range(20):
        env.call("query_inventory", {"filters": {"wine_type": "sparkling"}})
    payload = json.loads(summarise_unmatched(env.unmatched_calls))
    assert payload["total"] == 20
    assert len(payload["sample"]) == MAX_UNMATCHED_SAMPLES


def test_oversized_arguments_are_truncated_and_flagged():
    env = _env()
    env.call("query_inventory", {"note": "x" * 5000})
    entry = json.loads(summarise_unmatched(env.unmatched_calls))["sample"][0]
    assert entry["truncated"] is True
    assert len(entry["arguments"]) < 5000


# --- report ---


def _db(count: int, samples: str | None) -> ResultsDB:
    db = ResultsDB(tempfile.mktemp(suffix=".db"))
    db.create_run("run-001", ["model-a"])
    db.register_tasks("run-001", ["model-a"], [("agentic_wine_medium_01", "agentic", "medium")])
    db.mark_task_running("run-001", "model-a", "agentic_wine_medium_01")
    db.save_task_result(
        run_id="run-001",
        model_name="model-a",
        task_id="agentic_wine_medium_01",
        final_response="done",
        total_turns=3,
        total_tool_calls=7,
        total_latency_ms=1000.0,
        reached_max_turns=False,
        weighted_score=0.5,
        unmatched_mock_calls=count,
        unmatched_mock_samples=samples,
    )
    return db


def test_report_prints_the_recorded_arguments():
    env = _env()
    env.call("query_inventory", {"filters": {"wine_type": "sparkling"}})
    with _db(1, summarise_unmatched(env.unmatched_calls)) as db:
        report = generate_report(db, "run-001")

    assert "Unanswered Tool Calls" in report
    assert "### Recorded arguments" in report
    assert "sparkling" in report
    assert "no_matching_mock" in report


def test_report_names_both_causes_not_just_the_fixture():
    """The old text said 'check the fixture', which is wrong half the time."""
    env = _env()
    env.call("query_inventory", {"filters": {"wine_type": "sparkling"}})
    with _db(1, summarise_unmatched(env.unmatched_calls)) as db:
        report = generate_report(db, "run-001")

    assert "The fixture was too narrow" in report
    assert "The model emitted a call nothing could match" in report


def test_older_run_says_arguments_are_unavailable():
    """A NULL sample predates the column; it does not mean there were none."""
    with _db(4, None) as db:
        report = generate_report(db, "run-001")

    assert "| model-a | agentic_wine_medium_01 | 4 |" in report
    assert "No arguments were recorded" in report
    assert "### Recorded arguments" not in report
