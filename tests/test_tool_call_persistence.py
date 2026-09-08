"""Tool calls must be stored whatever the task's outcome.

`run_task` built its tool_calls rows after the failure branch had already
returned, so a failed task's calls were counted in `task_results.total_tool_calls`
and never written. Measured on the production DB: 32 failed coding rows counted
168 calls and stored 0, against 2195/2195 on the completed path.

The failing runs are the ones a determinism check most needs — a task that fails
in one run and completes in the next is the symptom being chased — so the trace
has to survive the failure, and the completed path must not change.
"""

import tempfile
from unittest.mock import MagicMock, patch

from nite_eval.conversation_runner import ConversationResult, TurnResult
from nite_eval.hermes_parser import ParsedResponse
from nite_eval.orchestrator import collect_tool_records, run_task
from nite_eval.results_db import ResultsDB
from nite_eval.task_loader import TaskDefinition

RUN_ID = "run-001"
MODEL = "model-a"
TASK_ID = "coding_mcp_hard_01"


def _db() -> ResultsDB:
    db = ResultsDB(tempfile.mktemp(suffix=".db"))
    db.create_run(RUN_ID, [MODEL])
    db.register_tasks(RUN_ID, [MODEL], [(TASK_ID, "coding", "hard")])
    return db


def _task() -> TaskDefinition:
    return TaskDefinition(
        id=TASK_ID,
        dimension="coding",
        difficulty="hard",
        description="",
        system_prompt="s",
        tools=[{"type": "function", "function": {"name": "write_file"}}],
        user_message="u",
        scoring={},
    )


def _turn(turn: int, tool_responses: list[dict]) -> TurnResult:
    return TurnResult(turn=turn, response="text", parsed=ParsedResponse(), tool_responses=tool_responses)


def _conv(turns: list[TurnResult], error: str | None = None) -> ConversationResult:
    return ConversationResult(
        turns=turns,
        final_response="" if error else "done",
        total_tool_calls=sum(len(t.tool_responses) for t in turns),
        total_latency_ms=1200.0,
        total_completion_tokens=10,
        total_prompt_tokens=100,
        total_predicted_ms=50.0,
        total_predicted_n=10,
        reached_max_turns=False,
        error=error,
    )


def _status(db: ResultsDB) -> str:
    return next(r["status"] for r in db.get_model_scores(RUN_ID, MODEL) if r["task_id"] == TASK_ID)


def _counted(db: ResultsDB) -> int:
    """task_results.total_tool_calls — what the stored rows are meant to match."""
    return db._conn.execute(
        "SELECT total_tool_calls FROM task_results WHERE run_id = ? AND model_name = ? AND task_id = ?",
        (RUN_ID, MODEL, TASK_ID),
    ).fetchone()[0]


def _run(db: ResultsDB, conv: ConversationResult) -> None:
    """Drive run_task with a canned conversation and no judge."""
    with (
        patch("nite_eval.orchestrator.run_conversation", return_value=conv),
        patch("nite_eval.orchestrator.score_task", return_value=([], 0.5, 0.0)),
    ):
        run_task(_task(), MODEL, "http://x", MagicMock(), db, RUN_ID, {})


# --- the bug ---


def test_failed_task_persists_its_tool_calls():
    conv = _conv(
        [
            _turn(1, [{"name": "read_file", "arguments": {"path": "a.go"}, "result": {"content": "package a"}}]),
            _turn(2, [{"name": "write_file", "arguments": {"path": "b.go"}, "result": {"ok": True}}]),
        ],
        error="degenerate_repetition (turn 3)",
    )
    with _db() as db:
        _run(db, conv)

        calls = db.get_tool_calls(RUN_ID, MODEL, TASK_ID)
        assert [c["name"] for c in calls] == ["read_file", "write_file"]
        assert calls[0]["arguments"] == {"path": "a.go"}


def test_failed_task_stores_as_many_calls_as_it_counted():
    """total_tool_calls and the stored rows must agree, the way they do on the
    completed path — that mismatch is how the loss was found."""
    turns = [_turn(1, [{"name": f"t{i}", "arguments": {}, "result": {}} for i in range(3)]), _turn(2, [])]
    conv = _conv(turns, error="task_timeout: 91s exceeded budget of 90s")
    with _db() as db:
        _run(db, conv)
        assert _status(db) == "failed"
        assert _counted(db) == 3
        assert len(db.get_tool_calls(RUN_ID, MODEL, TASK_ID)) == 3


def test_completed_task_still_persists_its_tool_calls():
    conv = _conv([_turn(1, [{"name": "web_search", "arguments": {"q": "x"}, "result": {"content": "hit"}}])])
    with _db() as db:
        _run(db, conv)
        assert _status(db) == "completed"
        calls = db.get_tool_calls(RUN_ID, MODEL, TASK_ID)
        assert len(calls) == 1
        assert calls[0]["name"] == "web_search"
        assert calls[0]["result"] == {"content": "hit"}


def test_trace_order_is_preserved_across_turns():
    """The determinism gate compares an ordered (turn, call_index) trace."""
    conv = _conv(
        [
            _turn(1, [{"name": "a", "arguments": {}, "result": {}}, {"name": "b", "arguments": {}, "result": {}}]),
            _turn(2, [{"name": "c", "arguments": {}, "result": {}}]),
        ],
        error="unparsed_tool_call: 1 on turn 3",
    )
    with _db() as db:
        _run(db, conv)
        calls = db.get_tool_calls(RUN_ID, MODEL, TASK_ID)
        assert [(c["turn"], c["call_index"], c["name"]) for c in calls] == [(1, 0, "a"), (1, 1, "b"), (2, 0, "c")]


def test_a_conversation_with_no_calls_writes_nothing():
    with _db() as db:
        _run(db, _conv([_turn(1, [])], error="no tool calls"))
        assert db.get_tool_calls(RUN_ID, MODEL, TASK_ID) == []


# --- partial data ---


def test_call_with_no_result_still_persists():
    """A failure can cut a call off before its result came back."""
    conv = _conv([_turn(1, [{"name": "run_tests", "arguments": {"path": "."}, "result": None}])], error="boom")
    with _db() as db:
        _run(db, conv)
        calls = db.get_tool_calls(RUN_ID, MODEL, TASK_ID)
        assert len(calls) == 1
        assert calls[0]["name"] == "run_tests"
        assert calls[0]["result"] == {}


def test_missing_result_and_arguments_keys_do_not_crash_the_write():
    conv = _conv([_turn(1, [{"name": "run_tests"}])], error="boom")
    with _db() as db:
        _run(db, conv)
        calls = db.get_tool_calls(RUN_ID, MODEL, TASK_ID)
        assert len(calls) == 1
        assert calls[0]["arguments"] == {}
        assert calls[0]["result"] == {}


def test_unserialisable_result_does_not_cost_the_whole_trace():
    conv = _conv(
        [
            _turn(1, [{"name": "read_file", "arguments": {}, "result": {"blob": object()}}]),
            _turn(2, [{"name": "write_file", "arguments": {}, "result": {"ok": True}}]),
        ],
        error="boom",
    )
    with _db() as db:
        _run(db, conv)
        assert [c["name"] for c in db.get_tool_calls(RUN_ID, MODEL, TASK_ID)] == ["read_file", "write_file"]


def test_nameless_record_is_dropped_without_losing_its_neighbours():
    """tool_name is NOT NULL and executemany aborts the batch on one bad row,
    so an unusable record must not take the rest of the task's trace with it."""
    with _db() as db:
        db.save_tool_calls(
            RUN_ID,
            MODEL,
            TASK_ID,
            [
                {"turn": 1, "call_index": 0, "tool_name": "read_file", "arguments": {}, "result": {}},
                {"turn": 1, "call_index": 1, "tool_name": None, "arguments": {}, "result": {}},
                {"turn": 2, "call_index": 0, "tool_name": "write_file", "arguments": {}, "result": {}},
            ],
        )
        assert [c["name"] for c in db.get_tool_calls(RUN_ID, MODEL, TASK_ID)] == ["read_file", "write_file"]


def test_record_missing_turn_is_dropped_not_raised():
    with _db() as db:
        db.save_tool_calls(RUN_ID, MODEL, TASK_ID, [{"call_index": 0, "tool_name": "read_file"}])
        assert db.get_tool_calls(RUN_ID, MODEL, TASK_ID) == []


# --- the flattening itself ---


def test_collect_tool_records_flattens_turns_in_order():
    conv = _conv(
        [
            _turn(1, [{"name": "a", "arguments": {"x": 1}, "result": {"ok": True}}]),
            _turn(2, [{"name": "b", "arguments": {}, "result": {}}, {"name": "c", "arguments": {}, "result": {}}]),
        ]
    )
    records = collect_tool_records(conv)
    assert [(r["turn"], r["call_index"], r["tool_name"]) for r in records] == [(1, 0, "a"), (2, 0, "b"), (2, 1, "c")]
    assert records[0]["arguments"] == {"x": 1}


def test_collect_tool_records_is_empty_for_a_conversation_that_never_ran():
    assert collect_tool_records(_conv([], error="connection refused")) == []
