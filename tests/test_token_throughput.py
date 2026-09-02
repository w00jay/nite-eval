"""Token accounting, from the server's usage block to the report's tok/s.

Latency was the only timing the harness kept, which measures how long a model
took rather than how fast it generates. A MoE activating a fraction of its
weights and a dense model that simply says less are indistinguishable in
seconds-per-task and nothing alike in tokens per second.

The property these tests protect is that an unmeasured run stays unmeasured: a
missing usage block must reach the report as "—", never as a 0 that reads like
a measurement.
"""

import tempfile
from unittest.mock import MagicMock, patch

from nite_eval.conversation_runner import Message, ModelReply, _call_model, run_conversation
from nite_eval.report import generate_report
from nite_eval.results_db import ResultsDB


def _response(payload: dict) -> MagicMock:
    resp = MagicMock()
    resp.json.return_value = payload
    resp.raise_for_status.return_value = None
    return resp


def _call(payload: dict) -> ModelReply:
    client = MagicMock()
    client.post.return_value = _response(payload)
    return _call_model(client, "http://x", "m", [Message(role="user", content="hi")], 0.0, 128)


# --- _call_model reads usage ---


def test_usage_is_read_from_the_completion():
    reply = _call(
        {
            "choices": [{"message": {"content": "hello"}, "finish_reason": "stop"}],
            "usage": {"completion_tokens": 42, "prompt_tokens": 913, "total_tokens": 955},
        }
    )
    assert reply.completion_tokens == 42
    assert reply.prompt_tokens == 913


def test_missing_usage_block_is_zero_not_an_error():
    reply = _call({"choices": [{"message": {"content": "hello"}, "finish_reason": "stop"}]})
    assert reply.completion_tokens == 0
    assert reply.prompt_tokens == 0


def test_usage_survives_the_reasoning_content_fallback():
    """A thought-only answer is still generation and still costs tokens."""
    reply = _call(
        {
            "choices": [{"message": {"content": "", "reasoning_content": "thinking"}, "finish_reason": "stop"}],
            "usage": {"completion_tokens": 77, "prompt_tokens": 12},
        }
    )
    assert reply.text == "thinking"
    assert reply.completion_tokens == 77


# --- run_conversation sums them ---


def _run(replies: list[ModelReply]):
    env = MagicMock()
    env.call.return_value = {"ok": True}
    with patch("nite_eval.conversation_runner._call_model", side_effect=replies):
        return run_conversation(
            base_url="http://x",
            model_name="m",
            system_prompt="s",
            tools=[],
            user_message="u",
            mock_env=env,
            max_turns=4,
        )


def test_totals_sum_across_turns():
    result = _run(
        [
            ModelReply(
                text='<tool_call>{"name": "f", "arguments": {}}</tool_call>',
                finish_reason="stop",
                completion_tokens=10,
                prompt_tokens=100,
            ),
            ModelReply(text="Final answer.", finish_reason="stop", completion_tokens=5, prompt_tokens=150),
        ]
    )
    assert result.total_completion_tokens == 15
    assert result.total_prompt_tokens == 250


def test_totals_are_zero_when_the_server_reports_nothing():
    result = _run([ModelReply(text="Final answer.", finish_reason="stop")])
    assert result.total_completion_tokens == 0
    assert result.total_prompt_tokens == 0


# --- the report ---


def _db(rows: list[tuple[str, int | None, int | None, float]]) -> ResultsDB:
    """rows: (model, completion_tokens, prompt_tokens, latency_ms)."""
    db = ResultsDB(tempfile.mktemp(suffix=".db"))
    models = [r[0] for r in rows]
    db.create_run("run-001", models)
    db.register_tasks("run-001", models, [("research_a_easy_01", "research", "easy")])
    for model, gen, prompt, latency in rows:
        db.mark_task_running("run-001", model, "research_a_easy_01")
        db.save_task_result(
            run_id="run-001",
            model_name=model,
            task_id="research_a_easy_01",
            final_response="done",
            total_turns=2,
            total_tool_calls=1,
            total_latency_ms=latency,
            reached_max_turns=False,
            weighted_score=0.5,
            completion_tokens=gen,
            prompt_tokens=prompt,
        )
    return db


def test_report_computes_tokens_per_second():
    # 2000 generated tokens over 10s of wall clock.
    with _db([("fast-moe", 2000, 8000, 10_000.0)]) as db:
        report = generate_report(db, "run-001")
    assert "Latency and throughput" in report
    assert "| fast-moe | 10000 | 10 | 200.0 | 2000 | 8000 |" in report


def test_unmeasured_run_shows_a_dash_not_a_zero():
    """A NULL count is 'we do not know', and printing 0.0 tok/s would be a claim."""
    with _db([("legacy", None, None, 10_000.0)]) as db:
        report = generate_report(db, "run-001")
    assert "| legacy | 10000 | 10 | — | — | — |" in report
    assert "Token counts unavailable" in report


def test_unmeasured_tasks_do_not_dilute_a_measured_model():
    """The tok/s divisor must come from the same rows as the numerator.

    A run half-measured — a resumed run, or a server that stopped reporting
    usage partway — would otherwise divide real tokens by every task's latency
    and report a model as slower than it is.
    """
    db = ResultsDB(tempfile.mktemp(suffix=".db"))
    db.create_run("run-001", ["m"])
    tasks = [("research_a_easy_01", "research", "easy"), ("research_b_easy_01", "research", "easy")]
    db.register_tasks("run-001", ["m"], tasks)
    for task_id, gen in (("research_a_easy_01", 2000), ("research_b_easy_01", None)):
        db.mark_task_running("run-001", "m", task_id)
        db.save_task_result(
            run_id="run-001",
            model_name="m",
            task_id=task_id,
            final_response="done",
            total_turns=1,
            total_tool_calls=0,
            total_latency_ms=10_000.0,
            reached_max_turns=False,
            weighted_score=0.5,
            completion_tokens=gen,
            prompt_tokens=None,
        )
    with db:
        report = generate_report(db, "run-001")
    # 2000 / 10s, not 2000 / 20s.
    assert "| m | 10000 | 20 | 200.0 | 2000 | — |" in report
