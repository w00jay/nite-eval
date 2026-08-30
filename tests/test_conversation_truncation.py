"""Truncation and unparsed-tool-call handling in the conversation runner.

Regression tests for the audit finding that a generation cut off at max_tokens
was indistinguishable from a completed one: the fragment became final_response
and was scored by the judge.
"""

from unittest.mock import patch

from nite_eval.conversation_runner import ModelReply, run_conversation
from nite_eval.mock_tools import MockToolEnv

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                "required": ["path", "content"],
            },
        },
    }
]
MOCKS = {"write_file": [{"match": {"path_contains": "any"}, "response": {"content": {"status": "written"}}}]}


def _run(replies):
    with patch("nite_eval.conversation_runner._call_model", side_effect=replies):
        return run_conversation(
            base_url="http://test",
            model_name="test-model",
            system_prompt="be helpful",
            tools=TOOLS,
            user_message="write a file",
            mock_env=MockToolEnv.from_task_yaml(MOCKS),
            max_turns=3,
            max_tokens=4096,
        )


def test_truncated_generation_fails_task_instead_of_being_judged():
    fragment = (
        'Let me write it.\n<tool_call>\n{"name": "write_file", "arguments": {"path": "/a.go", "content": "package'
    )
    result = _run([ModelReply(text=fragment, finish_reason="length")])

    assert result.error is not None
    assert "truncated" in result.error
    assert "max_tokens=4096" in result.error
    # The fragment must not reach the judge as if it were an answer.
    assert result.final_response == ""
    assert result.turns[0].truncated is True
    assert result.turns[0].finish_reason == "length"


# Missing opening quote on `arguments` — a real, reproducible qwen3.8 defect.
# The parser now repairs this, so it no longer reaches the retry path.
REPAIRABLE_CALL = '<tool_call>\n{"name": "write_file",\narguments": {"path": "/a.go", "content": "x"}}\n</tool_call>'
# Structurally broken beyond repair — no amount of quote fixing parses this.
UNREPAIRABLE_CALL = '<tool_call>\n{"name": "write_file", "arguments": {oops not json}}\n</tool_call>'
GOOD_CALL = '<tool_call>\n{"name": "write_file", "arguments": {"path": "/a.go", "content": "x"}}\n</tool_call>'


def test_dropped_key_quote_is_repaired_and_counted():
    """The qwen3.8 defect should salvage the call, not fail the task."""
    replies = [
        ModelReply(text=REPAIRABLE_CALL, finish_reason="stop"),
        ModelReply(text="Done — wrote /a.go.", finish_reason="stop"),
    ]
    result = _run(replies)

    assert result.error is None
    assert result.total_tool_calls == 1
    # The repair is recorded, not silently absorbed.
    assert result.repaired_tool_calls == 1


def test_unparsable_tool_call_gets_one_corrective_retry():
    """An unrepairable call should not kill a conversation that is progressing."""
    replies = [
        ModelReply(text=UNREPAIRABLE_CALL, finish_reason="stop"),
        ModelReply(text=GOOD_CALL, finish_reason="stop"),
        ModelReply(text="Done — wrote /a.go.", finish_reason="stop"),
    ]
    result = _run(replies)

    assert result.error is None
    assert result.final_response == "Done — wrote /a.go."
    assert result.total_tool_calls == 1


def test_repeated_unparsable_tool_calls_fail_the_task():
    """A model that cannot emit valid JSON after a retry is a real failure."""
    result = _run([ModelReply(text=UNREPAIRABLE_CALL, finish_reason="stop")] * 3)

    assert result.error is not None
    assert "unparsed_tool_call" in result.error
    assert "malformed_json" in result.error
    # The offending payload is retained for diagnosis.
    assert "oops" in result.error
    assert result.final_response == ""


def test_complete_response_still_succeeds():
    call = (
        '<tool_call>\n{"name": "write_file", "arguments": {"path": "/a.go", "content": "package main"}}\n</tool_call>'
    )
    result = _run(
        [ModelReply(text=call, finish_reason="stop"), ModelReply(text="Done — wrote /a.go.", finish_reason="stop")]
    )

    assert result.error is None
    assert result.final_response == "Done — wrote /a.go."
    assert result.total_tool_calls == 1
    assert result.turns[0].truncated is False


def test_finish_reason_recorded_on_normal_turns():
    result = _run([ModelReply(text="Here is my answer.", finish_reason="stop")])

    assert result.error is None
    assert result.turns[0].finish_reason == "stop"
    assert result.turns[0].truncated is False


def test_failed_synthesis_nudge_errors_the_task():
    """A nudge that fails leaves no synthesis to score.

    Previously this logged a warning and fell back to the best already-emitted
    turn, so a task with no real answer was scored on a mid-work fragment.
    """
    import httpx

    call = '<tool_call>\n{"name": "write_file", "arguments": {"path": "/a.go", "content": "x"}}\n</tool_call>'
    request = httpx.Request("POST", "http://test/v1/chat/completions")
    response = httpx.Response(400, text="context size exceeded", request=request)
    http_error = httpx.HTTPStatusError("400", request=request, response=response)

    # Every turn emits a tool call, so max_turns is reached and the nudge fires.
    replies = [ModelReply(text=call, finish_reason="stop")] * 3 + [http_error]
    with patch("nite_eval.conversation_runner._call_model", side_effect=replies):
        result = run_conversation(
            base_url="http://test",
            model_name="test-model",
            system_prompt="be helpful",
            tools=TOOLS,
            user_message="write a file",
            mock_env=MockToolEnv.from_task_yaml(MOCKS),
            max_turns=3,
            max_tokens=4096,
        )

    assert result.error is not None
    assert "synthesis nudge failed with HTTP 400" in result.error
    assert result.final_response == ""


def test_task_wall_clock_budget_is_enforced():
    """timeout_seconds used to be accepted and discarded (noqa: ARG001)."""
    call = '<tool_call>\n{"name": "write_file", "arguments": {"path": "/a.go", "content": "x"}}\n</tool_call>'

    def slow_reply(*_args, **_kwargs):
        import time as _t

        _t.sleep(0.05)
        return ModelReply(text=call, finish_reason="stop")

    with patch("nite_eval.conversation_runner._call_model", side_effect=slow_reply):
        result = run_conversation(
            base_url="http://test",
            model_name="test-model",
            system_prompt="be helpful",
            tools=TOOLS,
            user_message="write a file",
            mock_env=MockToolEnv.from_task_yaml(MOCKS),
            max_turns=50,
            timeout_seconds=0.1,
            max_tokens=4096,
        )

    assert result.error is not None
    assert "task_timeout" in result.error
    assert "exceeded budget of 0s" in result.error
    # Stopped well before max_turns.
    assert len(result.turns) < 50


def test_generous_budget_does_not_interrupt():
    call = '<tool_call>\n{"name": "write_file", "arguments": {"path": "/a.go", "content": "x"}}\n</tool_call>'
    replies = [ModelReply(text=call, finish_reason="stop"), ModelReply(text="Done.", finish_reason="stop")]
    with patch("nite_eval.conversation_runner._call_model", side_effect=replies):
        result = run_conversation(
            base_url="http://test",
            model_name="test-model",
            system_prompt="be helpful",
            tools=TOOLS,
            user_message="write a file",
            mock_env=MockToolEnv.from_task_yaml(MOCKS),
            max_turns=5,
            timeout_seconds=600,
            max_tokens=4096,
        )
    assert result.error is None
    assert result.final_response == "Done."


def test_truncated_synthesis_nudge_fails_the_task():
    """The nudge produces the final answer, so a truncated nudge is no answer.

    The main loop checked finish_reason and the nudge path did not, so any task
    that reached its turn cap could still have a truncated response judged.
    """
    call = '<tool_call>\n{"name": "write_file", "arguments": {"path": "/a.go", "content": "x"}}\n</tool_call>'
    replies = [ModelReply(text=call, finish_reason="stop")] * 3 + [
        ModelReply(text="A very long partial answer", finish_reason="length")
    ]
    result = _run(replies)

    assert result.error is not None
    assert "truncated" in result.error
    assert "synthesis nudge" in result.error
    assert result.final_response == ""


# --- history compaction ---


def test_large_tool_call_is_summarised_in_history():
    """98% of coding_mcp_hard_01's context was write_file bodies, carried twice."""
    from nite_eval.conversation_runner import compact_tool_call_payloads
    from nite_eval.hermes_parser import extract_tool_calls

    body = "package main\n" + ("// filler line\n" * 400)
    raw = f'<tool_call>\n{{"name": "write_file", "arguments": {{"path": "/a.go", "content": {body!r}}}}}\n</tool_call>'
    raw = raw.replace("'", '"')
    text = "Writing the gateway now.\n" + raw
    parsed = extract_tool_calls(text)
    assert parsed.tool_calls, "fixture should parse"

    compacted = compact_tool_call_payloads(text, parsed, threshold=1500)

    assert len(compacted) < len(text) / 4
    assert "filler line" not in compacted
    assert "write_file issued and executed" in compacted
    assert "path=/a.go" in compacted
    # Prose outside the call is preserved.
    assert "Writing the gateway now." in compacted


def test_small_tool_calls_are_left_alone():
    from nite_eval.conversation_runner import compact_tool_call_payloads
    from nite_eval.hermes_parser import extract_tool_calls

    text = '<tool_call>\n{"name": "search", "arguments": {"query": "mcp gateway"}}\n</tool_call>'
    parsed = extract_tool_calls(text)
    assert compact_tool_call_payloads(text, parsed, threshold=1500) == text


def test_compaction_is_a_no_op_without_tool_calls():
    from nite_eval.conversation_runner import compact_tool_call_payloads
    from nite_eval.hermes_parser import extract_tool_calls

    text = "Here is my final answer, at some length. " * 100
    assert compact_tool_call_payloads(text, extract_tool_calls(text), threshold=10) == text


def test_executed_call_still_receives_full_content():
    """Compaction must affect history only, never what the tool actually runs."""
    big = "x" * 5000
    call = f'<tool_call>\n{{"name": "write_file", "arguments": {{"path": "/a.go", "content": "{big}"}}}}\n</tool_call>'
    replies = [ModelReply(text=call, finish_reason="stop"), ModelReply(text="Done.", finish_reason="stop")]

    captured = {}
    env = MockToolEnv.from_task_yaml(MOCKS)
    original = env.call

    def spy(name, arguments):
        captured[name] = arguments
        return original(name, arguments)

    env.call = spy
    with patch("nite_eval.conversation_runner._call_model", side_effect=replies):
        result = run_conversation(
            base_url="http://test",
            model_name="test-model",
            system_prompt="be helpful",
            tools=TOOLS,
            user_message="write a file",
            mock_env=env,
            max_turns=3,
            max_tokens=4096,
        )

    assert result.error is None
    assert len(captured["write_file"]["content"]) == 5000
