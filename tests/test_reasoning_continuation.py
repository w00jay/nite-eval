"""Reasoning that runs out of budget gets another turn, rather than failing.

ornith-1.5 reasons at length before acting. On coding_artemis_medium_01 it spent
the entire 32768-token budget on turn 1 producing 121k chars of deliberation and
zero tool calls, and the task failed — even though the reasoning was sound and
the model would plausibly have called a tool next. Raising max_tokens only bought
more deliberation: 89k chars at 24576, 121k at 32768, same turn, still no call.

That the reasoning is worth keeping is measurable: coding_mcp_easy_01 is the one
coding task Ornith finished with thinking on, and it scored 0.89 there against
0.25 with thinking off, where it wrote fluent code that failed to compile.

So a turn that is pure prose, cut off at the budget, is continued instead of
ending the task. A turn cut off mid-tool-call is not — that call is unusable and
resuming risks a duplicate. Degenerate repetition still fails first and fast.
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
GOOD_CALL = '<tool_call>\n{"name": "write_file", "arguments": {"path": "/a.go", "content": "x"}}\n</tool_call>'
REASONING = "Let me think about this. " * 200


def _run(replies, continuations=0, max_turns=6):
    with patch("nite_eval.conversation_runner._call_model", side_effect=replies):
        return run_conversation(
            base_url="http://test",
            model_name="test-model",
            system_prompt="be helpful",
            tools=TOOLS,
            user_message="write a file",
            mock_env=MockToolEnv.from_task_yaml(MOCKS),
            max_turns=max_turns,
            max_tokens=4096,
            reasoning_continuations=continuations,
        )


def test_disabled_by_default_truncation_still_fails():
    """The other six models must keep today's semantics untouched."""
    result = _run([ModelReply(text=REASONING, finish_reason="length")])
    assert result.error is not None and "truncated" in result.error
    assert result.final_response == ""


def test_pure_reasoning_truncation_is_continued():
    result = _run(
        [
            ModelReply(text=REASONING, finish_reason="length"),
            ModelReply(text=GOOD_CALL, finish_reason="stop"),
            ModelReply(text="Done, the file is written.", finish_reason="stop"),
        ],
        continuations=2,
    )
    assert result.error is None
    assert result.total_tool_calls == 1
    assert result.reasoning_continuations == 1
    assert "Done" in result.final_response


def test_truncation_mid_tool_call_still_fails():
    """A call cut mid-write is unusable; resuming risks emitting it twice."""
    fragment = 'Writing.\n<tool_call>\n{"name": "write_file", "arguments": {"path": "/a.go", "content": "package'
    result = _run([ModelReply(text=fragment, finish_reason="length")], continuations=2)
    assert result.error is not None and "truncated" in result.error


def test_continuations_are_capped():
    """A model that only ruminates must not burn every turn."""
    result = _run([ModelReply(text=REASONING, finish_reason="length")] * 5, continuations=2)
    assert result.error is not None and "truncated" in result.error
    assert result.reasoning_continuations == 2


def test_tool_call_loop_is_not_handed_more_turns():
    """coding_mcp_hard_01's 2730-tag loop must not be granted continuations.

    Not via the degenerate detector, which misses it: its repeating unit
    "<tool_call>\n" is 12 chars against DEGENERATE_UNIT_MAX of 8, which is why
    that run was reported as a plain truncation. The attempted-call guard is
    what stops it — a response reaching for a tool is never pure reasoning.
    """
    result = _run([ModelReply(text="<tool_call>\n" * 3000, finish_reason="length")], continuations=2)
    assert result.error is not None and "truncated" in result.error
    assert result.reasoning_continuations == 0


def test_short_unit_degeneration_still_fails_as_degenerate():
    """The detector itself must keep working where it does apply."""
    result = _run([ModelReply(text="\\n" * 3000, finish_reason="length")], continuations=2)
    assert result.error is not None
    assert "degenerate_repetition" in result.error
    assert result.reasoning_continuations == 0


def test_continued_history_is_compacted():
    """Replaying 32k tokens of reasoning verbatim would fill the context.

    Only the tail is carried forward: the model's conclusions land at the end —
    ornith's truncated turn ended on "I think I'm way overthinking this. Let me
    make a pragmatic decision".
    """
    captured: list = []

    def _capture(client, base_url, model_name, messages, temperature, max_tokens, kwargs=None, **rest):
        captured.append([m.content for m in messages])
        return [
            ModelReply(text=REASONING, finish_reason="length"),
            ModelReply(text=GOOD_CALL, finish_reason="stop"),
            ModelReply(text="done", finish_reason="stop"),
        ][len(captured) - 1]

    with patch("nite_eval.conversation_runner._call_model", side_effect=_capture):
        run_conversation(
            base_url="http://test",
            model_name="test-model",
            system_prompt="be helpful",
            tools=TOOLS,
            user_message="write a file",
            mock_env=MockToolEnv.from_task_yaml(MOCKS),
            max_turns=6,
            max_tokens=4096,
            reasoning_continuations=2,
        )
    # The turn-2 history must carry a trimmed version of turn 1's reasoning:
    # its tail, marked as clipped, and not the whole thing.
    carried = [c for c in captured[1] if "Let me think about this." in c]
    assert len(carried) == 1
    assert len(carried[0]) < len(REASONING)
    assert carried[0].startswith("[earlier reasoning truncated]")
    assert carried[0].endswith(REASONING[-40:])
