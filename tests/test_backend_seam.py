"""The runner's agent loop must work identically when a backend drives it.

These exercise run_conversation end to end with a scripted backend, so the
nudges, caps and bookkeeping are proven against the same loop the local path
uses — not a reimplementation.
"""

import json

from nite_eval.conversation_runner import ModelReply, run_conversation
from nite_eval.hermes_parser import ToolCall
from nite_eval.mock_tools import MockToolEnv
from nite_eval.providers.base import openai_tool_entry

MOCK_SPEC = {"search": [{"response": {"hits": ["a", "b"]}}]}
TOOLS = [{"type": "function", "function": {"name": "search", "parameters": {}}}]


class ScriptedBackend:
    provider = "anthropic"
    model_id = "claude-opus-5"

    def __init__(self, replies):
        self._replies = list(replies)
        self.seen: list[list] = []
        self.closed = False

    def generate(self, messages, max_tokens, tools=None, native_tools=False):  # noqa: ARG002
        self.seen.append([m.to_payload() for m in messages])
        return self._replies.pop(0)

    def close(self):
        self.closed = True


def native_call(call_id="c1", name="search", **args):
    return ToolCall(name=name, arguments=args, raw=json.dumps(openai_tool_entry(call_id, name, args)))


def run(replies, **kwargs):
    backend = ScriptedBackend(replies)
    result = run_conversation(
        base_url="http://unused",
        model_name="claude-opus-5",
        system_prompt="You are a test agent.",
        tools=TOOLS,
        user_message="find something",
        mock_env=MockToolEnv.from_task_yaml(MOCK_SPEC),
        native_tools=True,
        backend=backend,
        **kwargs,
    )
    return backend, result


def test_backend_drives_a_single_turn_answer():
    backend, result = run([ModelReply(text="The answer is 42.", prompt_tokens=10, completion_tokens=5)])
    assert result.final_response == "The answer is 42."
    assert result.total_tool_calls == 0
    assert result.total_prompt_tokens == 10
    assert result.total_completion_tokens == 5


def test_backend_tool_call_round_trip_reaches_the_mock_env():
    backend, result = run(
        [
            ModelReply(text="", native_tool_calls=[native_call(q="gpu")], prompt_tokens=10, completion_tokens=20),
            ModelReply(text="Found a and b.", prompt_tokens=30, completion_tokens=8),
        ]
    )
    assert result.total_tool_calls == 1
    assert result.final_response == "Found a and b."
    # Usage is summed across every generation, nudges included
    assert result.total_prompt_tokens == 40
    assert result.total_completion_tokens == 28

    # The second call saw its own assistant turn and the tool result keyed to it
    history = backend.seen[1]
    assistant = next(m for m in history if m["role"] == "assistant")
    assert assistant["tool_calls"][0]["id"] == "c1"
    tool_msg = next(m for m in history if m["role"] == "tool")
    assert tool_msg["tool_call_id"] == "c1"
    assert "hits" in tool_msg["content"]


def test_max_tokens_finish_reason_still_marks_truncation():
    """Backends map their own stop reasons onto the runner's vocabulary."""
    reply = ModelReply(text="cut off mid-sen", finish_reason="length")
    assert reply.truncated is True


def test_backend_sees_growing_history_across_turns():
    backend, result = run(
        [
            ModelReply(text="", native_tool_calls=[native_call("c1")]),
            ModelReply(text="", native_tool_calls=[native_call("c2")]),
            ModelReply(text="Done investigating, here is the synthesis."),
        ],
        max_turns=3,
    )
    assert [len(h) for h in backend.seen] == [2, 4, 6]
    assert result.final_response == "Done investigating, here is the synthesis."
