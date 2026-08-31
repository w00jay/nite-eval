"""Per-model native tool calling.

Ornith's documented usage passes `tools` in the request and reads structured
`message.tool_calls`; llama-server renders the model's own template and parses
the call back. nite-eval's default is the opposite — tool definitions pasted
into the system prompt, the reply parsed out of text — and asking Ornith to
hand-write JSON in prose produced a different malformation nearly every turn
(4 distinct classes across 3 runs, 52 repairs on one task).

Through the native path the same model returned 5/5 clean calls with correct
types and no repairs, so it is opt-in per model rather than the harness default:
switching a model changes what it is asked to do and moves its scores.
"""

import json

import httpx

from nite_eval.conversation_runner import Message, _call_model

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}, "limit": {"type": "integer"}},
                "required": ["query"],
            },
        },
    }
]


def _client(payloads: list[dict], response: dict) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        payloads.append(json.loads(request.content))
        return httpx.Response(200, json=response)

    return httpx.Client(transport=httpx.MockTransport(handler))


NATIVE_REPLY = {
    "choices": [
        {
            "message": {
                "content": "",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "web_search",
                            "arguments": '{"query": "MCP gateway", "limit": 5}',
                        },
                    }
                ],
            },
            "finish_reason": "tool_calls",
        }
    ]
}


def test_tools_sent_when_native_enabled():
    payloads: list[dict] = []
    with _client(payloads, NATIVE_REPLY) as client:
        _call_model(
            client,
            "http://x",
            "ornith-1.5-35b-a3b",
            [Message(role="user", content="search")],
            0.0,
            128,
            None,
            tools=TOOLS,
            native_tools=True,
        )
    assert payloads[0]["tools"] == TOOLS
    assert payloads[0]["tool_choice"] == "auto"


def test_tools_not_sent_by_default():
    """The other six models must keep the measured prompt-text path."""
    payloads: list[dict] = []
    reply = {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}
    with _client(payloads, reply) as client:
        _call_model(
            client,
            "http://x",
            "qwen3.8-27b",
            [Message(role="user", content="hi")],
            0.0,
            128,
            None,
            tools=TOOLS,
        )
    assert "tools" not in payloads[0]
    assert "tool_choice" not in payloads[0]


def test_structured_tool_calls_are_returned():
    payloads: list[dict] = []
    with _client(payloads, NATIVE_REPLY) as client:
        reply = _call_model(
            client,
            "http://x",
            "ornith-1.5-35b-a3b",
            [Message(role="user", content="search")],
            0.0,
            128,
            None,
            tools=TOOLS,
            native_tools=True,
        )
    assert reply.native_tool_calls is not None
    assert len(reply.native_tool_calls) == 1
    call = reply.native_tool_calls[0]
    assert call.name == "web_search"
    assert call.arguments == {"query": "MCP gateway", "limit": 5}
    assert isinstance(call.arguments["limit"], int)


def test_empty_content_with_tool_calls_is_not_a_stall():
    """A native tool call turn has empty content by design.

    The reasoning_content fallback must not fire and hand the judge a chain of
    thought in place of the call.
    """
    payloads: list[dict] = []
    reply_body = json.loads(json.dumps(NATIVE_REPLY))
    reply_body["choices"][0]["message"]["reasoning_content"] = "let me think about this"
    with _client(payloads, reply_body) as client:
        reply = _call_model(
            client,
            "http://x",
            "ornith-1.5-35b-a3b",
            [Message(role="user", content="search")],
            0.0,
            128,
            None,
            tools=TOOLS,
            native_tools=True,
        )
    assert reply.text == ""
    assert reply.native_tool_calls and reply.native_tool_calls[0].name == "web_search"


def test_arguments_already_an_object_are_accepted():
    """Some servers return arguments as an object rather than a JSON string."""
    body = json.loads(json.dumps(NATIVE_REPLY))
    body["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = {"query": "x"}
    payloads: list[dict] = []
    with _client(payloads, body) as client:
        reply = _call_model(
            client,
            "http://x",
            "ornith-1.5-35b-a3b",
            [Message(role="user", content="search")],
            0.0,
            128,
            None,
            tools=TOOLS,
            native_tools=True,
        )
    assert reply.native_tool_calls[0].arguments == {"query": "x"}
