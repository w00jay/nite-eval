"""chat_template_kwargs must reach the llama-server payload.

Qwen3.8's only thinking switch is the template kwarg `enable_thinking`; its
chat template has no `/no_think` branch. If this plumbing breaks, the model
silently burns its whole max_tokens budget in `reasoning_content` and returns
empty `content` — which scores as a bad answer rather than as an error.
"""

import httpx

from nite_eval.conversation_runner import Message, _call_model


def _capture(payloads: list[dict]) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        import json

        payloads.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]})

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_chat_template_kwargs_forwarded():
    payloads: list[dict] = []
    with _capture(payloads) as client:
        _call_model(
            client,
            "http://x",
            "qwen3.8-27b",
            [Message(role="user", content="hi")],
            0.0,
            128,
            {"enable_thinking": False},
        )
    assert payloads[0]["chat_template_kwargs"] == {"enable_thinking": False}


def test_chat_template_kwargs_omitted_when_absent():
    """Models without the key must not gain an empty field in the request."""
    payloads: list[dict] = []
    with _capture(payloads) as client:
        _call_model(client, "http://x", "m", [Message(role="user", content="hi")], 0.0, 128, None)
    assert "chat_template_kwargs" not in payloads[0]
