"""Frontier backend translation — no network, no real API keys."""

import json

import pytest

from nite_eval.conversation_runner import Message
from nite_eval.providers import build_backend, is_local, model_id_for, native_tools_for
from nite_eval.providers.base import function_specs, openai_tool_entry, pending_call_ids

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_docs",
            "description": "Search the docs",
            "parameters": {"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]},
        },
    }
]


@pytest.fixture
def anthropic_backend(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    from nite_eval.providers.anthropic_backend import AnthropicBackend

    return AnthropicBackend(model_id="claude-opus-5")


@pytest.fixture
def openai_backend(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    from nite_eval.providers.openai_backend import OpenAIBackend

    return OpenAIBackend(model_id="gpt-test")


def assistant_with_calls(*ids):
    return Message(
        role="assistant",
        content="working",
        tool_calls=[openai_tool_entry(i, "search_docs", {"q": "x"}) for i in ids],
    )


def test_function_specs_unwraps_task_yaml_tools():
    assert function_specs(TOOLS)[0]["name"] == "search_docs"
    bare = [{"name": "x", "parameters": {}}]
    assert function_specs(bare) == bare
    assert function_specs(None) == []


def test_anthropic_tool_schema_uses_input_schema(anthropic_backend):
    tool = anthropic_backend._tools_param(TOOLS)[0]
    assert tool["name"] == "search_docs"
    assert tool["input_schema"]["properties"] == {"q": {"type": "string"}}
    assert "parameters" not in tool


def test_anthropic_tool_schema_defaults_missing_parameters(anthropic_backend):
    tool = anthropic_backend._tools_param([{"type": "function", "function": {"name": "ping"}}])[0]
    assert tool["input_schema"] == {"type": "object", "properties": {}}


def test_anthropic_splits_system_and_coalesces_tool_results(anthropic_backend):
    history = [
        Message(role="system", content="You are a test agent."),
        Message(role="user", content="hi"),
        assistant_with_calls("c1", "c2"),
        Message(role="tool", content='{"hits": 1}', tool_call_id="c1"),
        Message(role="tool", content='{"hits": 2}', tool_call_id="c2"),
        Message(role="user", content="now answer"),
    ]
    system, wire = anthropic_backend._translate(history)

    assert system == "You are a test agent."
    assert [m["role"] for m in wire] == ["user", "assistant", "user", "user"]
    # Both results land in ONE user message, as the API requires
    results = wire[2]["content"]
    assert [b["tool_use_id"] for b in results] == ["c1", "c2"]
    assert all(b["type"] == "tool_result" for b in results)
    # Assistant turn keeps its text plus one tool_use block per call
    assert [b["type"] for b in wire[1]["content"]] == ["text", "tool_use", "tool_use"]
    assert wire[1]["content"][1]["input"] == {"q": "x"}


def test_anthropic_answers_calls_the_runner_dropped_at_its_cap(anthropic_backend):
    """The runner stops executing at max_tool_calls; the API rejects an unanswered call."""
    history = [
        Message(role="user", content="hi"),
        assistant_with_calls("c1", "c2", "c3"),
        Message(role="tool", content='{"hits": 1}', tool_call_id="c1"),
    ]
    _, wire = anthropic_backend._translate(history)
    answered = {b["tool_use_id"] for m in wire if m["role"] == "user" for b in m["content"] if isinstance(b, dict)}
    assert answered == {"c1", "c2", "c3"}
    synthesized = [
        b for m in wire if m["role"] == "user" for b in m["content"] if isinstance(b, dict) and b.get("is_error")
    ]
    assert len(synthesized) == 2
    assert "budget exhausted" in synthesized[0]["content"]


def test_anthropic_never_emits_empty_assistant_content(anthropic_backend):
    _, wire = anthropic_backend._translate(
        [Message(role="user", content="hi"), Message(role="assistant", content="   ")]
    )
    assert wire[1]["content"] == [{"type": "text", "text": "(no output)"}]


def test_openai_passes_runner_payloads_through_and_repairs_unanswered(openai_backend):
    history = [
        Message(role="system", content="sys"),
        Message(role="user", content="hi"),
        assistant_with_calls("c1", "c2"),
        Message(role="tool", content='{"hits": 1}', tool_call_id="c1"),
    ]
    wire = openai_backend._translate(history)
    assert wire[0] == {"role": "system", "content": "sys"}
    assert wire[2]["tool_calls"][0]["id"] == "c1"
    assert json.loads(wire[2]["tool_calls"][0]["function"]["arguments"]) == {"q": "x"}
    assert wire[-1]["tool_call_id"] == "c2"
    assert "budget exhausted" in wire[-1]["content"]


def test_pending_call_ids_ignores_answered_calls():
    history = [
        assistant_with_calls("c1", "c2"),
        Message(role="tool", content="{}", tool_call_id="c1"),
    ]
    assert pending_call_ids(history) == [("c2", "search_docs")]


def test_native_tools_defaults_per_provider():
    # Local models keep Hermes, preserving comparability with past runs
    assert native_tools_for({"name": "qwen3.6", "backend": "llama.cpp"}) is False
    # API models default to their own tool API
    assert native_tools_for({"name": "opus", "provider": "anthropic"}) is True
    # Explicit config always wins, in both directions
    assert native_tools_for({"name": "opus", "provider": "anthropic", "native_tools": False}) is False
    assert native_tools_for({"name": "ornith", "native_tools": True}) is True


def test_registry_returns_none_for_local_models():
    """None means 'use the runner's own llama-server path', unchanged."""
    cfg = {"name": "qwen3.6-35b-a3b", "backend": "llama.cpp"}
    assert is_local(cfg)
    assert build_backend(cfg) is None


def test_registry_builds_api_backends(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")

    anthropic = build_backend({"name": "opus", "provider": "anthropic", "api_model": "claude-opus-5"})
    assert anthropic is not None
    assert anthropic.provider == "anthropic"
    assert anthropic.model_id == "claude-opus-5"
    anthropic.close()

    compat = build_backend(
        {
            "name": "or-model",
            "provider": "openai_compatible",
            "api_model": "meta/llama",
            "base_url": "https://openrouter.ai/api/v1",
            "api_key_env": "OPENROUTER_API_KEY",
            "provider_label": "openrouter",
        }
    )
    assert compat is not None
    assert compat.provider == "openrouter"
    assert model_id_for({"name": "or-model", "api_model": "meta/llama"}) == "meta/llama"
    compat.close()


def test_registry_rejects_unknown_provider():
    with pytest.raises(ValueError, match="unknown provider"):
        build_backend({"name": "x", "provider": "nope"})
