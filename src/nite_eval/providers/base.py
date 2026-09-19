"""Shared helpers for frontier-model backends.

The runner's `Message` / `ModelReply` / `ToolCall` types are reused as-is; a
backend's only job is translating them to and from one provider's wire format.
"""

from __future__ import annotations

import json
from typing import Any

# Finish reasons the runner reacts to. "length" is load-bearing: ModelReply.truncated
# keys on it, which drives truncation diagnosis and the parse-retry path.
FINISH_LENGTH = "length"
FINISH_STOP = "stop"
FINISH_TOOL_CALLS = "tool_calls"


def function_specs(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Unwrap task-YAML tools ({type: function, function: {...}}) to bare specs."""
    specs = []
    for tool in tools or []:
        if tool.get("type") == "function" and "function" in tool:
            specs.append(tool["function"])
        else:
            specs.append(tool)
    return specs


def openai_tool_entry(call_id: str, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Build the OpenAI-shaped tool_call entry the runner stores in ToolCall.raw.

    The runner rebuilds assistant turns from this (`_native_entries`) and pairs
    tool results to calls by `id`, so every backend hands back the same shape
    regardless of the provider's own representation.
    """
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


def answered_call_ids(messages: list[Any]) -> set[str]:
    """IDs of tool calls that already have a result in the history."""
    return {m.tool_call_id for m in messages if m.role == "tool" and m.tool_call_id}


def pending_call_ids(messages: list[Any]) -> list[tuple[str, str]]:
    """(id, name) of assistant tool calls left unanswered in the history.

    The runner stops executing calls once max_tool_calls is reached and does not
    synthesize results for the remainder. Local llama-server tolerates that;
    Anthropic and OpenAI both reject a turn whose tool calls go unanswered, so
    the backends close the gap themselves rather than changing runner semantics.
    """
    answered = answered_call_ids(messages)
    pending: list[tuple[str, str]] = []
    for msg in messages:
        for entry in msg.tool_calls or []:
            call_id = entry.get("id")
            if call_id and call_id not in answered:
                name = (entry.get("function") or {}).get("name", "")
                pending.append((call_id, name))
    return pending


BUDGET_EXHAUSTED = json.dumps({"error": "tool call budget exhausted; no more tool calls available"})
