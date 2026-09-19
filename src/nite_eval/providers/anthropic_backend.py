"""Anthropic Messages API backend.

Native mode maps the runner's OpenAI-shaped tool plumbing onto `tool_use` /
`tool_result` blocks and back. Hermes mode sends the same prompt-injected
format local models get, so a model can be run both ways and the format's
contribution to its score measured rather than assumed.

`messages.create` takes no `temperature` in anthropic>=1.7 — current models
reject sampling parameters — so none is sent. Thinking is on by default on
Opus 5 and shares the max_tokens budget with the answer; prefer lowering
`output_config.effort` over disabling it, since thinking-off has its own
tool-calling failure modes.
"""

from __future__ import annotations

import json
import logging
import os
from typing import TYPE_CHECKING, Any, cast

import anthropic

from nite_eval.hermes_parser import ToolCall
from nite_eval.providers.base import (
    BUDGET_EXHAUSTED,
    FINISH_LENGTH,
    FINISH_STOP,
    FINISH_TOOL_CALLS,
    function_specs,
    openai_tool_entry,
    pending_call_ids,
)

if TYPE_CHECKING:
    from anthropic.types import MessageParam, ToolParam

    from nite_eval.conversation_runner import Message, ModelReply

logger = logging.getLogger(__name__)

DEFAULT_MAX_RETRIES = 4

# Anthropic stop reasons → the finish_reason vocabulary the runner understands.
STOP_REASONS = {
    "max_tokens": FINISH_LENGTH,
    "tool_use": FINISH_TOOL_CALLS,
    "end_turn": FINISH_STOP,
    "stop_sequence": FINISH_STOP,
    "pause_turn": FINISH_STOP,
    "refusal": "refusal",
}


class AnthropicBackend:
    provider = "anthropic"

    def __init__(
        self,
        model_id: str,
        api_key_env: str = "ANTHROPIC_API_KEY",
        params: dict[str, Any] | None = None,
        timeout: float = 1200.0,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ):
        self.model_id = model_id
        self.params = params or {}
        # With no explicit key the SDK falls back to its own resolution chain
        # (ANTHROPIC_API_KEY, auth token, or an `ant auth login` profile).
        self._client = anthropic.Anthropic(
            api_key=os.environ.get(api_key_env), timeout=timeout, max_retries=max_retries
        )

    def _tools_param(self, tools: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
        return [
            {
                "name": spec["name"],
                "description": spec.get("description", ""),
                "input_schema": spec.get("parameters") or {"type": "object", "properties": {}},
            }
            for spec in function_specs(tools)
        ]

    def _translate(self, messages: list[Message]) -> tuple[str, list[dict[str, Any]]]:
        """Split off the system prompt and convert the rest to Anthropic blocks.

        Consecutive tool results are coalesced into a single user message, which
        is what the API requires for an assistant turn that made several calls.
        """
        system = ""
        wire: list[dict[str, Any]] = []
        pending_results: list[dict[str, Any]] = []

        def flush() -> None:
            if pending_results:
                wire.append({"role": "user", "content": list(pending_results)})
                pending_results.clear()

        for msg in messages:
            if msg.role == "system":
                system = msg.content
                continue
            if msg.role == "tool":
                pending_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": msg.tool_call_id or "unknown",
                        "content": msg.content,
                    }
                )
                continue
            flush()
            if msg.role == "assistant":
                blocks: list[dict[str, Any]] = []
                if msg.content.strip():
                    blocks.append({"type": "text", "text": msg.content})
                for entry in msg.tool_calls or []:
                    fn = entry.get("function") or {}
                    arguments = fn.get("arguments")
                    if isinstance(arguments, str):
                        try:
                            arguments = json.loads(arguments)
                        except json.JSONDecodeError:
                            arguments = {}
                    blocks.append(
                        {
                            "type": "tool_use",
                            "id": entry.get("id", "call_0"),
                            "name": fn.get("name", ""),
                            "input": arguments if isinstance(arguments, dict) else {},
                        }
                    )
                if not blocks:
                    blocks.append({"type": "text", "text": "(no output)"})
                wire.append({"role": "assistant", "content": blocks})
            else:
                wire.append({"role": msg.role, "content": msg.content or "(empty)"})

        # Calls the runner dropped at the tool-call cap never got a result;
        # the API rejects the turn without one.
        for call_id, name in pending_call_ids(messages):
            logger.debug("Synthesizing budget-exhausted result for unanswered call %s (%s)", call_id, name)
            pending_results.append(
                {"type": "tool_result", "tool_use_id": call_id, "content": BUDGET_EXHAUSTED, "is_error": True}
            )
        flush()
        return system, wire

    def generate(
        self,
        messages: list[Message],
        max_tokens: int,
        tools: list[dict[str, Any]] | None = None,
        native_tools: bool = False,
    ) -> ModelReply:
        from nite_eval.conversation_runner import ModelReply

        system, wire = self._translate(messages)
        kwargs: dict[str, Any] = {"messages": cast("list[MessageParam]", wire)}
        if system.strip():
            kwargs["system"] = system
        if native_tools and tools:
            kwargs["tools"] = cast("list[ToolParam]", self._tools_param(tools))

        response = self._client.messages.create(model=self.model_id, max_tokens=max_tokens, **kwargs, **self.params)

        finish = STOP_REASONS.get(response.stop_reason or "", FINISH_STOP)
        prompt_tokens = response.usage.input_tokens or 0
        completion_tokens = response.usage.output_tokens or 0

        if response.stop_reason == "refusal":
            category = getattr(response.stop_details, "category", None)
            logger.warning("%s refused the request (category=%s)", self.model_id, category)
            return ModelReply(
                text=f"[refusal from {self.model_id}: category={category}]",
                finish_reason=finish,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )

        text_parts: list[str] = []
        calls: list[ToolCall] = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                arguments = block.input if isinstance(block.input, dict) else {}
                calls.append(
                    ToolCall(
                        name=block.name,
                        arguments=arguments,
                        raw=json.dumps(openai_tool_entry(block.id, block.name, arguments)),
                    )
                )

        return ModelReply(
            text="\n".join(p for p in text_parts if p),
            finish_reason=finish,
            native_tool_calls=calls or None,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    def close(self) -> None:
        self._client.close()
