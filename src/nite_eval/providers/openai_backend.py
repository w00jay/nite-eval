"""OpenAI chat-completions backend, also used for OpenAI-compatible gateways.

Point `base_url` at OpenRouter, Together, or any other compatible endpoint and
name its key in `api_key_env`; the wire format is identical. Task-YAML tools are
already OpenAI function schema, so native mode passes them straight through —
and the runner's own tool plumbing is OpenAI-shaped, so little translation is
needed beyond answering calls the runner dropped at its cap.
"""

from __future__ import annotations

import json
import logging
import os
from typing import TYPE_CHECKING, Any, cast

import openai

from nite_eval.hermes_parser import ToolCall
from nite_eval.providers.base import BUDGET_EXHAUSTED, openai_tool_entry, pending_call_ids

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolUnionParam

    from nite_eval.conversation_runner import Message, ModelReply

logger = logging.getLogger(__name__)

DEFAULT_MAX_RETRIES = 4


class OpenAIBackend:
    def __init__(
        self,
        model_id: str,
        api_key_env: str = "OPENAI_API_KEY",
        base_url: str | None = None,
        provider: str = "openai",
        temperature: float | None = None,
        max_tokens_param: str = "max_tokens",
        params: dict[str, Any] | None = None,
        timeout: float = 1200.0,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ):
        self.model_id = model_id
        self.provider = provider
        self.temperature = temperature
        # Reasoning models reject `max_tokens` and want `max_completion_tokens`;
        # which applies is per-model, so it stays a config decision.
        self.max_tokens_param = max_tokens_param
        self.params = params or {}
        self._client = openai.OpenAI(
            api_key=os.environ.get(api_key_env), base_url=base_url, timeout=timeout, max_retries=max_retries
        )

    def _translate(self, messages: list[Message]) -> list[dict[str, Any]]:
        wire = [m.to_payload() for m in messages]
        # Calls the runner dropped at the tool-call cap never got a result;
        # the API rejects the turn without one.
        for call_id, name in pending_call_ids(messages):
            logger.debug("Synthesizing budget-exhausted result for unanswered call %s (%s)", call_id, name)
            wire.append({"role": "tool", "tool_call_id": call_id, "content": BUDGET_EXHAUSTED})
        return wire

    def generate(
        self,
        messages: list[Message],
        max_tokens: int,
        tools: list[dict[str, Any]] | None = None,
        native_tools: bool = False,
    ) -> ModelReply:
        from nite_eval.conversation_runner import ModelReply

        kwargs: dict[str, Any] = {self.max_tokens_param: max_tokens}
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if native_tools and tools:
            kwargs["tools"] = cast("list[ChatCompletionToolUnionParam]", tools)
            kwargs["tool_choice"] = "auto"

        response = self._client.chat.completions.create(
            model=self.model_id,
            messages=cast("list[ChatCompletionMessageParam]", self._translate(messages)),
            **kwargs,
            **self.params,
        )

        choice = response.choices[0]
        usage = response.usage
        reply_kwargs = {
            "finish_reason": choice.finish_reason or None,
            "prompt_tokens": (usage.prompt_tokens if usage else 0) or 0,
            "completion_tokens": (usage.completion_tokens if usage else 0) or 0,
        }

        calls: list[ToolCall] = []
        for call in choice.message.tool_calls or []:
            function = getattr(call, "function", None)
            if function is None:  # non-function tool call type
                logger.warning("Ignoring non-function tool call from %s: %s", self.model_id, call.type)
                continue
            raw_args = function.arguments or "{}"
            try:
                arguments = json.loads(raw_args)
            except json.JSONDecodeError:
                logger.warning("Malformed tool arguments from %s: %r", self.model_id, raw_args)
                continue
            if not isinstance(arguments, dict):
                arguments = {}
            calls.append(
                ToolCall(
                    name=function.name,
                    arguments=arguments,
                    raw=json.dumps(openai_tool_entry(call.id, function.name, arguments)),
                )
            )

        return ModelReply(
            text=choice.message.content or "",
            native_tool_calls=calls or None,
            **reply_kwargs,
        )

    def close(self) -> None:
        self._client.close()
