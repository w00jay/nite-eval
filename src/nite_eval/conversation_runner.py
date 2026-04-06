"""Multi-turn conversation runner for agentic evaluation tasks.

Implements the agent loop:
  user query → model response → parse tool calls → mock tool response → repeat
  until model produces a final text answer or max_turns is reached.
"""

import logging
import re
import time
from dataclasses import dataclass, field

import httpx

from nite_eval.hermes_parser import (
    ParsedResponse,
    extract_tool_calls,
    format_tool_definitions,
    format_tool_response,
)
from nite_eval.mock_tools import MockToolEnv

logger = logging.getLogger(__name__)


@dataclass
class Message:
    role: str
    content: str


@dataclass
class TurnResult:
    turn: int
    response: str
    parsed: ParsedResponse
    tool_responses: list[dict] = field(default_factory=list)
    latency_ms: float = 0.0


@dataclass
class ConversationResult:
    turns: list[TurnResult]
    final_response: str
    total_tool_calls: int
    total_latency_ms: float
    reached_max_turns: bool
    error: str | None = None


def run_conversation(
    base_url: str,
    model_name: str,
    system_prompt: str,
    tools: list[dict],
    user_message: str,
    mock_env: MockToolEnv,
    max_turns: int = 10,
    max_tool_calls: int = 20,
    timeout_seconds: float = 120.0,
    temperature: float = 0.0,
    max_tokens: int = 2048,
) -> ConversationResult:
    """Run a multi-turn conversation with Hermes-format tool calling.

    The system prompt gets tool definitions injected via <tools> tags.
    Each turn: send messages → get response → if tool calls, execute and loop.
    Stops early if max_tool_calls is reached to prevent search loops.
    """
    full_system = format_tool_definitions(tools) + "\n\n" + system_prompt.rstrip()

    messages: list[Message] = [
        Message(role="system", content=full_system),
        Message(role="user", content=user_message),
    ]

    turns: list[TurnResult] = []
    total_tool_calls = 0
    total_latency = 0.0

    client = httpx.Client(timeout=timeout_seconds)

    try:
        for turn_num in range(1, max_turns + 1):
            start = time.monotonic()

            response_text = _call_model(client, base_url, model_name, messages, temperature, max_tokens)

            latency = (time.monotonic() - start) * 1000
            total_latency += latency

            parsed = extract_tool_calls(response_text)
            turn = TurnResult(
                turn=turn_num,
                response=response_text,
                parsed=parsed,
                latency_ms=latency,
            )

            if not parsed.tool_calls:
                # No tool calls — model is done
                turns.append(turn)
                return ConversationResult(
                    turns=turns,
                    final_response=response_text,
                    total_tool_calls=total_tool_calls,
                    total_latency_ms=total_latency,
                    reached_max_turns=False,
                )

            # Execute tool calls and build response messages
            messages.append(Message(role="assistant", content=response_text))

            for tc in parsed.tool_calls:
                total_tool_calls += 1
                mock_result = mock_env.call(tc.name, tc.arguments)
                tool_resp = format_tool_response(tc.name, mock_result)
                turn.tool_responses.append({"name": tc.name, "result": mock_result})
                messages.append(Message(role="tool", content=tool_resp))

            turns.append(turn)

            if total_tool_calls >= max_tool_calls:
                logger.warning("Hit max_tool_calls (%d), stopping early", max_tool_calls)
                break

        # Reached max turns — find the best final response by walking back
        # through turns to find the last one with meaningful text content
        final = _extract_best_final_response(turns)
        return ConversationResult(
            turns=turns,
            final_response=final,
            total_tool_calls=total_tool_calls,
            total_latency_ms=total_latency,
            reached_max_turns=True,
        )

    except Exception as e:
        logger.exception("Conversation failed at turn %d", len(turns) + 1)
        return ConversationResult(
            turns=turns,
            final_response="",
            total_tool_calls=total_tool_calls,
            total_latency_ms=total_latency,
            reached_max_turns=False,
            error=str(e),
        )
    finally:
        client.close()


TOOL_CALL_TAG_RE = re.compile(r"<tool_call>.*?</tool_call>", re.DOTALL)


def _extract_best_final_response(turns: list[TurnResult]) -> str:
    """Find the best final response when max_turns is reached.

    Walks backwards through turns to find one with meaningful text
    (not just tool_call tags). Strips tool_call tags from the response.
    """
    for turn in reversed(turns):
        # Strip tool_call tags to get the text portion
        text = TOOL_CALL_TAG_RE.sub("", turn.response).strip()
        if len(text) > 20:
            return text
    # Fallback: return last turn's raw response stripped of tool calls
    if turns:
        return TOOL_CALL_TAG_RE.sub("", turns[-1].response).strip()
    return ""


def _call_model(
    client: httpx.Client,
    base_url: str,
    model_name: str,
    messages: list[Message],
    temperature: float,
    max_tokens: int,
) -> str:
    """Send messages to the model and return the response text."""
    resp = client.post(
        f"{base_url}/v1/chat/completions",
        json={
            "model": model_name,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]
