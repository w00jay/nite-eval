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
                # No tool calls — model is done (or silently stalled)
                if not response_text.strip() and total_tool_calls > 0:
                    # Empty response after prior tool calls: nudge once to elicit
                    # a final synthesis instead of silently accepting an empty answer.
                    logger.warning("Empty response on turn %d after %d tool calls; nudging", turn_num, total_tool_calls)
                    turns.append(turn)
                    messages.append(Message(role="assistant", content=response_text))
                    messages.append(
                        Message(
                            role="user",
                            content=(
                                "Your last response was empty. Based on the tool results above, "
                                "please provide your final answer to the original question now — "
                                "do not call more tools."
                            ),
                        )
                    )
                    nudge_start = time.monotonic()
                    nudged_text = _call_model(client, base_url, model_name, messages, temperature, max_tokens)
                    nudge_latency = (time.monotonic() - nudge_start) * 1000
                    total_latency += nudge_latency
                    nudge_parsed = extract_tool_calls(nudged_text)
                    nudge_turn = TurnResult(
                        turn=turn_num + 1,
                        response=nudged_text,
                        parsed=nudge_parsed,
                        latency_ms=nudge_latency,
                    )
                    turns.append(nudge_turn)
                    final = nudged_text.strip() or (
                        f"[Model returned empty response on turn {turn_num} after {total_tool_calls} tool calls; "
                        "nudge also returned empty]"
                    )
                    if not nudged_text.strip():
                        logger.warning("Nudge also produced empty response")
                    return ConversationResult(
                        turns=turns,
                        final_response=final,
                        total_tool_calls=total_tool_calls,
                        total_latency_ms=total_latency,
                        reached_max_turns=False,
                    )

                turns.append(turn)
                final = response_text.strip() or (
                    f"[Model returned empty response on turn {turn_num} with no tool calls]"
                )
                if not response_text.strip():
                    logger.warning("Empty response on turn %d with no tool calls", turn_num)
                return ConversationResult(
                    turns=turns,
                    final_response=final,
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
                turn.tool_responses.append({"name": tc.name, "arguments": tc.arguments, "result": mock_result})
                messages.append(Message(role="tool", content=tool_resp))

            turns.append(turn)

            if total_tool_calls >= max_tool_calls:
                logger.warning("Hit max_tool_calls (%d), nudging for synthesis", max_tool_calls)
                messages.append(
                    Message(
                        role="user",
                        content=(
                            f"You have used all {max_tool_calls} available tool calls. "
                            "Do not call any more tools. Based on everything you've gathered, "
                            "write your final answer to the original question now."
                        ),
                    )
                )
                nudge_start = time.monotonic()
                nudged_text = _call_model(client, base_url, model_name, messages, temperature, max_tokens)
                nudge_latency = (time.monotonic() - nudge_start) * 1000
                total_latency += nudge_latency
                nudge_parsed = extract_tool_calls(nudged_text)
                turns.append(
                    TurnResult(
                        turn=turn_num + 1,
                        response=nudged_text,
                        parsed=nudge_parsed,
                        latency_ms=nudge_latency,
                    )
                )
                break

        # Reached max turns. If every turn emitted tool calls and never
        # produced a free-text answer, nudge once for synthesis — symmetric
        # with the max_tool_calls branch above. Skip if we already nudged
        # in the cap branch (that branch breaks out before reaching here).
        if turns and all(t.parsed.tool_calls for t in turns):
            logger.warning("Hit max_turns (%d) with no free-text turn, nudging for synthesis", max_turns)
            messages.append(
                Message(
                    role="user",
                    content=(
                        f"You have used all {max_turns} available turns. "
                        "Do not call any more tools. Based on everything you've gathered, "
                        "write your final answer to the original question now."
                    ),
                )
            )
            nudge_start = time.monotonic()
            nudged_text = _call_model(client, base_url, model_name, messages, temperature, max_tokens)
            nudge_latency = (time.monotonic() - nudge_start) * 1000
            total_latency += nudge_latency
            nudge_parsed = extract_tool_calls(nudged_text)
            turns.append(
                TurnResult(
                    turn=max_turns + 1,
                    response=nudged_text,
                    parsed=nudge_parsed,
                    latency_ms=nudge_latency,
                )
            )

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
    If no turn has text, synthesizes a diagnostic marker so downstream
    scoring (judge) has context instead of an empty string.
    """
    for turn in reversed(turns):
        text = TOOL_CALL_TAG_RE.sub("", turn.response).strip()
        if len(text) > 20:
            return text
    # Fallback 1: any non-empty text across turns (prefer latest)
    for turn in reversed(turns):
        text = TOOL_CALL_TAG_RE.sub("", turn.response).strip()
        if text:
            return text
    # Fallback 2: all turns were tool-call-only — synthesize marker
    tc_count = sum(len(t.tool_responses) for t in turns)
    return f"[No text answer produced after {len(turns)} turns / {tc_count} tool calls]"


def _call_model(
    client: httpx.Client,
    base_url: str,
    model_name: str,
    messages: list[Message],
    temperature: float,
    max_tokens: int,
) -> str:
    """Send messages to the model and return the response text.

    Some llama-server builds route Qwen-style thinking output to a separate
    `reasoning_content` field when the model emits `<think>...</think>` blocks.
    If `content` is empty, fall back to `reasoning_content` so we don't treat
    a thought-only answer as a silent stall. Also logs the raw message keys
    and finish_reason once so diagnostic info lands in the run log.
    """
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
    choice = data["choices"][0]
    msg = choice.get("message", {})
    content = msg.get("content") or ""
    if not content.strip():
        reasoning = msg.get("reasoning_content") or ""
        finish = choice.get("finish_reason")
        logger.warning(
            "Empty content from %s (finish=%s, msg_keys=%s, reasoning_len=%d)",
            model_name,
            finish,
            sorted(msg.keys()),
            len(reasoning),
        )
        if reasoning.strip():
            return reasoning
    return content
