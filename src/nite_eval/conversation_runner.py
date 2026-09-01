"""Multi-turn conversation runner for agentic evaluation tasks.

Implements the agent loop:
  user query → model response → parse tool calls → mock tool response → repeat
  until model produces a final text answer or max_turns is reached.
"""

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Protocol

import httpx

from nite_eval.hermes_parser import (
    ParsedResponse,
    ToolCall,
    extract_tool_calls,
    format_tool_definitions,
    format_tool_response,
)

logger = logging.getLogger(__name__)


class ToolEnv(Protocol):
    """What the runner needs from a tool environment.

    Satisfied by both MockToolEnv and SandboxToolEnv, so a task can run against
    mocks or a real container without the runner knowing which.
    """

    def call(self, tool_name: str, arguments: dict) -> dict: ...


@dataclass
class Message:
    role: str
    content: str
    # Native tool calling only. The OpenAI protocol needs the assistant turn to
    # carry the calls it made and each tool result to name the call it answers;
    # without them the model sees itself say nothing and tool results arriving
    # unprompted, and gives up after a single round.
    tool_calls: list[dict] | None = None
    tool_call_id: str | None = None

    def to_payload(self) -> dict:
        payload: dict = {"role": self.role, "content": self.content}
        if self.tool_calls:
            payload["tool_calls"] = self.tool_calls
        if self.tool_call_id:
            payload["tool_call_id"] = self.tool_call_id
        return payload


@dataclass
class ModelReply:
    """One generation from the target model, with its stop reason.

    `finish_reason` was previously read only when content came back empty, so a
    response cut off at max_tokens with non-empty content was indistinguishable
    from a complete one. Truncated tool calls then failed to parse and the
    runner treated the fragment as the model's final answer.
    """

    text: str
    finish_reason: str | None = None
    # Structured calls from the server's own tool-call parsing, when the model
    # runs with native_tools. None means the model was not asked for them, and
    # the caller should parse the text instead.
    native_tool_calls: list[ToolCall] | None = None

    @property
    def truncated(self) -> bool:
        return self.finish_reason == "length"


@dataclass
class TurnResult:
    turn: int
    response: str
    parsed: ParsedResponse
    tool_responses: list[dict] = field(default_factory=list)
    latency_ms: float = 0.0
    finish_reason: str | None = None
    truncated: bool = False


# Per-HTTP-request read timeout — decoupled from the task wall-clock budget.
# A single generation must be able to spend its whole max_tokens budget without
# tripping this. Measured on qwen3.8-27b: ~28 tok/s, so a full 16384-token
# generation needs ~585s. The previous 600s ceiling left a 2.5% margin, which
# would have converted a legitimately long answer into a task failure.
HTTP_READ_TIMEOUT = 1200.0

# Corrective retries offered when a tool call does not parse. One is enough to
# separate a transient JSON slip from a model that cannot emit valid calls;
# more would let a broken model burn the whole turn budget on retries.
MAX_PARSE_RETRIES = 1

# How much of an unparsable payload to keep in the task's error field.
UNPARSED_RAW_CHARS = 2000

# Tool calls larger than this are summarised in conversation history rather than
# carried verbatim. Sized above a typical search query or shell command so only
# file-sized payloads are affected.
COMPACT_TOOL_CALL_OVER = 1500

# Below this a tool-free turn is treated as a fragment rather than a synthesis,
# though it is still preferred over no answer at all.
MIN_FINAL_ANSWER_CHARS = 20

# A response is called degenerate when one short substring repeats enough times
# to dominate it. qwen3.8 on coding_artemis_medium_01 wrote 276 characters of
# real content and then emitted 12249 consecutive "\\n" escapes, filling the
# whole budget. That surfaces as finish_reason=length, which reads as the
# harness's budget being too small — it is not, and raising max_tokens from
# 16384 to 24576 simply produced a longer loop (24725 -> 37013 chars, the same
# 1.51 chars per token both times).
DEGENERATE_MIN_REPEATS = 200
DEGENERATE_UNIT_MAX = 8
DEGENERATE_SHARE = 0.5


@dataclass
class ConversationResult:
    turns: list[TurnResult]
    final_response: str
    total_tool_calls: int
    total_latency_ms: float
    reached_max_turns: bool
    error: str | None = None
    # Tool calls that parsed only after JSON repair — a model-quality signal
    # that must stay visible, not be absorbed by the parser.
    repaired_tool_calls: int = 0


def run_conversation(
    base_url: str,
    model_name: str,
    system_prompt: str,
    tools: list[dict],
    user_message: str,
    mock_env: ToolEnv,
    max_turns: int = 10,
    max_tool_calls: int = 20,
    timeout_seconds: float = 1800.0,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    system_suffix: str = "",
    chat_template_kwargs: dict | None = None,
    native_tools: bool = False,
) -> ConversationResult:
    """Run a multi-turn conversation with Hermes-format tool calling.

    The system prompt gets tool definitions injected via <tools> tags.
    Each turn: send messages → get response → if tool calls, execute and loop.
    Stops early if max_tool_calls is reached to prevent search loops.

    `system_suffix` is appended to the system prompt — used for model-specific
    chat-template triggers like Qwen3's `/no_think` that disable the thinking
    budget and let the model emit a final answer directly.

    `chat_template_kwargs` is forwarded verbatim to llama-server, which passes it
    into the Jinja chat template. Needed for models whose thinking toggle is a
    template variable rather than a prompt string — Qwen3.8 has no `/no_think`
    branch at all and only responds to `{"enable_thinking": false}`.

    `timeout_seconds` is the task's wall-clock budget, checked between turns.
    It previously did nothing at all — accepted, then discarded — so a task
    YAML's timeout was decorative. A single generation is bounded separately
    by HTTP_READ_TIMEOUT, so a long answer is never killed mid-flight.
    """
    # With native tool calling the server renders the model's own template,
    # which carries its tool-call instructions — injecting ours as prose too
    # would give the model two conflicting formats to satisfy.
    if native_tools:
        full_system = system_prompt.rstrip()
    else:
        full_system = format_tool_definitions(tools) + "\n\n" + system_prompt.rstrip()
    if system_suffix:
        full_system = full_system.rstrip() + "\n\n" + system_suffix.strip()

    messages: list[Message] = [
        Message(role="system", content=full_system),
        Message(role="user", content=user_message),
    ]

    turns: list[TurnResult] = []
    total_tool_calls = 0
    total_latency = 0.0
    cap_nudged = False  # Set when the max_tool_calls nudge fires so the max_turns nudge doesn't double-nudge.
    parse_retries = 0  # Corrective retries spent on unparsable tool calls.
    repaired_total = 0  # Tool calls salvaged by JSON repair.

    # Per-request timeout stays module-level and generous: conflating it with
    # the task budget caused spurious ReadTimeouts when one generation ran
    # longer than the task YAML's short timeout. The task budget is enforced
    # separately, between turns, so a long single generation is never killed
    # mid-flight but a runaway conversation still terminates.
    client = httpx.Client(timeout=HTTP_READ_TIMEOUT)
    task_start = time.monotonic()

    try:
        for turn_num in range(1, max_turns + 1):
            elapsed = time.monotonic() - task_start
            if elapsed > timeout_seconds:
                logger.warning(
                    "Task wall-clock budget exhausted (%.0fs > %.0fs) after %d turns",
                    elapsed,
                    timeout_seconds,
                    len(turns),
                )
                return ConversationResult(
                    turns=turns,
                    final_response="",
                    total_tool_calls=total_tool_calls,
                    total_latency_ms=total_latency,
                    reached_max_turns=False,
                    repaired_tool_calls=repaired_total,
                    error=(
                        f"task_timeout: {elapsed:.0f}s exceeded budget of {timeout_seconds:.0f}s "
                        f"after {len(turns)} turns / {total_tool_calls} tool calls"
                    ),
                )

            start = time.monotonic()

            reply = _call_model(
                client,
                base_url,
                model_name,
                messages,
                temperature,
                max_tokens,
                chat_template_kwargs,
                tools=tools,
                native_tools=native_tools,
            )
            response_text = reply.text

            latency = (time.monotonic() - start) * 1000
            total_latency += latency

            if reply.native_tool_calls is not None:
                parsed = ParsedResponse(tool_calls=reply.native_tool_calls, text=response_text)
            else:
                parsed = extract_tool_calls(response_text, tools)
            repaired_total += parsed.repaired
            turn = TurnResult(
                turn=turn_num,
                response=response_text,
                parsed=parsed,
                latency_ms=latency,
                finish_reason=reply.finish_reason,
                truncated=reply.truncated,
            )

            # A generation cut off at max_tokens is not an answer. Previously
            # this fell through to "no tool calls -> model is done" and the
            # fragment was judged as a final response; 87% of leaked
            # <tool_call> tags in the results DB were truncations, not
            # malformed model output. Fail the task instead of scoring noise.
            if reply.truncated:
                turns.append(turn)
                degenerate = detect_degenerate_repetition(response_text)
                if degenerate:
                    repeats, unit = degenerate
                    return ConversationResult(
                        turns=turns,
                        final_response="",
                        total_tool_calls=total_tool_calls,
                        total_latency_ms=total_latency,
                        reached_max_turns=False,
                        repaired_tool_calls=repaired_total,
                        error=(
                            f"degenerate_repetition: {unit!r} repeated {repeats} times on turn "
                            f"{turn_num}, {repeats * len(unit) / len(response_text):.0%} of "
                            f"{len(response_text)} chars — model looped, not a budget shortfall"
                        ),
                    )
                return ConversationResult(
                    turns=turns,
                    final_response="",
                    total_tool_calls=total_tool_calls,
                    total_latency_ms=total_latency,
                    reached_max_turns=False,
                    repaired_tool_calls=repaired_total,
                    error=(
                        f"truncated: finish_reason=length on turn {turn_num} "
                        f"({len(response_text)} chars, max_tokens={max_tokens})"
                    ),
                )

            # Tool calls that were emitted but did not parse used to be invisible:
            # parsed.errors was populated and never read, so the loop treated the
            # turn as "model is done" and judged the raw fragment.
            #
            # A single malformed call is not necessarily a failed conversation —
            # real agent loops tell the model its call did not parse and let it
            # retry. Do that once per conversation; a model that cannot produce
            # valid JSON on the second attempt is a genuine failure.
            if not parsed.tool_calls and parsed.errors:
                kinds = sorted({e.get("error", "unknown") for e in parsed.errors})
                # 400 chars hid the shape of three separate ornith defects for
                # several runs: the head looked identical while the defect that
                # mattered was past the cut. final_response carries the whole
                # text, but the error field is what a failure is read from.
                bad_raw = str(parsed.errors[0].get("raw", ""))[:UNPARSED_RAW_CHARS]
                turns.append(turn)

                if parse_retries < MAX_PARSE_RETRIES:
                    parse_retries += 1
                    logger.warning(
                        "Unparsed tool call on turn %d (%s); asking model to retry (%d/%d)",
                        turn_num,
                        ", ".join(kinds),
                        parse_retries,
                        MAX_PARSE_RETRIES,
                    )
                    messages.append(Message(role="assistant", content=response_text))
                    messages.append(
                        Message(
                            role="user",
                            content=(
                                "Your last tool call could not be parsed as JSON "
                                f"({', '.join(kinds)}). Re-issue it as a single well-formed "
                                '<tool_call> block: {"name": "<tool>", "arguments": {...}} '
                                "with every key and string value in double quotes."
                            ),
                        )
                    )
                    continue

                return ConversationResult(
                    turns=turns,
                    final_response="",
                    total_tool_calls=total_tool_calls,
                    total_latency_ms=total_latency,
                    reached_max_turns=False,
                    repaired_tool_calls=repaired_total,
                    error=(
                        f"unparsed_tool_call: {len(parsed.errors)} on turn {turn_num} "
                        f"({', '.join(kinds)}) after {parse_retries} retry; raw: {bad_raw!r}"
                    ),
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
                    nudge_reply = _call_model(
                        client, base_url, model_name, messages, temperature, max_tokens, chat_template_kwargs
                    )
                    nudged_text = nudge_reply.text
                    nudge_latency = (time.monotonic() - nudge_start) * 1000
                    total_latency += nudge_latency
                    nudge_parsed = extract_tool_calls(nudged_text, tools)
                    nudge_turn = TurnResult(
                        turn=turn_num + 1,
                        response=nudged_text,
                        parsed=nudge_parsed,
                        latency_ms=nudge_latency,
                        finish_reason=nudge_reply.finish_reason,
                        truncated=nudge_reply.truncated,
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
                    repaired_tool_calls=repaired_total,
                )

            # Execute tool calls and build response messages. Gemma-family
            # models frequently batch 50+ tool calls into a single response;
            # stop executing inside the turn once we hit max_tool_calls so
            # we don't balloon the message history past the server's
            # ctx-size before the synthesis nudge gets a chance to fire.
            # On the native path the calls live in a structured field, not in
            # the text, so they have to be replayed as tool_calls or the model
            # sees an empty assistant turn followed by unexplained results.
            native_entries = _native_entries(parsed) if reply.native_tool_calls is not None else None
            messages.append(
                Message(
                    role="assistant",
                    content=compact_tool_call_payloads(response_text, parsed, COMPACT_TOOL_CALL_OVER),
                    tool_calls=native_entries,
                )
            )

            dropped = 0
            for idx, tc in enumerate(parsed.tool_calls):
                if total_tool_calls >= max_tool_calls:
                    dropped = len(parsed.tool_calls) - len(turn.tool_responses)
                    break
                total_tool_calls += 1
                mock_result = mock_env.call(tc.name, tc.arguments)
                tool_resp = format_tool_response(tc.name, mock_result)
                turn.tool_responses.append({"name": tc.name, "arguments": tc.arguments, "result": mock_result})
                call_id = native_entries[idx].get("id") if native_entries and idx < len(native_entries) else None
                messages.append(Message(role="tool", content=tool_resp, tool_call_id=call_id))

            if dropped > 0:
                logger.info(
                    "Turn %d: cap reached mid-response, dropped %d remaining tool calls",
                    turn_num,
                    dropped,
                )

            turns.append(turn)

            if total_tool_calls >= max_tool_calls:
                logger.warning("Hit max_tool_calls (%d), nudging for synthesis", max_tool_calls)
                nudge_content = (
                    f"You have used all {max_tool_calls} available tool calls. "
                    "Do not call any more tools. Based on everything you've gathered, "
                    "write your final answer to the original question now."
                )
                total_latency += _try_nudge(
                    client,
                    base_url,
                    model_name,
                    messages,
                    temperature,
                    max_tokens,
                    nudge_content,
                    turn_num + 1,
                    turns,
                    chat_template_kwargs,
                    tools,
                )
                cap_nudged = True
                break

        # Reached max turns. If every turn emitted tool calls and never
        # produced a free-text answer, nudge once for synthesis — symmetric
        # with the max_tool_calls branch above. Skip if we already nudged
        # in the cap branch (regardless of whether that nudge succeeded).
        if not cap_nudged and turns and all(t.parsed.tool_calls for t in turns):
            logger.warning("Hit max_turns (%d) with no free-text turn, nudging for synthesis", max_turns)
            nudge_content = (
                f"You have used all {max_turns} available turns. "
                "Do not call any more tools. Based on everything you've gathered, "
                "write your final answer to the original question now."
            )
            total_latency += _try_nudge(
                client,
                base_url,
                model_name,
                messages,
                temperature,
                max_tokens,
                nudge_content,
                max_turns + 1,
                turns,
                chat_template_kwargs,
                tools,
            )

        # The synthesis nudge produces the final answer on this path, so a
        # truncated nudge is a truncated answer. The main loop checked for this
        # and the nudge path did not, so a response cut off at max_tokens was
        # still handed to the judge whenever a task reached its turn cap.
        if turns and turns[-1].truncated:
            return ConversationResult(
                turns=turns,
                final_response="",
                total_tool_calls=total_tool_calls,
                total_latency_ms=total_latency,
                reached_max_turns=True,
                repaired_tool_calls=repaired_total,
                error=(
                    f"truncated: finish_reason=length on synthesis nudge "
                    f"({len(turns[-1].response)} chars, max_tokens={max_tokens})"
                ),
            )

        final = _extract_best_final_response(turns)
        return ConversationResult(
            turns=turns,
            final_response=final,
            total_tool_calls=total_tool_calls,
            total_latency_ms=total_latency,
            reached_max_turns=True,
            repaired_tool_calls=repaired_total,
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


def detect_degenerate_repetition(text: str) -> tuple[int, str] | None:
    """Find a short substring repeated enough times to dominate the response.

    Returns (repeat_count, unit) or None. Distinguishes a model stuck in a loop
    from a model that legitimately needed more room, which the finish_reason
    alone cannot.
    """
    if len(text) < DEGENERATE_MIN_REPEATS:
        return None

    for unit_len in range(1, DEGENERATE_UNIT_MAX + 1):
        unit = text[-unit_len:]
        if not unit:
            continue
        repeats = 0
        pos = len(text)
        while pos >= unit_len and text[pos - unit_len : pos] == unit:
            repeats += 1
            pos -= unit_len
        if repeats >= DEGENERATE_MIN_REPEATS and (repeats * unit_len) / len(text) >= DEGENERATE_SHARE:
            return repeats, unit
    return None


def compact_tool_call_payloads(response_text: str, parsed: ParsedResponse, threshold: int) -> str:
    """Shrink large tool-call arguments before they enter conversation history.

    On coding_mcp_hard_01, `write_file` arguments were 67181 of the 68263
    characters of tool-call payload — 98% — and each file body sits in the
    context twice, as the assistant turn that emitted it and again as history.
    That put the history at roughly 34.5k tokens against a 32768 budget, and
    the task died at turn 26 with finish_reason=length.

    The emitted call is executed in full; only the copy kept in history is
    replaced with a summary. The model already knows what it wrote, and can
    re-read a file through the tools if it needs the text back.
    """
    if not parsed.tool_calls:
        return response_text

    compacted = response_text
    for call in parsed.tool_calls:
        if len(call.raw) <= threshold:
            continue
        described = ", ".join(
            f"{key}={value if len(str(value)) <= 60 else f'<{len(str(value))} chars>'}"
            for key, value in call.arguments.items()
        )
        # The summary must not be wrapped in <tool_call> tags. It first was, and
        # the model — seeing that shape attributed to its own earlier turns —
        # imitated it, emitting on turn 31 of coding_mcp_hard_01:
        #   <tool_call>
        #   [write_file issued and executed: path=..., content=<1003 chars>]
        #   </tool_call>
        # which is not JSON and failed to parse. Compaction rewrites the model's
        # own words, so whatever shape it leaves behind becomes an example the
        # model may copy. A plain note cannot be mistaken for a call.
        summary = f"[earlier turn: called {call.name} with {described}; executed successfully]"
        compacted = compacted.replace(f"<tool_call>\n{call.raw}\n</tool_call>", summary)
        if call.raw in compacted:
            compacted = compacted.replace(call.raw, summary)
    return compacted


def _try_nudge(
    client: httpx.Client,
    base_url: str,
    model_name: str,
    messages: list[Message],
    temperature: float,
    max_tokens: int,
    nudge_content: str,
    turn_number: int,
    turns: list[TurnResult],
    chat_template_kwargs: dict | None = None,
    tools: list[dict] | None = None,
) -> float:
    """Append a synthesis-nudge user message and collect the model's reply.

    Returns the nudge's latency in ms.

    HTTP failures are raised, not swallowed. This used to log a warning and
    fall back to the best already-collected turn, which silently converted
    "the model never produced an answer" into "score whatever fragment it
    happened to emit" — the same class of failure as accepting a truncated
    generation. A nudge that fails means there is no synthesis to score, so
    the task is a failed measurement and the outer handler records it.
    """
    messages.append(Message(role="user", content=nudge_content))
    nudge_start = time.monotonic()
    try:
        nudge_reply = _call_model(client, base_url, model_name, messages, temperature, max_tokens, chat_template_kwargs)
        nudged_text = nudge_reply.text
    except httpx.HTTPStatusError as e:
        body = e.response.text[:300]
        raise RuntimeError(
            f"synthesis nudge failed with HTTP {e.response.status_code} on turn {turn_number} "
            f"({len(messages)} messages, max_tokens={max_tokens}): {body}"
        ) from e
    nudge_latency = (time.monotonic() - nudge_start) * 1000
    nudge_parsed = extract_tool_calls(nudged_text, tools)
    turns.append(
        TurnResult(
            turn=turn_number,
            response=nudged_text,
            parsed=nudge_parsed,
            latency_ms=nudge_latency,
            finish_reason=nudge_reply.finish_reason,
            truncated=nudge_reply.truncated,
        )
    )
    return nudge_latency


def _extract_best_final_response(turns: list[TurnResult]) -> str:
    """Find the model's final answer when max_turns is reached.

    Only turns that made no tool call can hold an answer. Prose emitted
    alongside a tool call is narration — "Let me write the config now:" — and
    scoring it as a considered response is how coding_artemis_medium_01 came to
    be judged on "Only 4 tests ran ... Let me check the environment and fix:".

    The previous rule took any turn's prose once it exceeded twenty characters,
    so a transitional sentence from mid-conversation was indistinguishable from
    a synthesis. When no tool-free turn exists the answer is that none was
    produced, which the judge should see as such rather than as a short one.
    """
    answer_turns = [t for t in turns if not t.parsed.tool_calls]

    for turn in reversed(answer_turns):
        text = TOOL_CALL_TAG_RE.sub("", turn.response).strip()
        if len(text) > MIN_FINAL_ANSWER_CHARS:
            return text

    for turn in reversed(answer_turns):
        text = TOOL_CALL_TAG_RE.sub("", turn.response).strip()
        if text:
            return text

    tc_count = sum(len(t.tool_responses) for t in turns)
    return (
        f"[No final answer produced: every one of {len(turns)} turns ended in a tool call "
        f"({tc_count} calls total), so the model never stopped to answer]"
    )


def _native_entries(parsed: ParsedResponse) -> list[dict]:
    """Rebuild the server's tool_call entries from a parsed native reply.

    Each ToolCall.raw holds the entry verbatim, including the id the tool
    result must reference. An id is synthesised when the server omitted one,
    since the protocol pairs results to calls by it.
    """
    entries: list[dict] = []
    for i, tc in enumerate(parsed.tool_calls):
        try:
            entry = json.loads(tc.raw)
        except (json.JSONDecodeError, TypeError):
            entry = {"type": "function", "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)}}
        if not isinstance(entry, dict):
            entry = {"type": "function", "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)}}
        entry.setdefault("id", f"call_{i}")
        entries.append(entry)
    return entries


def _parse_native_tool_calls(msg: dict, model_name: str) -> list[ToolCall] | None:
    """Convert the server's structured tool_calls into ToolCall objects.

    Returns None when the reply carries none, so the caller can fall back to
    parsing text — a native-tools model still answers in prose on its final
    turn. `arguments` is a JSON string per the OpenAI schema, but some servers
    hand back an object, so both are accepted.
    """
    raw = msg.get("tool_calls")
    if not raw:
        return None
    calls: list[ToolCall] = []
    for entry in raw:
        fn = entry.get("function") or {}
        name = fn.get("name")
        if not name:
            continue
        args = fn.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                logger.warning("Native tool call from %s had unparseable arguments: %r", model_name, args)
                continue
        if not isinstance(args, dict):
            args = {}
        calls.append(ToolCall(name=name, arguments=args, raw=json.dumps(entry)))
    return calls or None


def _call_model(
    client: httpx.Client,
    base_url: str,
    model_name: str,
    messages: list[Message],
    temperature: float,
    max_tokens: int,
    chat_template_kwargs: dict | None = None,
    tools: list[dict] | None = None,
    native_tools: bool = False,
) -> ModelReply:
    """Send messages to the model and return the response text.

    Some llama-server builds route Qwen-style thinking output to a separate
    `reasoning_content` field when the model emits `<think>...</think>` blocks.
    If `content` is empty, fall back to `reasoning_content` so we don't treat
    a thought-only answer as a silent stall. Also logs the raw message keys
    and finish_reason once so diagnostic info lands in the run log.
    """
    payload: dict = {
        "model": model_name,
        "messages": [m.to_payload() for m in messages],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if chat_template_kwargs:
        payload["chat_template_kwargs"] = chat_template_kwargs
    # Native tool calling: hand the server the schemas so it renders the model's
    # own template and parses the reply back into structured calls. Only for
    # models flagged for it — this changes what the model is asked to do.
    if native_tools and tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
    resp = client.post(f"{base_url}/v1/chat/completions", json=payload)
    resp.raise_for_status()
    data = resp.json()
    choice = data["choices"][0]
    msg = choice.get("message", {})
    content = msg.get("content") or ""
    finish = choice.get("finish_reason")

    if native_tools:
        native = _parse_native_tool_calls(msg, model_name)
        if native is not None:
            # Empty content alongside tool calls is correct here, not a stall,
            # so the reasoning_content fallback below must not fire and hand the
            # judge a chain of thought in place of the call.
            return ModelReply(text=content, finish_reason=finish, native_tool_calls=native)

    if not content.strip():
        reasoning = msg.get("reasoning_content") or ""
        logger.warning(
            "Empty content from %s (finish=%s, msg_keys=%s, reasoning_len=%d)",
            model_name,
            finish,
            sorted(msg.keys()),
            len(reasoning),
        )
        if reasoning.strip():
            return ModelReply(text=reasoning, finish_reason=finish)
    if finish == "length":
        logger.warning(
            "Truncated generation from %s (finish=length, content_len=%d) — raise max_tokens for this task",
            model_name,
            len(content),
        )
    return ModelReply(text=content, finish_reason=finish)
