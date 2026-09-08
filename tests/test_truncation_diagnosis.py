"""A truncated generation is usually a loop, not a budget shortfall.

Across the whole results DB there are 64 truncation failures. 57 of them — 89%,
spanning 11 different models — are the model repeating itself until the budget
runs out. Only 6 are genuinely long output, and 1 is a reasoning overrun.

The error string nonetheless ends `max_tokens=32768` and the log line says
"raise max_tokens for this task". That advice is wrong 89% of the time, and
CLAUDE.md already records the 24576 -> 32768 raise failing exactly this way:
gemma4 went from 64k to 85k characters and failed identically.

`detect_degenerate_repetition` misses these because it requires an exact,
contiguous, suffix-anchored unit of at most 8 characters repeated at least 200
times. The real failures do not have that shape:

    gemma4      85236 chars, 2 lines, one runaway <|tool_call> — unit >> 8 chars
    qwen3.6    103437 chars, a Go table row x39 — near-duplicates, not exact
    lfm2.5-2.6b 147581 chars, one prose line x88, 93% dup — interleaved, not a suffix

Compression ratio catches all three. Measured over the DB: truncation failures
have a median ratio of 0.040, while 400 real written files have a median of
0.303 and a *minimum* of 0.149. The separation is roughly 5x, and this test only
ever runs on output that already hit finish_reason=length.
"""

import pathlib

import nite_eval.conversation_runner
from nite_eval.conversation_runner import (
    DEGENERATE_COMPRESSION_RATIO,
    compression_ratio,
    detect_degenerate_repetition,
    diagnose_truncation,
)


def test_exact_suffix_loop_still_caught_by_the_fast_path():
    """The original detector's case must keep working and keep its message."""
    text = "real content here. " + "\\n" * 12_000
    assert detect_degenerate_repetition(text) is not None


def test_line_repeated_with_interleaving_is_caught():
    """lfm2.5-2.6b: one prose line 88 times, 93% duplicate lines, not a suffix."""
    body = "".join(
        f"So when we set `client._client.get = AsyncMock()` the patch applies.\nstep {i}\n" for i in range(300)
    )
    assert compression_ratio(body) < DEGENERATE_COMPRESSION_RATIO
    assert diagnose_truncation(body, 32768).startswith("degenerate_repetition")


def test_near_duplicate_rows_are_caught():
    """qwen3.6: a Go table row 39 times with only the name changing."""
    body = "".join(f'\t\t\t{{Name: "s{i}", BaseURL: us.URL, AuthType: "none"}},\n' for i in range(2000))
    assert compression_ratio(body) < DEGENERATE_COMPRESSION_RATIO
    assert diagnose_truncation(body, 32768).startswith("degenerate_repetition")


def test_runaway_single_line_is_caught():
    """gemma4: 85236 chars across 2 lines, a tool-call token repeated."""
    body = "<|tool_call>call<tool_call>" * 3000
    assert compression_ratio(body) < DEGENERATE_COMPRESSION_RATIO
    assert diagnose_truncation(body, 32768).startswith("degenerate_repetition")


def test_real_code_is_never_called_degenerate():
    """287 distinct real written files had a minimum ratio of 0.1468.

    Fixture is this repo's own source rather than a synthetic loop: templated
    filler compresses harder than real code does and would not test anything.
    """
    source = pathlib.Path(nite_eval.conversation_runner.__file__).read_text()
    assert len(source) > 20_000
    assert compression_ratio(source) >= DEGENERATE_COMPRESSION_RATIO
    assert diagnose_truncation(source, 32768).startswith("truncated:")


def test_empty_content_is_a_reasoning_overrun_not_a_content_truncation():
    """muse-glimmer on coding_mcp_hard_01: turn 14, 0 chars, finish=length.

    Points at the model's reasoning switch, not at max_tokens — a different
    knob entirely, so it must not read as a content truncation.
    """
    msg = diagnose_truncation("", 32768)
    assert msg.startswith("reasoning_overrun")
    assert "max_tokens" not in msg.split("reasoning_overrun")[1].split(":")[0]


def test_genuine_long_output_still_reports_max_tokens():
    """The 6 real budget shortfalls must keep pointing at the budget."""
    msg = diagnose_truncation(
        "unique prose. " + "".join(f"paragraph {i} about {i * 7} things.\n" for i in range(60)), 4096
    )
    assert msg.startswith("truncated:")
    assert "max_tokens=4096" in msg


def test_ratio_is_not_computed_on_tiny_strings():
    """zlib overhead dominates short input and would flag it as degenerate."""
    assert diagnose_truncation("short", 4096).startswith("truncated:")


def test_message_reports_the_ratio_so_the_call_can_be_checked():
    """Only the compression path reports a ratio; the exact-suffix path names its unit."""
    body = "".join(f"line {i % 3} of the loop\n" for i in range(4000))
    msg = diagnose_truncation(body, 32768)
    assert msg.startswith("degenerate_repetition")
    assert "compress" in msg
