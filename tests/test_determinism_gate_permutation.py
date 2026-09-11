"""The gate must tell a reordering apart from a leak.

A G1 failure means two repeats produced different call traces, and the note the
gate prints is what an operator uses to decide whether to investigate or to
waive. Before this, a pure line permutation printed "volatile markers present:
NONE — investigate, this may be real", which points at the one surface the
project has already measured, documented and deliberately declined to fix.

The data here is the real measured case: ornith-1.5-35b-a3b / coding_mcp_easy_01,
run-20260909-030053 against run-20260909-034638, where TestLoadInvalidURL's four
subtests came back in a different order because Go randomizes map iteration per
process. Same 40 lines, 8 positions moved, nothing added or removed.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_GATE = Path(__file__).resolve().parents[1] / "scripts" / "check_determinism_gate.py"
_spec = importlib.util.spec_from_file_location("check_determinism_gate", _GATE)
assert _spec and _spec.loader
gate = importlib.util.module_from_spec(_spec)
sys.modules["check_determinism_gate"] = gate
_spec.loader.exec_module(gate)


def _go_output(cases: list[str]) -> str:
    """A Go test result as it is actually stored: JSON, so newlines are \\n."""
    runs = "".join(f"=== RUN   TestLoadInvalidURL/{c}\\n" for c in cases)
    passes = "".join(f"    --- PASS: TestLoadInvalidURL/{c} (0.00s)\\n" for c in cases)
    return (
        '{"content": {"exit_code": 0, "stdout": "=== RUN   TestLoadInvalidURL\\n'
        f"{runs}--- PASS: TestLoadInvalidURL (0.00s)\\n{passes}"
        'ok  \\tmcpconfig/config\\t0.000s\\n"}}'
    )


ORDER_A = ["no_scheme", "no_host", "bad_scheme", "spaces"]
ORDER_B = ["spaces", "no_scheme", "no_host", "bad_scheme"]


def test_permuted_go_subtests_are_recognised():
    a, b = _go_output(ORDER_A), _go_output(ORDER_B)
    assert a != b, "the fixture must actually differ, or the test proves nothing"
    assert gate.line_permutation(a, b)


def test_permutation_survives_real_stored_json_escaping():
    """Results reach the gate as JSON, so a newline is the two chars \\ and n.

    Splitting on real newlines alone would see one line and call every result a
    permutation of itself.
    """
    a, b = _go_output(ORDER_A), _go_output(ORDER_B)
    assert "\\n" in a and "\n" not in a
    assert gate.line_permutation(a, b)


def test_identical_results_are_not_a_permutation():
    """Nothing was reordered, so the note must not claim a reordering."""
    a = _go_output(ORDER_A)
    assert not gate.line_permutation(a, a)


def test_a_volatile_token_is_not_a_permutation():
    """The deno case that 6a37967 fixed: same line order, a changed token.

    This is the failure mode the gate exists to catch. It must never be
    reported as a benign reordering.
    """
    a = "ok \\u001b[32mok\\u001b[0m (4ms)\\nnon-object body -> 400\\n"
    b = "ok \\u001b[32mok\\u001b[0m (3ms)\\nnon-object body -> 400\\n"
    assert not gate.line_permutation(a, b)


def test_added_or_removed_lines_are_not_a_permutation():
    """A model that wrote one more test is a real divergence, not a reordering."""
    a = _go_output(ORDER_A)
    b = _go_output([*ORDER_A, "empty"])
    assert not gate.line_permutation(a, b)


def test_a_line_repeated_a_different_number_of_times_is_not_a_permutation():
    """Multiset, not set: three PASS lines and two are not the same output."""
    a = "PASS\\nPASS\\nPASS\\n"
    b = "PASS\\nPASS\\n"
    assert not gate.line_permutation(a, b)


def test_moved_line_count_reports_positions_not_lines():
    a, b = _go_output(ORDER_A), _go_output(ORDER_B)
    total, moved = gate.permutation_shape(a, b)
    assert total == len(gate._split_lines(a))
    assert 0 < moved <= total
    # ORDER_A -> ORDER_B rotates all four subtests in both the RUN and PASS
    # blocks, so every one of the eight lines lands somewhere new.
    assert moved == 8
