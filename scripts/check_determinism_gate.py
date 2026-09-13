#!/usr/bin/env python3
"""Evaluate the coding-determinism acceptance gate over repeat runs.

Give it the run IDs of N back-to-back repeats of the same suite and it reports
whether the harness is reproducible, against four conditions:

  G1  the ordered (turn, call_index, tool_name, arguments) trace is
      byte-identical across every run
  G2  no task flips between completed and failed
  G3  no deterministic criterion moves (anything that is not judge_rubric)
  G4  weighted_score spread stays within --spread

G1 carries the statistical power. G2 alone is underpowered at n=3: given a
diverged trace the historical flip rate is ~26% per pair, so a handful of pairs
would be expected to show zero flips even with nothing fixed.

G4's default of 0.09 is the tightest bound no judge-only case in the recorded
history would have failed. Across 37 groups / 99 runs whose model output was
byte-identical, the worst weighted-score spread was 0.0875 and every criterion
that moved was judge_rubric.

A G1 failure prints the first differing byte of the tool result that PRECEDES
the divergence, because that result is the leak. The gate is only honestly
waived by showing that byte carries no clock, address, hostname or temp path.
"""

from __future__ import annotations

import argparse
import re
import sqlite3
import sys
from collections import defaultdict

DEFAULT_DB = "results/runs/eval_results.db"

# Everything that is not the judge is supposed to be a pure function of the
# model's output. Naming the exception rather than listing the deterministic
# methods means a method added later is covered without editing this script.
JUDGE_METHOD = "judge_rubric"

# Substrings that make a divergence self-evidently environmental. Used only to
# annotate a G1 failure — never to excuse one automatically.
VOLATILE_HINTS = (
    "0x",
    "HOSTNAME",
    "/tmp/",
    "elapsed",
    "ms)",
    "s)",
)

# Results are stored as JSON, so a newline inside one arrives as the two
# characters \ and n. Splitting on real newlines alone would see a single line
# and call every result a permutation of itself.
_LINE_SPLIT_RE = re.compile(r"\\n|\n")


def _split_lines(text: str) -> list[str]:
    return _LINE_SPLIT_RE.split(text)


def line_permutation(a: str, b: str) -> bool:
    """True when two results hold exactly the same lines in a different order.

    Go randomizes map iteration per process, so a table-driven test backed by a
    map emits its subtest blocks permuted between runs. That is the order of
    lines, not a token inside one, so no substitution reaches it and
    _normalize_volatile deliberately does not try — see its docstring and
    test_sandbox_normalization.test_go_map_iteration_order_remains_a_known_gap.

    Distinguishing it matters because the window-limited VOLATILE_HINTS scan
    reports NONE here, which reads as "this may be real" and points an operator
    at the one surface already measured, documented and declined.

    Multiset, not set: a line emitted three times and twice is a real
    divergence. Identical results are not a permutation — nothing was reordered.
    """
    if a == b:
        return False
    return sorted(_split_lines(a)) == sorted(_split_lines(b))


def permutation_shape(a: str, b: str) -> tuple[int, int]:
    """(total lines, positions whose line changed). Positions, not lines moved."""
    la, lb = _split_lines(a), _split_lines(b)
    moved = sum(1 for x, y in zip(la, lb, strict=False) if x != y)
    return len(la), moved


def fetch_pairs(conn: sqlite3.Connection, runs: list[str], dimension: str | None) -> list[tuple[str, str]]:
    placeholders = ",".join("?" * len(runs))
    q = f"SELECT DISTINCT model_name, task_id FROM task_results WHERE run_id IN ({placeholders})"
    args: list[str] = list(runs)
    if dimension:
        q += " AND dimension = ?"
        args.append(dimension)
    return [(r[0], r[1]) for r in conn.execute(q, args)]


def trace(conn: sqlite3.Connection, run: str, model: str, task: str) -> list[tuple]:
    """The ordered call trace, with 'no arguments' spelled one way.

    A NULL and a '{}' both mean the call carried no arguments — the storage
    layer has written each at different times — so comparing them raw would
    report a divergence where the model did the same thing twice.
    """
    rows = conn.execute(
        "SELECT turn, call_index, tool_name, arguments FROM tool_calls "
        "WHERE run_id=? AND model_name=? AND task_id=? ORDER BY turn, call_index",
        (run, model, task),
    )
    return [(t, i, n, (a if a not in (None, "null", "{}") else "{}")) for t, i, n, a in rows]


def results(conn: sqlite3.Connection, run: str, model: str, task: str) -> tuple:
    row = conn.execute(
        "SELECT status, weighted_score, total_tool_calls FROM task_results "
        "WHERE run_id=? AND model_name=? AND task_id=?",
        (run, model, task),
    ).fetchone()
    return row if row else (None, None, None)


def deterministic_scores(conn: sqlite3.Connection, run: str, model: str, task: str) -> dict[str, float]:
    return {
        d: n
        for d, n in conn.execute(
            "SELECT dimension, normalized FROM score_details "
            "WHERE run_id=? AND model_name=? AND task_id=? AND method != ?",
            (run, model, task, JUDGE_METHOD),
        )
    }


def preceding_result(conn: sqlite3.Connection, run: str, model: str, task: str, turn: int, idx: int) -> str:
    """The tool result the model saw immediately before the diverging call."""
    row = conn.execute(
        "SELECT result FROM tool_calls "
        "WHERE run_id=? AND model_name=? AND task_id=? "
        "AND (turn < ? OR (turn = ? AND call_index < ?)) "
        "ORDER BY turn DESC, call_index DESC LIMIT 1",
        (run, model, task, turn, turn, idx),
    ).fetchone()
    return row[0] if row and row[0] else ""


def first_diff(a: str, b: str) -> tuple[int, str]:
    """Position of the first differing character, and how the two differ.

    "prefix" must be reported distinctly from "differs": when one result is a
    truncation of the other, the surrounding context reads as identical and a
    bare position looks like a false alarm.
    """
    for i, (x, y) in enumerate(zip(a, b, strict=False)):
        if x != y:
            return i, "differs"
    if len(a) != len(b):
        return min(len(a), len(b)), f"prefix (lengths {len(a)} vs {len(b)})"
    return 0, "identical"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="Run IDs of the repeats (2 or more)")
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--dimension", default="coding", help="Restrict to one dimension (default: coding)")
    ap.add_argument("--spread", type=float, default=0.09, help="G4 max weighted_score spread (default 0.09)")
    ap.add_argument("--context", type=int, default=60, help="Chars of context around a G1 diff")
    args = ap.parse_args()

    if len(args.runs) < 2:
        print("Need at least 2 runs to compare.", file=sys.stderr)
        return 2

    conn = sqlite3.connect(args.db)
    pairs = sorted(fetch_pairs(conn, args.runs, args.dimension))
    if not pairs:
        print("No matching (model, task) pairs found.", file=sys.stderr)
        return 2

    failures: dict[str, list[str]] = defaultdict(list)
    incomplete: list[str] = []
    rows = []

    for model, task in pairs:
        label = f"{model} / {task}"
        statuses, scores, traces, dets = [], [], [], []
        missing = False
        for run in args.runs:
            status, score, counted = results(conn, run, model, task)
            if status is None:
                missing = True
                break
            statuses.append(status)
            scores.append(score)
            traces.append(trace(conn, run, model, task))
            dets.append(deterministic_scores(conn, run, model, task))
            # A trace that was counted but not stored cannot be compared. Before
            # the persistence fix this was every failed task.
            if counted and not traces[-1]:
                incomplete.append(f"{label} [{run}]: {counted} calls counted, 0 stored")
        if missing:
            incomplete.append(f"{label}: absent from at least one run")
            continue

        g1 = len({repr(t) for t in traces}) == 1
        g2 = len(set(statuses)) == 1
        keys = set().union(*(d.keys() for d in dets)) if dets else set()
        g3 = all(len({d.get(k) for d in dets}) == 1 for k in keys)
        numeric = [s for s in scores if s is not None]
        spread = (max(numeric) - min(numeric)) if numeric else 0.0
        g4 = spread <= args.spread

        rows.append((label, g1, g2, g3, spread, g4, statuses))

        if not g1:
            base = traces[0]
            perm_only = False
            for other_i, other in enumerate(traces[1:], start=1):
                for i in range(max(len(base), len(other))):
                    a = base[i] if i < len(base) else None
                    b = other[i] if i < len(other) else None
                    if a != b:
                        turn, idx = (a or b)[0], (a or b)[1]
                        ra = preceding_result(conn, args.runs[0], model, task, turn, idx)
                        rb = preceding_result(conn, args.runs[other_i], model, task, turn, idx)
                        p, kind = first_diff(ra, rb)
                        lo, hi = max(0, p - args.context), p + args.context
                        hint = [h for h in VOLATILE_HINTS if h in ra[lo:hi] or h in rb[lo:hi]]
                        if not ra and not rb:
                            note = "no preceding result stored — trace unavailable, not evidence"
                        elif kind == "identical":
                            note = "preceding result identical — the leak is further upstream"
                        elif line_permutation(ra, rb):
                            total, moved = permutation_shape(ra, rb)
                            perm_only = True
                            note = (
                                f"line-permutation only: {total} lines, {moved} positions reordered, "
                                "none added or removed — the known Go map-iteration exception, "
                                "not a leak"
                            )
                        else:
                            note = f"volatile markers present: {hint or 'NONE — investigate, this may be real'}"
                        failures["G1_detail"].append(
                            f"  {label}: first divergence at turn {turn} call {idx}\n"
                            f"    preceding result {kind} at byte {p}\n"
                            f"      run A: ...{ra[lo:hi]!r}...\n"
                            f"      run B: ...{rb[lo:hi]!r}...\n"
                            f"    {note}"
                        )
                        break
                break
            # Annotated, never excused: a permutation is still a G1 failure.
            failures["G1"].append(label + (" [line-permutation only]" if perm_only else ""))
        if not g2:
            failures["G2"].append(f"{label}: {' -> '.join(statuses)}")
        if not g3:
            moved = [k for k in keys if len({d.get(k) for d in dets}) != 1]
            failures["G3"].append(f"{label}: {', '.join(sorted(moved))}")
        if not g4:
            failures["G4"].append(f"{label}: spread {spread:.3f} > {args.spread}")

    width = max(len(r[0]) for r in rows) if rows else 20
    print(f"\nDeterminism gate over {len(args.runs)} runs: {', '.join(args.runs)}")
    print(f"Dimension: {args.dimension or 'all'}    G4 threshold: {args.spread}\n")
    print(f"{'pair'.ljust(width)}  G1    G2    G3    spread  G4")
    print("-" * (width + 34))
    for label, g1, g2, g3, spread, g4, _ in rows:
        mark = lambda ok: " ok " if ok else "FAIL"  # noqa: E731
        print(f"{label.ljust(width)}  {mark(g1)}  {mark(g2)}  {mark(g3)}  {spread:6.3f}  {mark(g4)}")

    if incomplete:
        print("\nUNCOMPARABLE (gate cannot be evaluated on these):")
        for line in incomplete:
            print(f"  {line}")

    for gate in ("G1", "G2", "G3", "G4"):
        if failures[gate]:
            print(f"\n{gate} FAILURES:")
            for line in failures[gate]:
                print(f"  {line}")
    if failures["G1_detail"]:
        print("\nG1 divergence origins (the leak is in the PRECEDING result):")
        for line in failures["G1_detail"]:
            print(line)

    hard = any(failures[g] for g in ("G1", "G2", "G3", "G4")) or bool(incomplete)
    print("\n" + ("GATE FAILED" if hard else "GATE PASSED"))
    if incomplete and not any(failures[g] for g in ("G1", "G2", "G3", "G4")):
        print("(failed only because some pairs were uncomparable — fix those, then re-run)")
    return 1 if hard else 0


if __name__ == "__main__":
    sys.exit(main())
