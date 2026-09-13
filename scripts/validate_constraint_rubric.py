#!/usr/bin/env python3
"""P4 gate: does `constraint_handling` actually discriminate?

The planning redesign puts 0.20 of every planning task's weight on a judge
criterion asking whether the plan routed around a limit the tools reported. That
is worth nothing unless the judge can tell a plan that handled the constraint
from one that ignored it — and especially from a confident plan written without
calling any tools, which is 22.8% of planning runs.

This asks the live judge directly, before any task YAML changes and before any
weight rides on it. Same method that proved the code-evidence anchor: fixed
responses, real judge, read the numbers.

Four responses against one constraint (Supabase edge function wall clock 9.5s;
Claude Vision on full-size label photos 14.2s p50 / 31.4s p95, so a synchronous
scan cannot work):

  A handles      cites the limit and goes async / downscales client-side
  B mentions     cites the limit, then plans the synchronous flow anyway
  C ignores      called the tools, plans synchronous, never mentions the limit
  D ungrounded   called NO tools, confident synchronous plan

D is the one that matters. It reaches the judge with `NO_TOOLS_CALLED` as its
evidence, which only became true once build_tool_evidence learned to state the
absence — before that it got an empty section and the judge filled the gap from
the task spec.

PASS when A beats D by at least --margin, and A beats C. Otherwise the rubric
does not carry 0.20 and the redesign needs a different lever.

Requires the judges running:
    uv run python scripts/validate_constraint_rubric.py
    uv run python scripts/validate_constraint_rubric.py --samples 5 --margin 1.5

Judge URLs and model names come from config/eval_config.yaml's `judge:` block,
the same source the orchestrator uses, so this cannot drift from a real run.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nite_eval.evidence import build_tool_evidence  # noqa: E402
from nite_eval.judge import JudgeResult, RoutedJudgeClient  # noqa: E402
from nite_eval.model_manager import check_health  # noqa: E402
from nite_eval.rubrics import get_rubric  # noqa: E402
from nite_eval.task_loader import load_tasks  # noqa: E402

DEFAULT_CONFIG = "config/eval_config.yaml"
DIMENSION = "constraint_handling"
TASK_ID = "planning_wine_easy_01"

# The constraint as the rewritten fixture will report it: measurements, no
# recommendation. The judge has to draw the conclusion, not read it.
TOOL_CALLS = [
    {
        "name": "check_dependency",
        "arguments": {"tech_a": "claude vision", "tech_b": "edge function"},
        "result": {
            "edge_function_wall_clock_limit_s": 9.5,
            "observed_vision_latency_s": {"p50": 14.2, "p95": 31.4},
            "measured_on": "4032x3024 label photographs",
        },
    },
    {
        "name": "estimate_effort",
        "arguments": {"task": "scan label edge function"},
        "result": {"estimate_hours": 8, "complexity": "high"},
    },
]

A_HANDLES = (
    "Week 1: Supabase schema (wines, bottles, racks, positions, events) and auth with RLS. "
    "Then the scan path — and it cannot be a synchronous edge function. Edge functions cut off "
    "at 9.5s, and Claude Vision on full-size label photos runs 14.2s at p50 and 31.4s at p95, so "
    "the median request would time out, never mind the tail. Instead: the app downscales the "
    "photo on device and uploads to Storage; a trigger enqueues a job row; a worker calls Claude "
    "Vision and writes the result back; the client subscribes to Realtime for completion. "
    "Week 2: inventory list, rack grid with drag-to-place, consume-bottle flow.\n\n"
    "Critical path: schema -> auth -> async scan pipeline -> inventory UI. The async hop is the "
    "risk; if the worker proves slow, fall back to a direct client-to-Anthropic call with the key "
    "brokered by an edge function, which keeps the long call off the 9.5s budget."
)

B_MENTIONS_ONLY = (
    "Week 1: Supabase schema and auth with RLS policies. Build the scan-label edge function that "
    "takes the uploaded image, calls Claude Vision, and writes the wine record. Note that edge "
    "functions have a 9.5s wall clock limit and Claude Vision measured 14.2s p50 on label photos. "
    "Week 2: inventory list view, rack grid with drag-to-place, consume-bottle flow and stats.\n\n"
    "Critical path: schema -> auth -> scan label -> inventory UI. "
    "Risk: Claude Vision accuracy on damaged labels varies."
)

C_IGNORES = (
    "Week 1: Set up Supabase, design the schema (wines, bottles, racks, positions, events), "
    "configure auth and RLS. Build the scan-label edge function calling Claude Vision and "
    "returning the identified wine to the app. Week 2: inventory list, bottle detail, rack grid "
    "with drag-to-place, consume-bottle flow.\n\n"
    "Critical path: schema -> auth -> scan label -> inventory UI. "
    "Risk: Claude Vision accuracy on wine labels varies by condition."
)

D_UNGROUNDED = (
    "Here is a clean 2-week plan.\n\n"
    "Week 1: Supabase project, schema for wines/bottles/racks/positions/events, auth with RLS. "
    "Scan-label edge function: accept the image, call Claude Vision, match or create the wine, "
    "return it. Edge functions handle external API calls comfortably and Claude Vision is fast, "
    "so a straightforward synchronous handler is the right call here — no queue needed for a POC. "
    "Week 2: inventory list, rack grid with drag-to-place, consume-bottle flow, basic stats.\n\n"
    "Critical path: schema -> auth -> scan label -> inventory UI. "
    "Risks: Claude Vision label accuracy, and edge function cold starts."
)


class _Turn:
    def __init__(self, tool_responses):
        self.tool_responses = tool_responses


class _Conv:
    def __init__(self, turns):
        self.turns = turns


def _evidence(called: bool, tools) -> str:
    """Exactly what the orchestrator would hand the judge, via the real builder."""
    return build_tool_evidence(_Conv([_Turn(TOOL_CALLS if called else [])]), tools=tools)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=DEFAULT_CONFIG, help="Read the judge block from here")
    ap.add_argument("--samples", type=int, default=3, help="Judge samples per response (default 3)")
    ap.add_argument("--margin", type=float, default=1.0, help="Required A-over-D gap on the 1-5 scale")
    args = ap.parse_args()

    # Build the client exactly the way the orchestrator does. The judges run as
    # two separate llama-servers on different ports — reward-anything on 9091,
    # flow-judge on 9092 — so a single hardcoded port points both at one model.
    judge_cfg = (yaml.safe_load(Path(args.config).read_text()) or {}).get("judge", {})
    reward_url = judge_cfg.get("reward_anything_url") or judge_cfg.get("base_url", "")
    health_url = reward_url.removesuffix("/v1")
    if not check_health(health_url):
        print(f"No judge server at {health_url} (constraint_handling routes to reward-anything).", file=sys.stderr)
        print("Start the judges, then re-run.", file=sys.stderr)
        return 2

    tasks = [t for t in load_tasks() if t.id == TASK_ID]
    if not tasks:
        print(f"Task {TASK_ID} not found.", file=sys.stderr)
        return 2
    task = tasks[0]

    scenarios = [
        ("A handles", A_HANDLES, True),
        ("B mentions only", B_MENTIONS_ONLY, True),
        ("C ignores", C_IGNORES, True),
        ("D ungrounded, no tools", D_UNGROUNDED, False),
    ]

    rubric = get_rubric(DIMENSION)
    print(f"\nRubric for {DIMENSION}:\n  {rubric}\n")
    print(f"Task: {task.id}   judge: {reward_url}   samples: {args.samples}   margin: {args.margin}\n")
    print(f"{'scenario':26s} {'score':>6s}  reasoning")
    print("-" * 100)

    scores: dict[str, float] = {}
    with RoutedJudgeClient(
        base_url=judge_cfg.get("base_url", reward_url),
        flow_judge_model=judge_cfg.get("flow_judge_model", "flow-judge"),
        reward_anything_model=judge_cfg.get("reward_anything_model", "reward-anything"),
        flow_judge_url=judge_cfg.get("flow_judge_url"),
        reward_anything_url=judge_cfg.get("reward_anything_url"),
        temperature=judge_cfg.get("temperature", 0.1),
        max_tokens=judge_cfg.get("max_tokens", 2048),
        timeout=120.0,
    ) as judge:
        for label, response, called_tools in scenarios:
            result = judge.evaluate_with_averaging(
                dimension=DIMENSION,
                rubric=rubric,
                task_description=task.user_message,
                model_response=response,
                n_runs=args.samples,
                evidence=_evidence(called_tools, task.tools),
            )
            if not isinstance(result, JudgeResult):
                print(f"{label:26s} {'ERR':>6s}  {result.error}")
                return 1
            scores[label] = result.score
            print(f"{label:26s} {result.score:6.2f}  {result.reasoning[:70]}")

    a, c, d = scores["A handles"], scores["C ignores"], scores["D ungrounded, no tools"]
    gap_ad, gap_ac = a - d, a - c
    print("\n" + "-" * 100)
    print(f"A - D (handled vs confident ungrounded): {gap_ad:+.2f}   required >= {args.margin}")
    print(f"A - C (handled vs ignored):              {gap_ac:+.2f}   required > 0")

    ok = gap_ad >= args.margin and gap_ac > 0
    print(
        "\n"
        + (
            "GATE PASSED — constraint_handling discriminates, 0.20 weight is justified"
            if ok
            else "GATE FAILED — the criterion does not separate these; do not weight it at 0.20"
        )
    )
    if not ok and gap_ad < args.margin:
        print("  The ungrounded plan scored too close to the handled one. That is the case the")
        print("  redesign exists to catch, so a different lever is needed, not a bigger weight.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
