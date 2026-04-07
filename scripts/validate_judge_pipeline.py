#!/usr/bin/env python3
"""Validate the judge pipeline end-to-end with eval tasks.

Loads tasks, uses synthetic model responses, and runs judge-rubric dimensions
through RoutedJudgeClient to verify:
  1. Routing works (Flow-Judge gets agentic dims, RewardAnything gets the rest)
  2. Score parsing succeeds
  3. Scores are in valid range (1, 3, or 5)
  4. The full task→judge→score pipeline connects

Requires judge llama-swap running on :9091 with both models configured.

Usage:
    # Start judge llama-swap first:
    CUDA_VISIBLE_DEVICES=1 $LLAMA_SWAP_BIN \
      --config config/judge_swap_config.yaml --listen :9091

    # Run validation:
    uv run python scripts/validate_judge_pipeline.py
    uv run python scripts/validate_judge_pipeline.py --tasks 3  # limit tasks
    uv run python scripts/validate_judge_pipeline.py --dimension agentic  # filter
"""

import argparse
import logging
import sys

from nite_eval.judge import (
    FLOW_JUDGE_DIMENSIONS,
    JudgeResult,
    RoutedJudgeClient,
)
from nite_eval.model_manager import check_health
from nite_eval.task_loader import TaskDefinition, load_tasks
from rich.console import Console
from rich.table import Table

logger = logging.getLogger(__name__)
console = Console()

JUDGE_PORT = 9091

# Rubrics for judge dimensions — used when task scoring config says method: judge_rubric
# These match the calibration rubrics from run_calibration.py
JUDGE_RUBRICS: dict[str, str] = {
    "coverage": (
        "1 (Poor): Misses most aspects of the question\n"
        "3 (Acceptable): Covers most aspects adequately\n"
        "5 (Excellent): Comprehensive, addresses all aspects with depth"
    ),
    "synthesis": (
        "1 (Poor): Pure list of facts, no connections drawn\n"
        "3 (Acceptable): Some meaningful connections between ideas\n"
        "5 (Excellent): Excellent synthesis with original cross-cutting insights"
    ),
    "accuracy": (
        "1 (Poor): Major factual errors or hallucinations\n"
        "3 (Acceptable): Mostly accurate with minor issues\n"
        "5 (Excellent): Fully accurate, no hallucinations, properly caveated"
    ),
    "recommendation_quality": (
        "1 (Poor): No clear recommendation or generic platitude\n"
        "3 (Acceptable): Clear recommendation with basic reasoning\n"
        "5 (Excellent): Specific, well-reasoned, addresses constraints, actionable"
    ),
    "specificity": (
        "1 (Poor): Entirely vague ('set up backend')\n"
        "3 (Acceptable): Mix of specific and general steps\n"
        "5 (Excellent): All steps are concrete with clear deliverables"
    ),
    "risk_awareness": (
        "1 (Poor): No risks mentioned\n"
        "3 (Acceptable): Identifies key risks with basic mitigation\n"
        "5 (Excellent): Thorough risks, mitigations, fallbacks, rollback plan"
    ),
    "completeness": (
        "1 (Poor): Major gaps in coverage\n"
        "3 (Acceptable): Covers main phases adequately\n"
        "5 (Excellent): Complete with all phases, dependencies, edge cases"
    ),
    "dependency_correctness": (
        "1 (Poor): Dependencies wrong or ignored\n"
        "3 (Acceptable): Mostly correct ordering\n"
        "5 (Excellent): Perfect ordering with explicit dependency justification"
    ),
    "feasibility": (
        "1 (Poor): Unrealistic or ignores constraints\n"
        "3 (Acceptable): Feasible with some optimistic estimates\n"
        "5 (Excellent): Pragmatic, right-sized, acknowledges trade-offs"
    ),
    "code_quality": (
        "1 (Poor): No types, bad naming, no structure\n"
        "3 (Acceptable): Types present, reasonable structure\n"
        "5 (Excellent): Production-quality, documented, testable, idiomatic"
    ),
    "error_handling": (
        "1 (Poor): No error handling\n"
        "3 (Acceptable): Handles main error cases\n"
        "5 (Excellent): Specific error types, logging, graceful degradation"
    ),
    "architecture": (
        "1 (Poor): Monolithic, no separation\n"
        "3 (Acceptable): Reasonable structure\n"
        "5 (Excellent): Composable, extensible, right level of abstraction"
    ),
    "reasoning_quality": (
        "1 (Poor): No reasoning, just restates data\n"
        "3 (Acceptable): Adequate reasoning with some logic\n"
        "5 (Excellent): Explicit reasoning chain, considers alternatives, well-justified"
    ),
    "practical_output": (
        "1 (Poor): Generic or unhelpful output\n"
        "3 (Acceptable): Useful with some specifics\n"
        "5 (Excellent): Directly actionable with concrete next steps"
    ),
    # Additional rubrics for task-specific dimensions
    "technical_depth": (
        "1 (Poor): Surface-level, no understanding of trade-offs\n"
        "3 (Acceptable): Demonstrates basic understanding with some reasoning\n"
        "5 (Excellent): Deep architectural understanding with nuanced trade-offs"
    ),
    "practical_judgment": (
        "1 (Poor): Recommendations ignore real-world constraints\n"
        "3 (Acceptable): Addresses main constraints with reasonable suggestions\n"
        "5 (Excellent): Pragmatic, specific to user's context, acknowledges trade-offs"
    ),
    "structure": (
        "1 (Poor): Disorganized wall of text\n"
        "3 (Acceptable): Basic organization with sections\n"
        "5 (Excellent): Well-structured with clear flow and easy to follow"
    ),
    "actionability": (
        "1 (Poor): Generic advice, no specific next steps\n"
        "3 (Acceptable): Some actionable suggestions\n"
        "5 (Excellent): Specific, implementable recommendations with clear priorities"
    ),
    "phased_approach": (
        "1 (Poor): No phases, monolithic plan\n"
        "3 (Acceptable): Basic phasing with some logic\n"
        "5 (Excellent): Clear phases with dependencies, parallel-run, and rollback"
    ),
    "risk_mitigation": (
        "1 (Poor): No risks identified\n"
        "3 (Acceptable): Major risks identified with basic mitigation\n"
        "5 (Excellent): Comprehensive risks with specific mitigations and fallbacks"
    ),
    "architecture_quality": (
        "1 (Poor): No clear architecture, muddled responsibilities\n"
        "3 (Acceptable): Reasonable separation of concerns\n"
        "5 (Excellent): Clean layers, clear data flow, appropriate technology choices"
    ),
    "scalability_awareness": (
        "1 (Poor): No consideration of growth\n"
        "3 (Acceptable): Mentions scalability concerns\n"
        "5 (Excellent): Pragmatic growth path with clear upgrade triggers"
    ),
    "edge_case_handling": (
        "1 (Poor): No edge cases considered\n"
        "3 (Acceptable): Handles common edge cases\n"
        "5 (Excellent): Comprehensive edge case handling with graceful degradation"
    ),
    "cache_design": (
        "1 (Poor): No caching or broken cache logic\n"
        "3 (Acceptable): Basic TTL caching works\n"
        "5 (Excellent): Correct TTL, thread-safe, no memory leaks"
    ),
    "answer_quality": (
        "1 (Poor): Wrong or unusable answer\n"
        "3 (Acceptable): Correct with basic explanation\n"
        "5 (Excellent): Clear, specific, with practical advice"
    ),
    "context_assembly": (
        "1 (Poor): Hallucinated data, didn't use tool results\n"
        "3 (Acceptable): Used most tool results correctly\n"
        "5 (Excellent): Systematic data gathering then reasoning, no hallucinations"
    ),
    "analysis_structure": (
        "1 (Poor): Unstructured, missing key sections\n"
        "3 (Acceptable): Basic structure with main sections\n"
        "5 (Excellent): Systematic process, each section references specific data"
    ),
    "judgment_quality": (
        "1 (Poor): No clear recommendation or ignores conflicting signals\n"
        "3 (Acceptable): Clear recommendation with basic reasoning\n"
        "5 (Excellent): Addresses conflicting signals, states confidence, specific levels"
    ),
    "diagnostic_approach": (
        "1 (Poor): Blind retry or random actions\n"
        "3 (Acceptable): Some investigation before fixing\n"
        "5 (Excellent): Systematic diagnosis: check health, read logs, identify root cause"
    ),
    "communication": (
        "1 (Poor): No explanation of what happened\n"
        "3 (Acceptable): Basic explanation of the fix\n"
        "5 (Excellent): Clear explanation of root cause, fix applied, and prevention"
    ),
}

# Synthetic model responses for validation — one per task, designed to be
# "acceptable" quality (should score ~3 on most dimensions)
SYNTHETIC_RESPONSES: dict[str, str] = {
    "research_mcp_easy_01": (
        "Based on my research, Docker's mcp-gateway is a Go-based proxy that runs "
        "MCP servers in isolated containers and supports Streamable HTTP transport. "
        "Microsoft's mcp-gateway is .NET-based and designed to be Kubernetes-native "
        "with session affinity and a control plane for lifecycle management.\n\n"
        "For transport, Docker supports Streamable HTTP while Microsoft focuses on "
        "K8s-native communication. Auth-wise, Docker uses profile-based grouping "
        "while Microsoft provides enterprise access control.\n\n"
        "For a small team on GKE, I'd recommend Microsoft's gateway since it's "
        "designed for Kubernetes from the start, though Docker's is simpler to set up."
    ),
    "planning_wine_easy_01": (
        "Here's a 2-week POC plan:\n\n"
        "Week 1:\n"
        "- Day 1-2: Set up Supabase project, design database schema (wines, bottles, "
        "racks, positions, events tables), configure auth and RLS policies\n"
        "- Day 3-4: Build scan-label edge function with Claude Vision API\n"
        "- Day 5: Basic React Native app shell with auth flow\n\n"
        "Week 2:\n"
        "- Day 1-2: Inventory list view and bottle detail screen\n"
        "- Day 3-4: Rack grid visualization with drag-to-place\n"
        "- Day 5: Consume bottle flow and basic stats\n\n"
        "Critical path: Schema → Auth → Scan Label → Inventory UI\n"
        "Risk: Claude Vision accuracy on wine labels varies by condition."
    ),
    "agentic_brain_easy_01": (
        "I've saved your insight about applying the autoresearch pattern to the MCP "
        "gateway with tool-call routing latency as the eval metric.\n\n"
        "Looking at your related thoughts, I found two relevant connections:\n"
        "1. You previously noted the need for consistent LLM benchmarking across models\n"
        "2. You captured Karpathy's insight that the fixed eval makes the agent loop work\n\n"
        "This connects nicely — your MCP gateway could use the autoresearch pattern "
        "where routing latency is the fixed eval metric, and the agent loop iterates "
        "on routing strategies."
    ),
    "agentic_wine_medium_01": (
        "For grilled salmon tonight, I'd recommend the 2022 Cloudy Bay Sauvignon Blanc "
        "from rack A, position A3.\n\n"
        "Here's my reasoning:\n"
        "- Sauvignon Blanc pairs excellently with salmon — crisp acidity cuts through "
        "the fish's richness\n"
        "- It's in its drinking window (peak 2023-2025), so perfect timing\n"
        "- Your taste profile shows you prefer crisp whites and dislike heavily oaked "
        "chardonnay, which rules out the Cakebread\n"
        "- The Riesling is approaching peak but not there yet (2025-2032)\n\n"
        "Pull from rack A, position A3. Serve chilled to 45°F."
    ),
    "agentic_finance_hard_01": (
        "Here's my analysis of NVDA for a swing trade:\n\n"
        "**Technical Setup:** Price at $142.50, pulled back 9% from highs. RSI at 38.5 "
        "(approaching oversold). Near lower Bollinger band ($135). MACD bearish but "
        "narrowing. Still above SMA 200 ($128.50) — long-term uptrend intact.\n\n"
        "**Fundamentals:** Data center revenue grew 65% YoY. Guided Q2 at $44B, above "
        "consensus. Strong Blackwell Ultra announcement.\n\n"
        "**Risks:** Export controls tightened — could impact 12-15% of data center revenue. "
        "VIX slightly elevated at 18.5.\n\n"
        "**Recommendation:** Cautious LONG at $142.50. The technical pullback to oversold "
        "levels with intact fundamentals is a reasonable entry. Stop at $135 (below lower "
        "Bollinger), target $156 (prior high). Position: 345 shares ($49,162), risking "
        "$2,000 (2% of portfolio). R:R = 2.3:1.\n\n"
        "Confidence: Medium. The export control risk is real but the fundamental thesis "
        "remains strong. Would increase conviction if RSI dips below 35."
    ),
}


def get_judge_dimensions(task: TaskDefinition) -> list[tuple[str, str, float]]:
    """Extract (dimension_name, rubric, weight) for judge-rubric scored dimensions."""
    dims = []
    for name, cfg in task.scoring.items():
        if cfg.get("method") == "judge_rubric":
            rubric = JUDGE_RUBRICS.get(name, f"Rate {name} from 1 (poor) to 5 (excellent)")
            weight = cfg.get("weight", 0.0)
            dims.append((name, rubric, weight))
    return dims


def run_validation(
    tasks: list[TaskDefinition],
    judge: RoutedJudgeClient,
) -> list[dict]:
    """Run judge on all tasks with synthetic responses. Returns results list."""
    results = []

    for task in tasks:
        synthetic = SYNTHETIC_RESPONSES.get(task.id)
        if not synthetic:
            console.print(f"  [dim]SKIP {task.id} (no synthetic response)[/dim]")
            continue

        judge_dims = get_judge_dimensions(task)
        if not judge_dims:
            console.print(f"  [dim]SKIP {task.id} (no judge dimensions)[/dim]")
            continue

        console.print(f"\n  [bold]{task.id}[/bold] ({task.dimension}/{task.difficulty})")

        for dim_name, rubric, _weight in judge_dims:
            routed_to = "flow-judge" if dim_name in FLOW_JUDGE_DIMENSIONS else "reward-anything"
            result = judge.evaluate(
                dimension=dim_name,
                rubric=rubric,
                task_description=task.user_message,
                model_response=synthetic,
            )

            if isinstance(result, JudgeResult):
                # Snap to ternary
                raw = result.score
                snapped = 1 if raw <= 2.0 else (5 if raw > 4.0 else 3)
                valid = snapped in (1, 3, 5)
                status = f"[green]{snapped}[/green]" if valid else f"[red]{raw:.1f} INVALID[/red]"
                console.print(
                    f"    {dim_name:25s} → {routed_to:20s} score={status}  "
                    f"(raw={raw:.1f})  reason: {result.reasoning[:80]}..."
                )
                results.append(
                    {
                        "task_id": task.id,
                        "dimension": dim_name,
                        "routed_to": routed_to,
                        "raw_score": raw,
                        "snapped_score": snapped,
                        "reasoning": result.reasoning,
                        "status": "ok",
                    }
                )
            else:
                console.print(
                    f"    {dim_name:25s} → {routed_to:20s} [red]PARSE FAIL[/red]  "
                    f"err={result.error}  raw={result.raw_response[:100]}"
                )
                results.append(
                    {
                        "task_id": task.id,
                        "dimension": dim_name,
                        "routed_to": routed_to,
                        "raw_score": None,
                        "snapped_score": None,
                        "reasoning": None,
                        "status": f"error: {result.error}",
                    }
                )

    return results


def print_summary(results: list[dict]) -> None:
    """Print summary table."""
    console.print("\n[bold]═══ Validation Summary ═══[/bold]\n")

    total = len(results)
    ok = sum(1 for r in results if r["status"] == "ok")
    failed = total - ok

    # Routing breakdown
    flow_count = sum(1 for r in results if r["routed_to"] == "flow-judge")
    reward_count = sum(1 for r in results if r["routed_to"] == "reward-anything")

    table = Table(show_header=True, border_style="cyan")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    table.add_row("Total judge calls", str(total))
    table.add_row("Successful parses", f"[green]{ok}[/green]")
    table.add_row("Parse failures", f"[red]{failed}[/red]" if failed else "[green]0[/green]")
    table.add_row("Routed to flow-judge", str(flow_count))
    table.add_row("Routed to reward-anything", str(reward_count))

    if ok:
        scores = [r["snapped_score"] for r in results if r["status"] == "ok"]
        table.add_row("Score distribution (1/3/5)", f"{scores.count(1)}/{scores.count(3)}/{scores.count(5)}")
        avg = sum(s for s in scores) / len(scores)
        table.add_row("Mean snapped score", f"{avg:.1f}")

    console.print(table)

    # Per-judge breakdown
    for judge_name in ["flow-judge", "reward-anything"]:
        judge_results = [r for r in results if r["routed_to"] == judge_name and r["status"] == "ok"]
        if judge_results:
            dims = [r["dimension"] for r in judge_results]
            scores = [r["snapped_score"] for r in judge_results]
            console.print(f"\n  [cyan]{judge_name}[/cyan]: {len(judge_results)} calls")
            console.print(f"    Dimensions: {', '.join(sorted(set(dims)))}")
            console.print(f"    Scores: {scores}")

    # Overall verdict
    parse_rate = ok / total if total else 0
    if parse_rate >= 0.9 and flow_count > 0 and reward_count > 0:
        console.print("\n  [bold green]✓ PASS[/bold green] — routing works, parsing reliable")
    elif parse_rate >= 0.7:
        console.print("\n  [bold yellow]⚠ PARTIAL[/bold yellow] — some parse failures, investigate")
    else:
        console.print("\n  [bold red]✗ FAIL[/bold red] — pipeline broken")


def main():
    logging.basicConfig(level=logging.WARNING, format="%(message)s")

    parser = argparse.ArgumentParser(description="Validate judge pipeline with eval tasks")
    parser.add_argument("--dimension", help="Filter to one dimension")
    parser.add_argument("--difficulty", help="Filter by difficulty")
    parser.add_argument("--tasks", type=int, help="Limit number of tasks")
    parser.add_argument("--port", type=int, default=JUDGE_PORT)
    args = parser.parse_args()

    # Check judge server
    if not check_health(f"http://127.0.0.1:{args.port}"):
        console.print(f"[red]No judge server on :{args.port}[/red]")
        console.print(
            "Start with: CUDA_VISIBLE_DEVICES=1 $LLAMA_SWAP_BIN "
            "--config config/judge_swap_config.yaml --listen :9091"
        )
        sys.exit(1)

    # Load tasks
    tasks = load_tasks(
        dimension=args.dimension,
        difficulty=args.difficulty,
    )
    # Filter to tasks with synthetic responses
    tasks = [t for t in tasks if t.id in SYNTHETIC_RESPONSES]
    if args.tasks:
        tasks = tasks[: args.tasks]

    if not tasks:
        console.print("[red]No tasks with synthetic responses found[/red]")
        sys.exit(1)

    console.print(f"[bold]Validating judge pipeline with {len(tasks)} tasks[/bold]")
    console.print(f"  Judge: RoutedJudgeClient on :{args.port}")
    console.print(f"  Flow-Judge dims: {', '.join(sorted(FLOW_JUDGE_DIMENSIONS))}")

    with RoutedJudgeClient(
        base_url=f"http://127.0.0.1:{args.port}/v1",
        timeout=120.0,
    ) as judge:
        results = run_validation(tasks, judge)

    print_summary(results)


if __name__ == "__main__":
    main()
