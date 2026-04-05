"""Main evaluation orchestrator.

Runs the full pipeline: load tasks → for each model → run conversations →
score (deterministic + judge) → persist to SQLite → generate summary.
Supports checkpoint/resume — restarting picks up from the last incomplete task.

Usage:
    uv run python -m nite_eval.orchestrator
    uv run python -m nite_eval.orchestrator --models qwen3.5-9b
    uv run python -m nite_eval.orchestrator --dimension agentic --resume run-20260405
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

import yaml
from rich.console import Console
from rich.table import Table

from nite_eval.conversation_runner import ConversationResult, run_conversation
from nite_eval.judge import FLOW_JUDGE_DIMENSIONS, RoutedJudgeClient
from nite_eval.mock_tools import MockToolEnv
from nite_eval.model_manager import check_health, warm_up_model
from nite_eval.results_db import ResultsDB
from nite_eval.rubrics import get_rubric
from nite_eval.scoring import (
    ScoreResult,
    aggregate_task_scores,
    compute_composite,
    score_checklist,
    score_contains_check,
    score_sequence_match,
    score_subset_match,
    score_with_judge,
)
from nite_eval.task_loader import TaskDefinition, load_tasks

logger = logging.getLogger(__name__)
console = Console()

DEFAULT_CONFIG = "config/eval_config.yaml"


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def generate_run_id() -> str:
    return datetime.now(UTC).strftime("run-%Y%m%d-%H%M%S")


def score_task(
    task: TaskDefinition,
    conv: ConversationResult,
    judge: RoutedJudgeClient,
) -> tuple[list[ScoreResult], float]:
    """Score a completed conversation against a task's scoring config.

    Returns (list of per-dimension ScoreResults, weighted total).
    """
    scores: list[ScoreResult] = []

    # Collect tool calls for deterministic scoring
    actual_calls = []
    for turn in conv.turns:
        for tr in turn.tool_responses:
            actual_calls.append({"name": tr["name"], "arguments": tr.get("arguments", {})})

    for dim_name, dim_cfg in task.scoring.items():
        method = dim_cfg.get("method", "")
        weight = dim_cfg.get("weight", 0.0)
        criteria = dim_cfg.get("criteria", "")

        if method == "judge_rubric":
            rubric = get_rubric(dim_name)
            sr = score_with_judge(
                judge=judge,
                dimension=dim_name,
                rubric=rubric,
                task_description=task.user_message,
                model_response=conv.final_response,
                use_averaging=False,
            )
            sr.weight = weight
            scores.append(sr)

        elif method == "sequence_match":
            raw = score_sequence_match(actual_calls, task.expected_tool_sequence)
            scores.append(
                ScoreResult(
                    dimension=dim_name,
                    method=method,
                    score=raw,
                    weight=weight,
                    details={"criteria": criteria},
                )
            )

        elif method == "subset_match":
            raw = score_subset_match(actual_calls, task.expected_tools_called)
            scores.append(
                ScoreResult(
                    dimension=dim_name,
                    method=method,
                    score=raw,
                    weight=weight,
                    details={"criteria": criteria},
                )
            )

        elif method == "checklist":
            criteria_list = criteria if isinstance(criteria, list) else [criteria]
            raw = score_checklist(conv.final_response, criteria_list)
            scores.append(
                ScoreResult(
                    dimension=dim_name,
                    method=method,
                    score=raw,
                    weight=weight,
                    details={"criteria": criteria_list},
                )
            )

        elif method == "contains_check":
            criteria_list = criteria if isinstance(criteria, list) else [criteria]
            raw = score_contains_check(conv.final_response, criteria_list)
            scores.append(
                ScoreResult(
                    dimension=dim_name,
                    method=method,
                    score=raw,
                    weight=weight,
                    details={"criteria": criteria_list},
                )
            )

        elif method in ("deterministic", "partial_match", "exact_match"):
            # These need task-specific logic; score based on tool call presence for now
            if task.expected_tools_called:
                raw = score_subset_match(actual_calls, task.expected_tools_called)
            else:
                raw = 1.0 if not conv.error else 0.0
            scores.append(
                ScoreResult(
                    dimension=dim_name,
                    method=method,
                    score=raw,
                    weight=weight,
                    details={"criteria": criteria},
                )
            )

        elif method == "automated":
            # Test suite scoring — placeholder until coding eval is wired
            scores.append(
                ScoreResult(
                    dimension=dim_name,
                    method=method,
                    score=0.0,
                    weight=weight,
                    details={"note": "automated scoring not yet implemented"},
                )
            )

        else:
            logger.warning("Unknown scoring method %s for %s/%s", method, task.id, dim_name)

    weighted = aggregate_task_scores(scores)
    return scores, weighted


def run_task(
    task: TaskDefinition,
    model_name: str,
    target_url: str,
    judge: RoutedJudgeClient,
    db: ResultsDB,
    run_id: str,
    eval_cfg: dict,
) -> float:
    """Run a single task for a model and persist results. Returns weighted score."""
    db.mark_task_running(run_id, model_name, task.id)
    console.print(f"    [bold]{task.id}[/bold] ({task.difficulty})", end="")

    # Set up mock environment
    mock_env = MockToolEnv.from_task_yaml(task.mock_responses)

    # Run conversation
    conv = run_conversation(
        base_url=target_url,
        model_name=model_name,
        system_prompt=task.system_prompt,
        tools=task.tools,
        user_message=task.user_message,
        mock_env=mock_env,
        max_turns=task.max_turns,
        timeout_seconds=task.timeout_seconds,
        temperature=eval_cfg.get("temperature", 0.0),
        max_tokens=eval_cfg.get("max_tokens", 2048),
    )

    if conv.error:
        console.print(f" [red]ERROR: {conv.error}[/red]")
        db.save_task_result(
            run_id=run_id,
            model_name=model_name,
            task_id=task.id,
            final_response="",
            total_turns=len(conv.turns),
            total_tool_calls=conv.total_tool_calls,
            total_latency_ms=conv.total_latency_ms,
            reached_max_turns=conv.reached_max_turns,
            weighted_score=0.0,
            error=conv.error,
        )
        return 0.0

    # Record tool calls
    tool_records = []
    for turn in conv.turns:
        for i, tr in enumerate(turn.tool_responses):
            tool_records.append(
                {
                    "turn": turn.turn,
                    "call_index": i,
                    "tool_name": tr["name"],
                    "arguments": tr.get("arguments", {}),
                    "result": tr.get("result", {}),
                }
            )
    if tool_records:
        db.save_tool_calls(run_id, model_name, task.id, tool_records)

    # Score
    scores, weighted = score_task(task, conv, judge)

    # Persist scores
    for sr in scores:
        judge_model = None
        reasoning = None
        confidence = None
        if sr.method == "judge_rubric":
            judge_model = "flow-judge" if sr.dimension in FLOW_JUDGE_DIMENSIONS else "reward-anything"
            reasoning = sr.details.get("reasoning")
            confidence = sr.details.get("confidence")

        db.save_score(
            run_id=run_id,
            model_name=model_name,
            task_id=task.id,
            dimension=sr.dimension,
            method=sr.method,
            raw_score=sr.details.get("raw_score", sr.score),
            normalized=sr.score,
            weight=sr.weight,
            judge_model=judge_model,
            reasoning=reasoning,
            confidence=confidence,
            details=sr.details,
        )

    # Save task result (checkpoint)
    db.save_task_result(
        run_id=run_id,
        model_name=model_name,
        task_id=task.id,
        final_response=conv.final_response,
        total_turns=len(conv.turns),
        total_tool_calls=conv.total_tool_calls,
        total_latency_ms=conv.total_latency_ms,
        reached_max_turns=conv.reached_max_turns,
        weighted_score=weighted,
    )

    turns_str = f"{len(conv.turns)}t/{conv.total_tool_calls}tc"
    console.print(f" → {weighted:.2f} ({turns_str}, {conv.total_latency_ms:.0f}ms)")
    return weighted


def print_results(db: ResultsDB, run_id: str, models: list[str], weights: dict[str, float]) -> None:
    """Print a summary table of results."""
    console.print("\n[bold]═══ Evaluation Results ═══[/bold]\n")

    # Per-model table
    table = Table(show_header=True, border_style="cyan")
    table.add_column("Model", style="bold")
    table.add_column("Research", justify="right")
    table.add_column("Planning", justify="right")
    table.add_column("Coding", justify="right")
    table.add_column("Agentic", justify="right")
    table.add_column("Composite", justify="right", style="bold")
    table.add_column("Tasks", justify="right")

    for model in models:
        dim_avgs = db.get_dimension_averages(run_id, model)
        composite = compute_composite(dim_avgs, weights) if dim_avgs else 0.0
        summary = db.get_run_summary(run_id).get(model, {})

        table.add_row(
            model,
            f"{dim_avgs.get('research', 0):.2f}",
            f"{dim_avgs.get('planning', 0):.2f}",
            f"{dim_avgs.get('coding', 0):.2f}",
            f"{dim_avgs.get('agentic', 0):.2f}",
            f"{composite:.2f}",
            f"{summary.get('completed', 0)}/{summary.get('total', 0)}",
        )

    console.print(table)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # Suppress noisy loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    parser = argparse.ArgumentParser(description="Run nite-eval evaluation pipeline")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Eval config path")
    parser.add_argument("--models", nargs="+", help="Override models to evaluate")
    parser.add_argument("--dimension", help="Filter tasks to one dimension")
    parser.add_argument("--difficulty", help="Filter tasks by difficulty")
    parser.add_argument("--resume", help="Resume a previous run by ID")
    parser.add_argument("--skip-server-check", action="store_true", help="Skip server health checks")
    args = parser.parse_args()

    cfg = load_config(args.config)
    target_url = cfg["target"]["base_url"]
    judge_cfg = cfg["judge"]
    eval_cfg = cfg.get("evaluation", {})
    weights = cfg.get("scoring", {}).get("dimension_weights")
    results_dir = Path(cfg.get("results", {}).get("dir", "results/runs"))
    db_name = cfg.get("results", {}).get("db_name", "eval_results.db")

    models = args.models or [m["name"] for m in cfg.get("models", [])]
    if not models:
        console.print("[red]No models configured[/red]")
        sys.exit(1)

    # Check servers
    if not args.skip_server_check:
        console.print("Checking servers...")
        if not check_health(target_url):
            console.print(f"[red]Target server not responding at {target_url}[/red]")
            sys.exit(1)
        judge_base = judge_cfg["base_url"].replace("/v1", "")
        if not check_health(judge_base):
            console.print(f"[red]Judge server not responding at {judge_base}[/red]")
            sys.exit(1)
        console.print("[green]Servers OK[/green]")

    # Load tasks
    tasks = load_tasks(dimension=args.dimension, difficulty=args.difficulty)
    if not tasks:
        console.print("[red]No tasks found[/red]")
        sys.exit(1)

    console.print(f"Loaded {len(tasks)} tasks across {len({t.dimension for t in tasks})} dimensions")

    # Set up run
    run_id = args.resume or generate_run_id()
    results_dir.mkdir(parents=True, exist_ok=True)
    db = ResultsDB(results_dir / db_name)

    if not args.resume:
        db.create_run(run_id, models, cfg)
        db.register_tasks(
            run_id,
            models,
            [(t.id, t.dimension, t.difficulty) for t in tasks],
        )
        console.print(f"Created run [bold]{run_id}[/bold] — {len(models)} models × {len(tasks)} tasks")
    else:
        status = db.get_run_status(run_id)
        if not status:
            console.print(f"[red]Run {run_id} not found in database[/red]")
            sys.exit(1)
        console.print(f"Resuming run [bold]{run_id}[/bold] (status: {status})")

    # Initialize judge
    judge = RoutedJudgeClient(
        base_url=judge_cfg["base_url"],
        flow_judge_model=judge_cfg.get("flow_judge_model", "flow-judge"),
        reward_anything_model=judge_cfg.get("reward_anything_model", "reward-anything"),
        temperature=judge_cfg.get("temperature", 0.1),
        max_tokens=judge_cfg.get("max_tokens", 1024),
    )

    try:
        for model in models:
            console.print(f"\n[bold cyan]═══ {model} ═══[/bold cyan]")

            # Warm up model via llama-swap
            if eval_cfg.get("warm_up", True):
                console.print(f"  Warming up {model}...")
                if not warm_up_model(target_url, model, timeout=120.0):
                    console.print(f"  [red]Failed to warm up {model}, skipping[/red]")
                    continue

            # Get pending tasks for resume support
            pending = db.get_pending_tasks(run_id, model)
            pending_ids = {p.task_id for p in pending}

            if not pending_ids:
                console.print("  All tasks complete, skipping")
                continue

            console.print(f"  {len(pending_ids)} tasks to run")

            for task in tasks:
                if task.id not in pending_ids:
                    continue
                try:
                    run_task(task, model, target_url, judge, db, run_id, eval_cfg)
                except Exception:
                    logger.exception("Task %s failed for %s", task.id, model)
                    db.save_task_result(
                        run_id=run_id,
                        model_name=model,
                        task_id=task.id,
                        final_response="",
                        total_turns=0,
                        total_tool_calls=0,
                        total_latency_ms=0,
                        reached_max_turns=False,
                        weighted_score=0.0,
                        error="unhandled exception",
                    )

        db.finish_run(run_id)
        print_results(db, run_id, models, weights)

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted — progress saved, resume with --resume {run_id}[/yellow]")
        db.finish_run(run_id, status="interrupted")
        sys.exit(130)
    finally:
        judge.close()
        db.close()


if __name__ == "__main__":
    main()
