"""Main evaluation orchestrator.

Runs the full pipeline: load tasks → for each model → run conversations →
score (deterministic + judge) → persist to SQLite → generate summary.
Supports checkpoint/resume — restarting picks up from the last incomplete task.

Usage:
    uv run python -m nite_eval.orchestrator
    uv run python -m nite_eval.orchestrator --models qwen3.8-27b
    uv run python -m nite_eval.orchestrator --dimension agentic --resume run-20260405
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

import yaml
from rich.console import Console
from rich.table import Table

from nite_eval.automated_scoring import run_automated_checks
from nite_eval.conversation_runner import ConversationResult, run_conversation
from nite_eval.gpu_check import GpuPlacementError, resolve_expected_uuids, verify_runtime_placement
from nite_eval.gpu_check import preflight as gpu_preflight
from nite_eval.judge import FLOW_JUDGE_DIMENSIONS, RoutedJudgeClient
from nite_eval.mock_tools import MockToolEnv, summarise_unmatched
from nite_eval.model_manager import check_health, warm_up_model
from nite_eval.report import save_report  # noqa: TC001
from nite_eval.results_db import ResultsDB
from nite_eval.rubrics import get_rubric
from nite_eval.sandbox import SandboxError, SandboxSpec, SandboxToolEnv, docker_available, reap_orphans
from nite_eval.scoring import (
    ScoreResult,
    aggregate_task_scores,
    compute_composite,
    score_checklist_with_judge,
    score_contains_check,
    score_distractor_avoidance,
    score_sequence_match,
    score_subset_match,
    score_tool_args_match,
    score_tool_ordering,
    score_with_judge,
)
from nite_eval.task_loader import TaskDefinition, load_tasks

logger = logging.getLogger(__name__)
console = Console()

DEFAULT_CONFIG = "config/eval_config.yaml"

# Criteria that ask whether the response's facts match what the tools returned.
# These get the tool results appended to the judge prompt; every other criterion
# is about the response itself and does not need them.
EVIDENCE_DIMENSIONS = frozenset({"no_hallucination", "data_accuracy", "data_threading"})


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def generate_run_id() -> str:
    return datetime.now(UTC).strftime("run-%Y%m%d-%H%M%S")


def score_task(
    task: TaskDefinition,
    conv: ConversationResult,
    judge: RoutedJudgeClient,
    judge_averaging: bool = True,
    automated_results: dict[str, tuple[float, dict]] | None = None,
) -> tuple[list[ScoreResult], float, float]:
    """Score a completed conversation against a task's scoring config.

    Returns (per-criterion ScoreResults, weighted total, excluded weight
    fraction). The third value is what makes the second interpretable: a task
    scored over 35% of its declared criteria is a narrower claim than the same
    number over all of them.
    """
    scores: list[ScoreResult] = []

    # Collect tool calls for deterministic scoring
    actual_calls = []
    for turn in conv.turns:
        for tr in turn.tool_responses:
            actual_calls.append({"name": tr["name"], "arguments": tr.get("arguments", {})})

    # Ground truth for fact-checking criteria. The judge otherwise sees only the
    # prompt and the final answer, so "do the cited numbers match the data" was
    # unanswerable and the criterion fell through to a free 1.0.
    evidence_lines = []
    for turn in conv.turns:
        for tr in turn.tool_responses:
            args = json.dumps(tr.get("arguments", {}))
            result = json.dumps(tr.get("result", {}))
            evidence_lines.append(f"{tr['name']}({args}) -> {result}")
    evidence = "\n".join(evidence_lines)

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
                use_averaging=judge_averaging,
                evidence=evidence if dim_name in EVIDENCE_DIMENSIONS else "",
            )
            # score_with_judge sets weight=0.0 on JudgeError as an escape hatch.
            # Unconditionally assigning the task weight here overwrote it, so a
            # judge that failed to respond scored 0 at full weight — nine
            # historical criteria carry a 0.0 with no reasoning for this reason.
            # A failed judge is a missing measurement, not a bad answer.
            if sr.details.get("error"):
                logger.warning(
                    "Judge failed for %s/%s — excluded from weighting: %s",
                    task.id,
                    dim_name,
                    sr.details["error"],
                )
                sr.details["unscored"] = True
                sr.details["declared_weight"] = weight
            else:
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
            raw, details = score_checklist_with_judge(
                judge=judge,
                criteria=criteria_list,
                task_description=task.user_message,
                response_text=conv.final_response,
            )
            sr = ScoreResult(
                dimension=dim_name,
                method=method,
                score=raw,
                weight=weight,
                details=details,
            )
            # A failed checklist judge is a missing measurement. Falling back to
            # substring matching would quietly restore the scoring this replaced.
            if details.get("error"):
                logger.warning("Checklist judge failed for %s/%s: %s", task.id, dim_name, details["error"])
                sr.weight = 0.0
                sr.details["unscored"] = True
                sr.details["declared_weight"] = weight
            scores.append(sr)

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

        elif method == "tool_args_match":
            raw = score_tool_args_match(actual_calls, task.expected_tool_sequence)
            scores.append(
                ScoreResult(
                    dimension=dim_name,
                    method=method,
                    score=raw,
                    weight=weight,
                    details={"criteria": criteria},
                )
            )

        elif method == "tool_absence":
            forbidden = criteria if isinstance(criteria, list) else task.distractor_tools
            raw = score_distractor_avoidance(actual_calls, forbidden)
            scores.append(
                ScoreResult(
                    dimension=dim_name,
                    method=method,
                    score=raw,
                    weight=weight,
                    details={"forbidden": forbidden},
                )
            )

        elif method == "tool_ordering":
            raw = score_tool_ordering(actual_calls, task.expected_tool_ordering)
            scores.append(
                ScoreResult(
                    dimension=dim_name,
                    method=method,
                    score=raw,
                    weight=weight,
                    details={"ordering": task.expected_tool_ordering},
                )
            )

        elif method == "automated" and (automated_results or {}).get(dim_name):
            raw, details = (automated_results or {})[dim_name]
            scores.append(
                ScoreResult(
                    dimension=dim_name,
                    method=method,
                    score=raw,
                    weight=weight,
                    details=details,
                )
            )

        elif method in ("deterministic", "partial_match", "exact_match", "automated"):
            # No implementation exists for these. They used to be faked:
            # `deterministic` returned 1.0 whenever the conversation did not
            # error (free marks for not crashing), or re-ran the same
            # subset_match as the task's own tool_coverage dimension — the
            # identical measurement counted two or three times. `automated`
            # returned a hardcoded 0.0 at 40-70% of each coding task's weight.
            #
            # An unmeasurable criterion is excluded from the weighted average
            # rather than scored, so a task's number reflects only what was
            # actually measured. weight=0 removes it from aggregation.
            logger.warning(
                "Unscored criterion %s/%s (method=%s) — excluded from weighting",
                task.id,
                dim_name,
                method,
            )
            scores.append(
                ScoreResult(
                    dimension=dim_name,
                    method=method,
                    score=0.0,
                    weight=0.0,
                    details={
                        "unscored": True,
                        "declared_weight": weight,
                        "note": f"no implementation for method '{method}'",
                    },
                )
            )

        else:
            logger.warning("Unknown scoring method %s for %s/%s", method, task.id, dim_name)

    weighted = aggregate_task_scores(scores)

    declared = sum(cfg.get("weight", 0.0) for cfg in task.scoring.values())
    excluded = sum(sr.details.get("declared_weight", 0.0) for sr in scores if sr.details.get("unscored"))
    unscored_fraction = (excluded / declared) if declared else 0.0

    return scores, weighted, unscored_fraction


def run_task(
    task: TaskDefinition,
    model_name: str,
    target_url: str,
    judge: RoutedJudgeClient,
    db: ResultsDB,
    run_id: str,
    eval_cfg: dict,
    system_suffix: str = "",
    chat_template_kwargs: dict | None = None,
    native_tools: bool = False,
) -> float:
    """Run a single task for a model and persist results. Returns weighted score."""
    db.mark_task_running(run_id, model_name, task.id)
    console.print(f"    [bold]{task.id}[/bold] ({task.difficulty})", end="")

    # A task that declares an `environment:` runs against a real container; the
    # rest keep their mocks. SandboxToolEnv exposes the same call() interface,
    # so run_conversation does not know the difference.
    sandbox: SandboxToolEnv | None = None
    spec = SandboxSpec.from_task_yaml(task.environment)
    if spec is not None:
        try:
            sandbox = SandboxToolEnv(spec)
            sandbox.start()
        except SandboxError as e:
            console.print(f" [red]sandbox unavailable: {e}[/red]")
            db.save_task_result(
                run_id=run_id,
                model_name=model_name,
                task_id=task.id,
                final_response="",
                total_turns=0,
                total_tool_calls=0,
                total_latency_ms=0,
                reached_max_turns=False,
                weighted_score=0.0,
                error=f"sandbox_unavailable: {e}",
            )
            return 0.0

    tool_env = sandbox if sandbox is not None else MockToolEnv.from_task_yaml(task.mock_responses)

    # Run conversation
    conv = run_conversation(
        base_url=target_url,
        model_name=model_name,
        system_prompt=task.system_prompt,
        tools=task.tools,
        user_message=task.user_message,
        mock_env=tool_env,
        max_turns=task.max_turns,
        max_tool_calls=task.max_tool_calls,
        timeout_seconds=task.timeout_seconds,
        temperature=eval_cfg.get("temperature", 0.0),
        max_tokens=task.max_tokens or eval_cfg.get("max_tokens", 2048),
        system_suffix=system_suffix,
        chat_template_kwargs=chat_template_kwargs,
        native_tools=native_tools,
    )

    if conv.error:
        if sandbox is not None:
            sandbox.stop()
        console.print(f" [red]ERROR: {conv.error}[/red]")
        # Persist the last raw response. A failed task with an empty
        # final_response is undiagnosable after the fact — the offending text
        # is exactly what you need to tell a model bug from a harness bug.
        last_response = conv.turns[-1].response if conv.turns else ""
        db.save_task_result(
            run_id=run_id,
            model_name=model_name,
            task_id=task.id,
            final_response=last_response,
            total_turns=len(conv.turns),
            total_tool_calls=conv.total_tool_calls,
            total_latency_ms=conv.total_latency_ms,
            reached_max_turns=conv.reached_max_turns,
            weighted_score=0.0,
            error=conv.error,
            repaired_tool_calls=conv.repaired_tool_calls,
            unmatched_mock_calls=len(getattr(tool_env, "unmatched_calls", [])),
            unmatched_mock_samples=summarise_unmatched(getattr(tool_env, "unmatched_calls", [])),
            # 0 across a whole task means the server never reported a usage block,
            # not that the model generated nothing, so store NULL and let the
            # report say "unavailable" rather than print a fabricated 0 tok/s.
            completion_tokens=conv.total_completion_tokens or None,
            prompt_tokens=conv.total_prompt_tokens or None,
            predicted_ms=conv.total_predicted_ms or None,
            predicted_n=conv.total_predicted_n or None,
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
    # Automated criteria are decided by running the code, after the conversation
    # ends so the hidden suite is never visible to the model.
    automated_results: dict[str, tuple[float, dict]] = {}
    if sandbox is not None:
        try:
            automated_results = run_automated_checks(
                sandbox=sandbox,
                task_id=task.id,
                scoring=task.scoring,
                suite_root=Path(eval_cfg.get("suite_root", "tasks/coding/suites")),
                language=task.environment.get("language", ""),
                checks=task.environment.get("checks", {}),
            )
        except SandboxError as e:
            logger.warning("Automated checks failed for %s: %s", task.id, e)
        finally:
            sandbox.stop()

    scores, weighted, unscored_weight = score_task(
        task,
        conv,
        judge,
        judge_averaging=eval_cfg.get("judge_averaging", True),
        automated_results=automated_results,
    )

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
        repaired_tool_calls=conv.repaired_tool_calls,
        unmatched_mock_calls=len(getattr(tool_env, "unmatched_calls", [])),
        unmatched_mock_samples=summarise_unmatched(getattr(tool_env, "unmatched_calls", [])),
        unscored_weight=unscored_weight,
        # 0 across a whole task means the server never reported a usage block,
        # not that the model generated nothing, so store NULL and let the
        # report say "unavailable" rather than print a fabricated 0 tok/s.
        completion_tokens=conv.total_completion_tokens or None,
        prompt_tokens=conv.total_prompt_tokens or None,
        predicted_ms=conv.total_predicted_ms or None,
        predicted_n=conv.total_predicted_n or None,
    )

    turns_str = f"{len(conv.turns)}t/{conv.total_tool_calls}tc"
    repaired = f", {conv.repaired_tool_calls} repaired" if conv.repaired_tool_calls else ""
    unscored = f", {unscored_weight:.0%} unscored" if unscored_weight else ""
    console.print(f" → {weighted:.2f} ({turns_str}, {conv.total_latency_ms:.0f}ms{repaired}{unscored})")
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
    parser.add_argument(
        "--skip-gpu-check",
        action="store_true",
        help="Skip GPU placement verification (target/judge pinned to the right GPUs)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    target_url = cfg["target"]["base_url"]
    judge_cfg = cfg["judge"]
    eval_cfg = cfg.get("evaluation", {})
    weights = cfg.get("scoring", {}).get("dimension_weights")
    results_dir = Path(cfg.get("results", {}).get("dir", "results/runs"))
    db_name = cfg.get("results", {}).get("db_name", "eval_results.db")

    models_cfg = cfg.get("models", [])
    models = args.models or [m["name"] for m in models_cfg]
    if not models:
        console.print("[red]No models configured[/red]")
        sys.exit(1)

    # Per-model system-prompt suffix (e.g. "/no_think" for Qwen3 models).
    # Applied to every task for that model via chat-template trigger.
    system_suffix_by_model: dict[str, str] = {m["name"]: m.get("system_suffix", "") for m in models_cfg}

    # Per-model chat-template kwargs, forwarded to llama-server and into the
    # Jinja template. Qwen3.8's template has no `/no_think` branch — its only
    # thinking switch is `{"enable_thinking": false}`.
    template_kwargs_by_model: dict[str, dict] = {m["name"]: m.get("chat_template_kwargs") or {} for m in models_cfg}
    # Opt-in per model: send tool schemas in the request and read the server's
    # structured tool_calls, instead of pasting definitions into the prompt and
    # parsing the reply out of text. Ornith's documented usage is the native
    # path and it returns clean calls there; asked to hand-write JSON in prose
    # it emitted four distinct malformation classes across three runs.
    native_tools_by_model: dict[str, bool] = {m["name"]: bool(m.get("native_tools")) for m in models_cfg}

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

    # Verify GPU placement. A judge sharing a GPU with the model under
    # evaluation contends for VRAM and pushes the target's layers to host
    # memory — the run still completes and still writes scores, but the
    # latency numbers are meaningless. Fail loudly instead.
    if not args.skip_gpu_check:
        console.print("Checking GPU placement...")
        swap_cfg = Path(cfg.get("target", {}).get("llama_swap_config", "config/llama_swap_config.yaml"))
        try:
            _, gpu_warnings = gpu_preflight(swap_cfg, strict=True)
        except GpuPlacementError as e:
            console.print(f"[red]{e}[/red]")
            sys.exit(1)
        for w in gpu_warnings:
            console.print(f"  [yellow]warning: {w}[/yellow]")
        console.print("[green]GPU placement OK[/green]")

    # Clear sandboxes left by an interrupted previous run. A process killed
    # mid-task never runs stop(), and the container then idles holding memory.
    if docker_available():
        orphans = reap_orphans()
        if orphans:
            console.print(f"[yellow]Reaped {len(orphans)} orphaned sandbox container(s)[/yellow]")

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
        flow_judge_url=judge_cfg.get("flow_judge_url"),
        reward_anything_url=judge_cfg.get("reward_anything_url"),
        temperature=judge_cfg.get("temperature", 0.1),
        max_tokens=judge_cfg.get("max_tokens", 2048),
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

                # Re-verify placement now that the model is actually resident.
                # llama-swap loads on first request, so the startup check ran
                # while the target GPU was empty and could only validate config,
                # never where this model really landed.
                if not args.skip_gpu_check:
                    try:
                        target_uuid, judge_uuid = resolve_expected_uuids()
                        gpu_errors, _ = verify_runtime_placement(target_uuid, judge_uuid)
                    except GpuPlacementError as e:
                        console.print(f"  [red]{e}[/red]")
                        sys.exit(1)
                    if gpu_errors:
                        console.print(f"  [red]GPU placement wrong after loading {model}:[/red]")
                        for err in gpu_errors:
                            console.print(f"    [red]- {err}[/red]")
                        sys.exit(1)

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
                    run_task(
                        task,
                        model,
                        target_url,
                        judge,
                        db,
                        run_id,
                        eval_cfg,
                        system_suffix=system_suffix_by_model.get(model, ""),
                        chat_template_kwargs=template_kwargs_by_model.get(model) or None,
                        native_tools=native_tools_by_model.get(model, False),
                    )
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

        # Generate Markdown report
        report_path = save_report(
            db,
            run_id,
            results_dir,
            weights,
            mdd=float(cfg.get("scoring", {}).get("min_detectable_difference", 0.05)),
            dimension_mdd={
                str(k): float(v)
                for k, v in (cfg.get("scoring", {}).get("dimension_min_detectable_difference") or {}).items()
            },
            judge_samples=3 if eval_cfg.get("judge_averaging", True) else 1,
        )
        console.print(f"\nReport saved to [bold]{report_path}[/bold]")

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted — progress saved, resume with --resume {run_id}[/yellow]")
        db.finish_run(run_id, status="interrupted")
        sys.exit(130)
    finally:
        judge.close()
        db.close()
        # A run that ends mid-task — Ctrl-C, an unhandled error, a killed
        # process — leaves its sandbox container running. Reaping only at
        # startup meant one could idle for hours holding memory.
        if docker_available():
            with contextlib.suppress(Exception):
                leftover = reap_orphans()
                if leftover:
                    console.print(f"[yellow]Cleaned up {len(leftover)} sandbox container(s)[/yellow]")


if __name__ == "__main__":
    main()
