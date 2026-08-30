"""Generate Markdown comparison reports from evaluation results.

Reads from the SQLite results DB and produces a structured report with:
- Summary table (models × dimensions)
- Per-task breakdown with scores and latency
- Per-dimension analysis
- Score distribution and notable results
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path  # noqa: TC003

from nite_eval.results_db import ResultsDB  # noqa: TC001
from nite_eval.scoring import compute_composite

logger = logging.getLogger(__name__)


def generate_report(
    db: ResultsDB,
    run_id: str,
    weights: dict[str, float] | None = None,
) -> str:
    """Generate a Markdown report for a run."""
    lines: list[str] = []

    # Header
    run_info = _get_run_info(db, run_id)
    models = json.loads(run_info["models"]) if run_info else []
    started = datetime.fromtimestamp(run_info["started_at"], tz=UTC) if run_info else None

    lines.append(f"# Evaluation Report: {run_id}")
    lines.append("")
    if started:
        lines.append(f"**Date:** {started.strftime('%Y-%m-%d %H:%M UTC')}")
    lines.append(f"**Models:** {', '.join(models)}")
    summary = db.get_run_summary(run_id)
    total_tasks = summary[models[0]]["total"] if models and models[0] in summary else 0
    lines.append(f"**Tasks:** {total_tasks}")
    lines.append("")

    # Summary table
    lines.append("## Summary")
    lines.append("")
    dimensions = ["research", "planning", "coding", "agentic"]
    header = "| Model | " + " | ".join(d.capitalize() for d in dimensions) + " | Composite | Tasks |"
    sep = "|" + "|".join(["-------"] * (len(dimensions) + 3)) + "|"
    lines.append(header)
    lines.append(sep)

    model_composites: dict[str, float] = {}
    for model in models:
        dim_avgs = db.get_dimension_averages(run_id, model)
        composite = compute_composite(dim_avgs, weights) if dim_avgs else 0.0
        model_composites[model] = composite
        s = summary.get(model, {})
        cells = [f"{dim_avgs.get(d, 0):.2f}" for d in dimensions]
        lines.append(
            f"| {model} | "
            + " | ".join(cells)
            + f" | **{composite:.2f}** | {s.get('completed', 0)}/{s.get('total', 0)} |"
        )
    lines.append("")

    # Rank models
    ranked = sorted(model_composites.items(), key=lambda x: x[1], reverse=True)
    if len(ranked) > 1:
        lines.append("**Ranking:** " + " > ".join(f"{m} ({s:.2f})" for m, s in ranked))
        lines.append("")

    # Per-task breakdown
    lines.append("## Per-Task Results")
    lines.append("")

    for dim in dimensions:
        lines.append(f"### {dim.capitalize()}")
        lines.append("")
        header = "| Task | Diff | " + " | ".join(models) + " | Turns | TCs | Rep | Unscored |"
        sep = "|" + "|".join(["------"] * (len(models) + 6)) + "|"
        lines.append(header)
        lines.append(sep)

        # Get tasks for this dimension from first model
        all_scores = db.get_model_scores(run_id, models[0])
        dim_tasks = [s for s in all_scores if s["dimension"] == dim]

        for task_info in sorted(dim_tasks, key=lambda x: x["task_id"]):
            tid = task_info["task_id"]
            diff = task_info["difficulty"][:1].upper()
            cells = []
            turns_str = ""
            for model in models:
                model_scores = db.get_model_scores(run_id, model)
                task_score = next((s for s in model_scores if s["task_id"] == tid), None)
                if task_score:
                    score = task_score["weighted_score"] or 0
                    cells.append(f"{score:.2f}")
                    if model == models[0]:
                        # Get turn/tc info from task_results
                        tr = _get_task_result(db, run_id, model, tid)
                        if tr:
                            turns_str = (
                                f"{tr['total_turns']} | {tr['total_tool_calls']} "
                                f"| {tr['repaired_tool_calls']} | {tr['unscored_weight']:.0%}"
                            )
                else:
                    cells.append("-")

            if not turns_str:
                turns_str = "- | - | - | -"

            short_id = tid.replace(f"{dim}_", "")
            lines.append(f"| {short_id} | {diff} | " + " | ".join(cells) + f" | {turns_str} |")
        lines.append("")

    # Latency comparison
    lines.append("## Latency")
    lines.append("")
    lines.append("| Model | Avg (ms) | Total (s) |")
    lines.append("|-------|----------|-----------|")
    for model in models:
        scores = db.get_model_scores(run_id, model)
        latencies = [s["latency_ms"] for s in scores if s["latency_ms"]]
        if latencies:
            avg_ms = sum(latencies) / len(latencies)
            total_s = sum(latencies) / 1000
            lines.append(f"| {model} | {avg_ms:.0f} | {total_s:.0f} |")
    lines.append("")

    # Malformed tool-call rate. Repaired calls would otherwise be invisible:
    # the parser salvages them, so nothing in the scores reflects that a model
    # emitted broken JSON. qwen3.8 was measured at 23 repairs / 28 tool calls
    # on coding_mcp_hard_01 — a model-quality signal worth reporting.
    repair_rows = []
    for model in models:
        cur = db._conn.execute(
            "SELECT COALESCE(SUM(repaired_tool_calls), 0), COALESCE(SUM(total_tool_calls), 0) "
            "FROM task_results WHERE run_id = ? AND model_name = ?",
            (run_id, model),
        )
        repaired, total_tc = cur.fetchone()
        if repaired:
            pct = (repaired / total_tc * 100) if total_tc else 0.0
            repair_rows.append(f"| {model} | {repaired} | {total_tc} | {pct:.0f}% |")

    if repair_rows:
        lines.append("## Malformed Tool Calls (repaired)")
        lines.append("")
        lines.append("Tool calls that parsed only after JSON repair. A high rate means the")
        lines.append("model reliably emits broken tool-call JSON; the harness salvages these")
        lines.append("rather than discarding the call, so scores reflect the work, not the defect.")
        lines.append("")
        lines.append("| Model | Repaired | Tool Calls | Rate |")
        lines.append("|-------|----------|------------|------|")
        lines.extend(repair_rows)
        lines.append("")

    # Partial measurement warning. A dimension carrying excluded weight is not
    # comparable to one that is fully scored, and is not comparable to its own
    # historical values from before the criteria were excluded.
    partial = []
    for model in models:
        cur = db._conn.execute(
            "SELECT dimension, AVG(COALESCE(unscored_weight, 0)) FROM task_results "
            "WHERE run_id = ? AND model_name = ? GROUP BY dimension HAVING AVG(COALESCE(unscored_weight, 0)) > 0",
            (run_id, model),
        )
        for dim, frac in cur.fetchall():
            partial.append(f"| {model} | {dim} | {frac:.0%} |")

    if partial:
        lines.append("## Partially Scored Dimensions")
        lines.append("")
        lines.append("Criteria with no implementation are excluded from the weighted average")
        lines.append("rather than scored 0 or 1. These dimensions are therefore scored over a")
        lines.append("subset of their declared criteria — the numbers are narrower claims, not")
        lines.append("better results, and are **not comparable** to fully scored dimensions or")
        lines.append("to historical runs.")
        lines.append("")
        lines.append("| Model | Dimension | Weight excluded |")
        lines.append("|-------|-----------|-----------------|")
        lines.extend(partial)
        lines.append("")

    # Notable results
    lines.append("## Notable Results")
    lines.append("")
    for model in models:
        scores = db.get_model_scores(run_id, model)
        completed = [s for s in scores if s["weighted_score"] is not None]
        if not completed:
            lines.append(f"**{model}:** skipped (no completed tasks)")
            lines.append("")
            continue
        best = max(completed, key=lambda s: s["weighted_score"])
        worst = min(completed, key=lambda s: s["weighted_score"])
        lines.append(f"**{model}:**")
        lines.append(f"- Best: {best['task_id']} ({best['weighted_score']:.2f})")
        lines.append(f"- Worst: {worst['task_id']} ({worst['weighted_score']:.2f})")
        lines.append("")

    return "\n".join(lines)


def save_report(
    db: ResultsDB,
    run_id: str,
    output_dir: Path,
    weights: dict[str, float] | None = None,
) -> Path:
    """Generate and save a report to disk."""
    report = generate_report(db, run_id, weights)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{run_id}.md"
    path.write_text(report)
    logger.info("Report saved to %s", path)
    return path


def _get_run_info(db: ResultsDB, run_id: str) -> dict | None:
    cur = db._conn.execute(
        "SELECT started_at, finished_at, status, models FROM eval_runs WHERE run_id = ?",
        (run_id,),
    )
    row = cur.fetchone()
    if not row:
        return None
    return {"started_at": row[0], "finished_at": row[1], "status": row[2], "models": row[3]}


def _get_task_result(db: ResultsDB, run_id: str, model: str, task_id: str) -> dict | None:
    cur = db._conn.execute(
        "SELECT total_turns, total_tool_calls, total_latency_ms, "
        "COALESCE(repaired_tool_calls, 0) AS repaired_tool_calls, "
        "COALESCE(unscored_weight, 0) AS unscored_weight "
        "FROM task_results WHERE run_id = ? AND model_name = ? AND task_id = ?",
        (run_id, model, task_id),
    )
    row = cur.fetchone()
    if not row:
        return None
    return {
        "total_turns": row[0],
        "total_tool_calls": row[1],
        "latency_ms": row[2],
        "repaired_tool_calls": row[3],
        "unscored_weight": row[4],
    }
