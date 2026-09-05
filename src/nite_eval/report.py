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

from nite_eval.conversation_runner import NO_ANSWER_PREFIX
from nite_eval.results_db import ResultsDB  # noqa: TC001
from nite_eval.scoring import compute_composite

logger = logging.getLogger(__name__)

# Composite gaps smaller than this are inside judge variance. The target is
# essentially deterministic at temperature 0 — the same task reproduces to
# identical scores — so the variance that remains is the judge's, which is why
# judge averaging is the lever and repeat target runs are not.
MIN_DETECTABLE_DIFFERENCE = 0.05


def generate_report(
    db: ResultsDB,
    run_id: str,
    weights: dict[str, float] | None = None,
    mdd: float = MIN_DETECTABLE_DIFFERENCE,
    dimension_mdd: dict[str, float] | None = None,
    judge_samples: int = 3,
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

    # Rank models, and say which places in that ranking are not real.
    ranked = sorted(model_composites.items(), key=lambda x: x[1], reverse=True)
    if len(ranked) > 1:
        lines.append("**Ranking:** " + " > ".join(f"{m} ({s:.2f})" for m, s in ranked))
        lines.append("")

        indistinguishable = [
            (a, b, abs(sa - sb)) for (a, sa), (b, sb) in zip(ranked, ranked[1:], strict=False) if abs(sa - sb) < mdd
        ]
        lines.append("### Resolution")
        lines.append("")
        lines.append(
            f"{total_tasks} tasks, one sample per task, judge scores averaged over "
            f"{judge_samples}. Composite differences below **{mdd:.2f}** are inside "
            "judge variance and carry no information — the ordering above is "
            "directional, not a measurement."
        )
        lines.append("")
        if indistinguishable:
            lines.append("Adjacent pairs that cannot be separated by this run:")
            lines.append("")
            for a, b, gap in indistinguishable:
                lines.append(f"- **{a}** and **{b}** differ by {gap:.3f} — treat as tied")
            lines.append("")
        else:
            lines.append(f"Every adjacent pair differs by more than {mdd:.2f}.")
            lines.append("")

        # Some dimensions are noisier than the composite and must not be read
        # against its threshold.
        if dimension_mdd:
            lines.append(
                "Dimensions with a wider noise floor than the composite, whose gaps "
                "must be read against their own threshold:"
            )
            lines.append("")
            for dim, threshold in sorted(dimension_mdd.items()):
                lines.append(f"- **{dim}**: gaps below {threshold:.2f} carry no information")
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

    # Latency alone measures how long a model took, not how fast it generates,
    # so a terse model and a quick one read the same. LFM2.5-8B-A1B activating
    # ~1B of 8B parameters per token and a dense 2.6B saying less are the case
    # this table exists to separate.
    lines.append("## Latency and throughput")
    lines.append("")
    lines.append("| Model | Avg (ms) | Total (s) | Decode tok/s | Gen tok/s | Avg gen tok | Avg prompt tok |")
    lines.append("|-------|----------|-----------|--------------|-----------|-------------|----------------|")
    measured_any = False
    decode_any = False
    for model in models:
        scores = db.get_model_scores(run_id, model)
        latencies = [s["latency_ms"] for s in scores if s["latency_ms"]]
        if not latencies:
            continue
        avg_ms = sum(latencies) / len(latencies)
        total_s = sum(latencies) / 1000

        # Restricted to rows that actually carry counts, so the divisor matches
        # the numerator. A run that predates this column, or a server that
        # reported no usage, leaves NULL — mixing those tasks' latency into the
        # denominator would understate tok/s instead of admitting it is unknown.
        cur = db._conn.execute(
            "SELECT COALESCE(SUM(completion_tokens), 0), COALESCE(SUM(prompt_tokens), 0), "
            "COALESCE(SUM(total_latency_ms), 0), COUNT(*) "
            "FROM task_results "
            "WHERE run_id = ? AND model_name = ? AND completion_tokens IS NOT NULL",
            (run_id, model),
        )
        gen_tok, prompt_tok, tok_latency_ms, measured = cur.fetchone()

        # Decode speed is a separate query on purpose: predicted_* divides into
        # itself, never into total_latency_ms, and a run may carry usage without
        # timings (or the reverse) depending on when it ran.
        dcur = db._conn.execute(
            "SELECT COALESCE(SUM(predicted_n), 0), COALESCE(SUM(predicted_ms), 0) "
            "FROM task_results "
            "WHERE run_id = ? AND model_name = ? AND predicted_n IS NOT NULL",
            (run_id, model),
        )
        dec_n, dec_ms = dcur.fetchone()
        if dec_n and dec_ms > 0:
            decode_any = True
            decode_s = f"{dec_n / (dec_ms / 1000):.1f}"
        else:
            decode_s = "—"
        if measured and gen_tok and tok_latency_ms > 0:
            measured_any = True
            tok_s = f"{gen_tok / (tok_latency_ms / 1000):.1f}"
            avg_gen = f"{gen_tok / measured:.0f}"
            avg_prompt = f"{prompt_tok / measured:.0f}" if prompt_tok else "—"
        else:
            tok_s = avg_gen = avg_prompt = "—"
        lines.append(f"| {model} | {avg_ms:.0f} | {total_s:.0f} | {decode_s} | {tok_s} | {avg_gen} | {avg_prompt} |")
    lines.append("")
    if measured_any:
        lines.append(
            "Gen tok/s is generated tokens over wall-clock request time, so it includes "
            "prompt processing and tool-result round trips — it is end-to-end throughput "
            "for the task, not decode speed. Avg prompt tok is per task across all its "
            "turns, so it grows with conversation length and is what history compaction acts on."
        )
    if decode_any:
        lines.append("")
        lines.append(
            "**Decode tok/s is the column to compare models on.** It comes from the "
            "server's `timings` block (`predicted_n` / `predicted_ms`) and excludes "
            "prompt processing entirely, so it is unaffected by how many turns a task "
            "took. Gen tok/s is confounded by exactly that: a model that loops spends "
            "its wall clock on prompt processing and reads slow even when its decoder "
            "is fast. Where the two disagree, the gap is turn count, not speed."
        )
    elif measured_any:
        lines.append("")
        lines.append(
            "Decode tok/s unavailable — this run predates the `predicted_ms` column, "
            "or the server reported no `timings` block."
        )
    else:
        lines.append(
            "Token counts unavailable for this run — either it predates the "
            "`completion_tokens` column or the server reported no usage block."
        )
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

    # Fixture gaps. A score cannot tell "answered badly" from "the fixture had
    # nothing to answer with", and the difference was previously visible only as
    # a warning in the run log. A model penalised here was not necessarily worse.
    gap_rows = []
    gap_samples: list[tuple[str, str, str]] = []
    for model in models:
        cur = db._conn.execute(
            "SELECT task_id, unmatched_mock_calls, unmatched_mock_samples FROM task_results "
            "WHERE run_id = ? AND model_name = ? AND COALESCE(unmatched_mock_calls, 0) > 0 "
            "ORDER BY unmatched_mock_calls DESC",
            (run_id, model),
        )
        for task_id, n, samples in cur.fetchall():
            gap_rows.append(f"| {model} | {task_id} | {n} |")
            rendered = _render_unmatched_sample(samples)
            if rendered:
                gap_samples.append((model, task_id, rendered))

    if gap_rows:
        lines.append("## Unanswered Tool Calls (mock gaps)")
        lines.append("")
        lines.append("Calls the task's mocks could not answer. Two different things produce")
        lines.append("this and the count cannot tell them apart:")
        lines.append("")
        lines.append("- **The fixture was too narrow.** The model made a reasonable call the")
        lines.append("  mocks did not anticipate — searching `Nvidia` against a matcher keyed")
        lines.append("  on `NVDA`. The model is right and the score is unfairly depressed.")
        lines.append("- **The model emitted a call nothing could match.** A wrong shape, an")
        lines.append("  argument nested a level too deep, a tool that was never declared. The")
        lines.append("  fixture is right and so is the score.")
        lines.append("")
        lines.append("Either way the model was handed an error and scored on what it did next.")
        lines.append("The recorded arguments below are what separates the two cases; read them")
        lines.append("before attributing a low score to either the harness or the model.")
        lines.append("")
        lines.append("| Model | Task | Unanswered |")
        lines.append("|-------|------|------------|")
        lines.extend(gap_rows)
        lines.append("")
        if gap_samples:
            lines.append("### Recorded arguments")
            lines.append("")
            for model, task_id, rendered in gap_samples:
                lines.append(f"**{model} / {task_id}**")
                lines.append("")
                lines.append("```")
                lines.append(rendered)
                lines.append("```")
                lines.append("")
        else:
            lines.append("No arguments were recorded — this run predates the")
            lines.append("`unmatched_mock_samples` column, so the cause cannot be read off the")
            lines.append("report and must be dug out of the run log.")
            lines.append("")

    # Tasks where the model never answered at all. These score near zero for a
    # termination failure rather than a wrong answer, and a dimension mean
    # cannot tell those apart: Qwopus3.8-27B-Flash lost most of a research
    # dimension to one such task (0.00 on research_finance_hard_01, dragging
    # 0.77 to 0.56) while looking merely mediocre in the summary table.
    no_answer = db._conn.execute(
        "SELECT model_name, task_id, dimension, total_turns, total_tool_calls, weighted_score "
        "FROM task_results "
        "WHERE run_id = ? AND final_response LIKE ? "
        "ORDER BY model_name, task_id",
        (run_id, NO_ANSWER_PREFIX + "%"),
    ).fetchall()
    if no_answer:
        lines.append("## Tasks That Produced No Answer")
        lines.append("")
        lines.append("Every turn ended in a tool call and no final answer was ever emitted.")
        lines.append("")
        lines.append(
            "These score near zero for **failing to terminate**, not for answering "
            "badly, and a dimension average cannot tell those apart. A model that "
            "loops here is not necessarily weak at the dimension — read this table "
            "before attributing a low score to capability."
        )
        lines.append("")
        lines.append("| Model | Task | Dimension | Turns | Tool calls | Score |")
        lines.append("|-------|------|-----------|-------|------------|-------|")
        for model, task_id, dim, turns, tcs, score in no_answer:
            lines.append(
                f"| {model} | {task_id} | {dim} | {turns or 0} | {tcs or 0} | "
                f"{(score if score is not None else 0):.2f} |"
            )
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
    mdd: float = MIN_DETECTABLE_DIFFERENCE,
    dimension_mdd: dict[str, float] | None = None,
    judge_samples: int = 3,
) -> Path:
    """Generate and save a report to disk."""
    report = generate_report(db, run_id, weights, mdd=mdd, dimension_mdd=dimension_mdd, judge_samples=judge_samples)
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


def _render_unmatched_sample(raw: str | None) -> str:
    """Format the stored sample of unanswered calls for the report.

    Returns "" for a run recorded before the column existed, so the report can
    say the arguments are unavailable rather than imply there were none.
    """
    if not raw:
        return ""
    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return ""
    sample = payload.get("sample") or []
    if not sample:
        return ""
    out = []
    for call in sample:
        args = call.get("arguments", "")
        if call.get("truncated"):
            args += " …"
        out.append(f"{call.get('name', '?')}({args})  [{call.get('reason', 'unknown')}]")
    total = payload.get("total", len(sample))
    if total > len(sample):
        out.append(f"… and {total - len(sample)} more")
    return "\n".join(out)


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
