#!/usr/bin/env python3
"""Run all judge candidates on calibration set and compare against human scores.

Computes Cohen's κ per dimension and overall for each judge candidate.
Produces a comparison report.

Usage:
    # Start each judge one at a time on port 8081, run this script for each:
    uv run python scripts/run_calibration.py --judge-model selene-1-mini \
        --judge-gguf ~/models/Selene-1-Mini-Llama-3.1-8B-Q6_K.gguf

    # Or if the judge is already running:
    uv run python scripts/run_calibration.py --judge-model selene-1-mini --skip-start
"""

import argparse
import json
import math
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

from rich.console import Console
from rich.table import Table

from nite_eval.judge import JudgeClient, JudgeResult
from nite_eval.model_manager import check_health

console = Console()

CALIBRATION_PATH = Path("judges/calibration/calibration_set.jsonl")
RESULTS_DIR = Path("judges/calibration/results")

LLAMA_SERVER = "/home/woojay/P/llama.cpp/build/bin/llama-server"
JUDGE_PORT = 8081

# Rubric text used for judge prompts (matches score_calibration.py rubrics)
JUDGE_RUBRICS = {
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
}


def load_calibration_set(path: Path) -> list[dict]:
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def cohens_kappa(human_scores: list[int], judge_scores: list[int]) -> float:
    """Compute Cohen's κ for ordinal data (linear weighted)."""
    if len(human_scores) != len(judge_scores) or not human_scores:
        return 0.0

    # Use linear weighting for ordinal scale
    categories = sorted(set(human_scores) | set(judge_scores))
    n = len(categories)
    if n <= 1:
        return 1.0  # Perfect agreement if only one category

    cat_idx = {c: i for i, c in enumerate(categories)}
    k = len(human_scores)

    # Observed agreement (linear weighted)
    max_dist = n - 1
    observed_disagreement = (
        sum(abs(cat_idx[h] - cat_idx[j]) / max_dist for h, j in zip(human_scores, judge_scores, strict=False)) / k
    )

    # Expected disagreement under independence
    h_counts = defaultdict(int)
    j_counts = defaultdict(int)
    for h in human_scores:
        h_counts[h] += 1
    for j in judge_scores:
        j_counts[j] += 1

    expected_disagreement = 0.0
    for h_cat in categories:
        for j_cat in categories:
            expected_disagreement += (
                (h_counts[h_cat] / k) * (j_counts[j_cat] / k) * (abs(cat_idx[h_cat] - cat_idx[j_cat]) / max_dist)
            )

    if expected_disagreement == 0:
        return 1.0

    return 1.0 - (observed_disagreement / expected_disagreement)


def mean_absolute_error(human_scores: list[int], judge_scores: list[int]) -> float:
    if not human_scores:
        return 0.0
    return sum(abs(h - j) for h, j in zip(human_scores, judge_scores, strict=False)) / len(human_scores)


def correlation(human_scores: list[int], judge_scores: list[int]) -> float:
    """Pearson correlation coefficient."""
    n = len(human_scores)
    if n < 2:
        return 0.0

    h_mean = sum(human_scores) / n
    j_mean = sum(judge_scores) / n

    numerator = sum((h - h_mean) * (j - j_mean) for h, j in zip(human_scores, judge_scores, strict=False))
    h_var = sum((h - h_mean) ** 2 for h in human_scores)
    j_var = sum((j - j_mean) ** 2 for j in judge_scores)

    denom = math.sqrt(h_var * j_var)
    if denom == 0:
        return 0.0
    return numerator / denom


def start_judge_server(gguf_path: str, gpu_id: int = 1) -> subprocess.Popen:
    """Start a judge model server."""
    cmd = [
        LLAMA_SERVER,
        "-m",
        gguf_path,
        "--port",
        str(JUDGE_PORT),
        "-ngl",
        "999",
        "--ctx-size",
        "8192",
        "-fa",
        "on",
        "--no-webui",
    ]
    console.print(f"Starting judge: {Path(gguf_path).name}")
    proc = subprocess.Popen(
        cmd,
        env={**subprocess.os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_id)},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Wait for ready
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        if check_health(f"http://127.0.0.1:{JUDGE_PORT}"):
            console.print("[green]Judge server ready[/green]")
            return proc
        time.sleep(1)

    proc.kill()
    console.print("[red]Judge server failed to start[/red]")
    sys.exit(1)


def stop_judge_server(proc: subprocess.Popen) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)
    # Wait for port to free
    time.sleep(2)


def run_judge_on_entries(
    entries: list[dict],
    judge_model: str,
) -> dict[str, dict[str, float]]:
    """Run a judge on all entries. Returns {entry_id: {dimension: score}}."""
    judge = JudgeClient(
        base_url=f"http://127.0.0.1:{JUDGE_PORT}/v1",
        model=judge_model,
    )

    results: dict[str, dict[str, float]] = {}

    for i, entry in enumerate(entries):
        if not entry.get("human_scores"):
            continue

        entry_scores: dict[str, float] = {}
        dims = entry.get("judge_dimensions", [])

        console.print(f"  [{i + 1}/{len(entries)}] {entry['id']}: ", end="")

        for dim in dims:
            if dim not in entry["human_scores"]:
                continue

            rubric = JUDGE_RUBRICS.get(dim, f"Rate {dim} from 1 (poor) to 5 (excellent)")

            result = judge.evaluate(
                dimension=dim,
                rubric=rubric,
                task_description=entry["user_message"],
                model_response=entry["response"],
            )

            if isinstance(result, JudgeResult):
                # Clamp to 1-5 range
                score = max(1.0, min(5.0, round(result.score)))
                entry_scores[dim] = score
            else:
                console.print("[red]ERR[/red] ", end="")
                entry_scores[dim] = 3.0  # Default on error

        results[entry["id"]] = entry_scores
        scores_str = " ".join(f"{d}={s:.0f}" for d, s in entry_scores.items())
        console.print(scores_str)

    judge.close()
    return results


def print_comparison_report(
    entries: list[dict],
    judge_results: dict[str, dict[str, dict[str, float]]],
) -> None:
    """Print a rich comparison table."""
    console.print()
    console.print("[bold]═══ Judge Calibration Report ═══[/bold]")
    console.print()

    # Collect per-dimension scores across all judges
    all_dimensions: set[str] = set()
    for entry in entries:
        for dim in entry.get("human_scores", {}):
            all_dimensions.add(dim)

    for judge_name, judge_data in judge_results.items():
        console.print(f"\n[bold cyan]Judge: {judge_name}[/bold cyan]")

        # Per-dimension metrics
        table = Table(show_header=True, border_style="cyan")
        table.add_column("Dimension", style="bold")
        table.add_column("N", justify="right")
        table.add_column("κ (weighted)", justify="right")
        table.add_column("Pearson r", justify="right")
        table.add_column("MAE", justify="right")
        table.add_column("Status", justify="center")

        dim_kappas: list[float] = []

        for dim in sorted(all_dimensions):
            human_list: list[int] = []
            judge_list: list[int] = []

            for entry in entries:
                hs = entry.get("human_scores", {})
                if dim not in hs:
                    continue
                eid = entry["id"]
                if eid in judge_data and dim in judge_data[eid]:
                    human_list.append(int(hs[dim]))
                    judge_list.append(int(judge_data[eid][dim]))

            if not human_list:
                continue

            k = cohens_kappa(human_list, judge_list)
            r = correlation(human_list, judge_list)
            mae = mean_absolute_error(human_list, judge_list)
            dim_kappas.append(k)

            status = "[green]PASS[/green]" if k >= 0.6 else "[red]FAIL[/red]"

            table.add_row(
                dim,
                str(len(human_list)),
                f"{k:.3f}",
                f"{r:.3f}",
                f"{mae:.2f}",
                status,
            )

        console.print(table)

        # Overall
        if dim_kappas:
            avg_kappa = sum(dim_kappas) / len(dim_kappas)
            status = "[bold green]PASS[/bold green]" if avg_kappa >= 0.6 else "[bold red]FAIL[/bold red]"
            console.print(f"  Overall κ (mean): {avg_kappa:.3f}  {status}")


def main():
    parser = argparse.ArgumentParser(description="Run judge calibration")
    parser.add_argument("--input", default=str(CALIBRATION_PATH))
    parser.add_argument("--judge-model", required=True, help="Judge model name for API calls")
    parser.add_argument("--judge-gguf", help="Path to GGUF file (starts server automatically)")
    parser.add_argument("--skip-start", action="store_true", help="Judge already running on :8081")
    parser.add_argument("--gpu", type=int, default=1, help="GPU ID for judge server")
    args = parser.parse_args()

    path = Path(args.input)
    if not path.exists():
        console.print(f"[red]Not found: {path}[/red]")
        sys.exit(1)

    entries = load_calibration_set(path)
    scored = [e for e in entries if e.get("human_scores")]
    if not scored:
        console.print("[red]No human-scored entries found. Run score_calibration.py first.[/red]")
        sys.exit(1)

    console.print(f"Loaded {len(scored)} human-scored entries")

    # Start judge if needed
    proc = None
    if not args.skip_start:
        if not args.judge_gguf:
            console.print("[red]Provide --judge-gguf or --skip-start[/red]")
            sys.exit(1)
        proc = start_judge_server(args.judge_gguf, args.gpu)
    else:
        if not check_health(f"http://127.0.0.1:{JUDGE_PORT}"):
            console.print(f"[red]No judge server on :{JUDGE_PORT}[/red]")
            sys.exit(1)

    try:
        console.print(f"\n[bold]Running judge: {args.judge_model}[/bold]")
        results = run_judge_on_entries(entries, args.judge_model)

        # Save results
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        result_path = RESULTS_DIR / f"{args.judge_model}.json"
        with open(result_path, "w") as f:
            json.dump(results, f, indent=2)
        console.print(f"Saved to {result_path}")

    finally:
        if proc:
            console.print("Stopping judge server...")
            stop_judge_server(proc)

    # Load all judge results for comparison
    all_judge_results: dict[str, dict[str, dict[str, float]]] = {}
    for result_file in RESULTS_DIR.glob("*.json"):
        judge_name = result_file.stem
        with open(result_file) as f:
            all_judge_results[judge_name] = json.load(f)

    print_comparison_report(entries, all_judge_results)


if __name__ == "__main__":
    main()
