#!/usr/bin/env python3
"""Terminal app for hand-scoring calibration examples.

Walks through each example, displays context, and asks for human scores.
Saves progress after each entry so you can quit and resume.

Usage:
    uv run python scripts/score_calibration.py
    uv run python scripts/score_calibration.py --input judges/calibration/calibration_set.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console = Console()

RUBRICS = {
    "coverage": {
        "description": "Does the response address all aspects of the question?",
        "scale": {
            1: "Misses most aspects",
            2: "Covers some, misses key ones",
            3: "Covers most aspects adequately",
            4: "Covers all aspects well",
            5: "Comprehensive, addresses all aspects with depth",
        },
    },
    "synthesis": {
        "description": "Does the response connect ideas and draw insights, or just list facts?",
        "scale": {
            1: "Pure list, no connections",
            2: "Mostly listing with superficial links",
            3: "Some meaningful connections drawn",
            4: "Good synthesis, clear cross-cutting themes",
            5: "Excellent synthesis, original insights from combining sources",
        },
    },
    "accuracy": {
        "description": "Are the claims factually correct and consistent with the data provided?",
        "scale": {
            1: "Major factual errors or hallucinations",
            2: "Some errors or unsupported claims",
            3: "Mostly accurate, minor issues",
            4: "Accurate, well-supported claims",
            5: "Fully accurate, no hallucinations, properly caveated",
        },
    },
    "recommendation_quality": {
        "description": "Is the recommendation clear, reasoned, and actionable?",
        "scale": {
            1: "No clear recommendation or generic platitude",
            2: "Vague recommendation without reasoning",
            3: "Clear recommendation with basic reasoning",
            4: "Well-reasoned recommendation with trade-offs",
            5: "Excellent: specific, reasoned, addresses constraints, actionable",
        },
    },
    "specificity": {
        "description": "Are the steps/tasks concrete and actionable, not vague?",
        "scale": {
            1: "Entirely vague ('set up backend')",
            2: "Mostly vague with some specifics",
            3: "Mix of specific and general",
            4: "Mostly specific and actionable",
            5: "All steps are concrete with clear deliverables",
        },
    },
    "risk_awareness": {
        "description": "Does the plan identify risks and mitigation strategies?",
        "scale": {
            1: "No risks mentioned",
            2: "Mentions risks without mitigation",
            3: "Identifies key risks with basic mitigation",
            4: "Good risk coverage with specific mitigation",
            5: "Thorough: risks, mitigations, fallbacks, rollback plan",
        },
    },
    "completeness": {
        "description": "Does the plan cover all necessary phases/components?",
        "scale": {
            1: "Major gaps in coverage",
            2: "Covers some phases, misses critical ones",
            3: "Covers main phases adequately",
            4: "Comprehensive coverage of all phases",
            5: "Complete: all phases, dependencies, and edge cases",
        },
    },
    "dependency_correctness": {
        "description": "Are task dependencies and ordering correct?",
        "scale": {
            1: "Dependencies wrong or ignored",
            2: "Some ordering issues",
            3: "Mostly correct ordering",
            4: "Correct dependencies with clear reasoning",
            5: "Perfect ordering with explicit dependency justification",
        },
    },
    "feasibility": {
        "description": "Is the plan realistic given the stated constraints?",
        "scale": {
            1: "Unrealistic or ignores constraints",
            2: "Partially realistic, some overreach",
            3: "Feasible with some optimistic estimates",
            4: "Realistic and well-scoped",
            5: "Pragmatic: right-sized, acknowledges trade-offs",
        },
    },
    "code_quality": {
        "description": "Is the code idiomatic, well-typed, and well-structured?",
        "scale": {
            1: "Poor: no types, bad naming, no structure",
            2: "Below average: some types, messy structure",
            3: "Adequate: types present, reasonable structure",
            4: "Good: idiomatic, typed, clean separation",
            5: "Excellent: production-quality, documented, testable",
        },
    },
    "error_handling": {
        "description": "Does the code handle errors gracefully and specifically?",
        "scale": {
            1: "No error handling",
            2: "Bare except or generic handling",
            3: "Handles main error cases",
            4: "Specific error types, good recovery",
            5: "Comprehensive: specific types, logging, graceful degradation",
        },
    },
    "architecture": {
        "description": "Is the code well-architected with clean abstractions?",
        "scale": {
            1: "Monolithic, no separation",
            2: "Some separation but tangled",
            3: "Reasonable structure",
            4: "Clean separation, testable components",
            5: "Excellent: composable, extensible, right level of abstraction",
        },
    },
    "reasoning_quality": {
        "description": "Does the response show clear reasoning connecting data to conclusions?",
        "scale": {
            1: "No reasoning, just restates data",
            2: "Superficial reasoning",
            3: "Adequate reasoning with some logic",
            4: "Clear reasoning chain from data to conclusion",
            5: "Excellent: explicit reasoning, considers alternatives, well-justified",
        },
    },
    "practical_output": {
        "description": "Is the final output specific, actionable, and useful?",
        "scale": {
            1: "Generic or unhelpful",
            2: "Somewhat useful but vague",
            3: "Useful with some specifics",
            4: "Specific and actionable",
            5: "Directly actionable with concrete next steps",
        },
    },
}


def load_calibration_set(path: Path) -> list[dict]:
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def save_calibration_set(entries: list[dict], path: Path) -> None:
    with open(path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")


def count_scored(entries: list[dict]) -> int:
    return sum(1 for e in entries if e.get("human_scores"))


def display_entry(entry: dict, index: int, total: int) -> None:
    console.clear()

    # Header
    header = Table.grid(padding=(0, 2))
    header.add_row(
        Text(f"[{index + 1}/{total}]", style="bold cyan"),
        Text(entry["id"], style="bold white"),
        Text(f"dimension: {entry['dimension']}", style="yellow"),
    )
    console.print(header)
    console.print()

    # User message
    console.print(Panel(entry["user_message"], title="User Message", border_style="blue"))

    # Model response
    response_text = entry["response"]
    if len(response_text) > 2000:
        response_text = response_text[:2000] + "\n\n... [truncated, full response available with 'f']"
    console.print(Panel(Markdown(response_text), title=f"Model Response ({entry['model']})", border_style="green"))

    # Metadata
    meta = f"Tool calls: {entry.get('tool_calls_made', '?')} | Turns: {entry.get('turns_used', '?')}"
    console.print(Text(meta, style="dim"))
    console.print()


def score_entry(entry: dict) -> dict[str, int]:
    """Prompt for scores on each judge dimension for this entry."""
    dimensions = entry.get("judge_dimensions", [])
    scores: dict[str, int] = {}

    for dim in dimensions:
        rubric = RUBRICS.get(dim)
        if not rubric:
            console.print(f"[yellow]Unknown dimension: {dim}, skipping[/yellow]")
            continue

        # Show rubric
        table = Table(title=f"Score: {dim}", show_header=True, border_style="cyan", width=90)
        table.add_column("Score", style="bold", width=6)
        table.add_column("Description", width=80)
        for score_val, desc in rubric["scale"].items():
            table.add_row(str(score_val), desc)

        console.print(table)
        console.print(f"[dim]{rubric['description']}[/dim]")

        while True:
            try:
                raw = console.input(f"  [bold cyan]{dim}[/bold cyan] (1-5, s=skip, f=full response): ")
                raw = raw.strip().lower()
                if raw == "s":
                    break
                if raw == "f":
                    console.print(Panel(Markdown(entry["response"]), title="Full Response", border_style="green"))
                    continue
                score = int(raw)
                if 1 <= score <= 5:
                    scores[dim] = score
                    break
                console.print("[red]Enter 1-5[/red]")
            except (ValueError, EOFError):
                console.print("[red]Enter 1-5, 's' to skip, or 'f' for full response[/red]")

    return scores


def main():
    parser = argparse.ArgumentParser(description="Score calibration examples")
    parser.add_argument("--input", default="judges/calibration/calibration_set.jsonl")
    args = parser.parse_args()

    path = Path(args.input)
    if not path.exists():
        console.print(f"[red]File not found: {path}[/red]")
        console.print("Run generate_calibration_set.py first.")
        sys.exit(1)

    entries = load_calibration_set(path)
    scored = count_scored(entries)

    console.clear()
    console.print(
        Panel.fit(
            f"[bold]Judge Calibration Scorer[/bold]\n\n"
            f"Examples: {len(entries)}\n"
            f"Already scored: {scored}\n"
            f"Remaining: {len(entries) - scored}\n\n"
            f"[dim]Commands: 1-5 to score, s=skip dimension, f=full response, q=quit[/dim]",
            border_style="cyan",
        )
    )
    console.input("\nPress Enter to start...")

    for i, entry in enumerate(entries):
        if entry.get("human_scores"):
            # Show already-scored entries briefly
            continue

        display_entry(entry, i, len(entries))

        scores = score_entry(entry)
        if scores:
            entry["human_scores"] = scores
            save_calibration_set(entries, path)
            console.print(f"[green]Saved: {scores}[/green]")
        else:
            console.print("[yellow]No scores entered, skipping[/yellow]")

        console.print()
        action = console.input("[dim]Enter=next, q=quit: [/dim]").strip().lower()
        if action == "q":
            console.print(f"\n[cyan]Progress saved. {count_scored(entries)}/{len(entries)} scored.[/cyan]")
            break

    # Final summary
    scored = count_scored(entries)
    console.print(f"\n[bold green]Done! {scored}/{len(entries)} examples scored.[/bold green]")
    if scored == len(entries):
        console.print("Run: [bold]uv run python scripts/run_calibration.py[/bold]")


if __name__ == "__main__":
    main()
