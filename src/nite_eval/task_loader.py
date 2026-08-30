"""Load task definitions from tasks/ directory.

Tasks are YAML files organized by dimension: tasks/{dimension}/{task_id}.yaml
Each task is a self-contained definition with tools, mock responses, scoring, etc.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

DEFAULT_TASKS_DIR = Path("tasks")


@dataclass
class TaskDefinition:
    """A single evaluation task loaded from YAML."""

    id: str
    dimension: str
    difficulty: str
    description: str
    system_prompt: str
    tools: list[dict[str, Any]]
    user_message: str
    scoring: dict[str, Any]
    max_turns: int = 8
    max_tool_calls: int = 20
    timeout_seconds: int = 90
    # Per-task generation budget. None falls back to evaluation.max_tokens.
    # Coding tasks need far more than the global default: the model writes
    # whole source files inside a JSON string in a tool call, and running out
    # mid-string truncates the call so it never parses.
    max_tokens: int | None = None
    source_project: str = ""
    mock_responses: dict[str, Any] = field(default_factory=dict)
    test_suite: dict[str, Any] = field(default_factory=dict)
    expected_tool_sequence: list[dict[str, Any]] = field(default_factory=list)
    expected_tools_called: list[str] = field(default_factory=list)
    expected_tool_sequence_flexible: bool = False
    # Tools the model should NOT call, for `tool_absence` scoring.
    distractor_tools: list[str] = field(default_factory=list)
    # [before, after] pairs for `tool_ordering` scoring.
    expected_tool_ordering: list[list[str]] = field(default_factory=list)
    # Real execution environment. When present the task runs against a
    # container instead of mocks, and `automated` criteria become measurable.
    environment: dict[str, Any] = field(default_factory=dict)

    @property
    def judge_dimensions(self) -> list[str]:
        """Return scoring dimensions that use judge_rubric method."""
        return [name for name, cfg in self.scoring.items() if cfg.get("method") == "judge_rubric"]

    @property
    def deterministic_dimensions(self) -> list[str]:
        """Return scoring dimensions that use deterministic methods."""
        return [
            name
            for name, cfg in self.scoring.items()
            if cfg.get("method")
            in (
                "deterministic",
                "sequence_match",
                "subset_match",
                "checklist",
                "contains_check",
                "exact_match",
                "partial_match",
                "automated",
                "tool_args_match",
                "tool_absence",
                "tool_ordering",
            )
        ]


def load_task(path: Path) -> TaskDefinition:
    """Load a single task from a YAML file."""
    with open(path) as f:
        data = yaml.safe_load(f)

    return TaskDefinition(
        id=data["id"],
        dimension=data["dimension"],
        difficulty=data["difficulty"],
        description=data.get("description", ""),
        system_prompt=data.get("system_prompt", ""),
        tools=data.get("tools", []),
        user_message=data["user_message"],
        scoring=data.get("scoring", {}),
        max_turns=data.get("max_turns", 8),
        max_tool_calls=data.get("max_tool_calls", 20),
        timeout_seconds=data.get("timeout_seconds", 90),
        max_tokens=data.get("max_tokens"),
        source_project=data.get("source_project", ""),
        mock_responses=data.get("mock_responses", {}),
        test_suite=data.get("test_suite", {}),
        expected_tool_sequence=data.get("expected_tool_sequence", []),
        expected_tools_called=data.get("expected_tools_called", []),
        expected_tool_sequence_flexible=data.get("expected_tool_sequence_flexible", False),
        distractor_tools=data.get("distractor_tools", []),
        expected_tool_ordering=data.get("expected_tool_ordering", []),
        environment=data.get("environment", {}),
    )


def load_tasks(
    tasks_dir: Path = DEFAULT_TASKS_DIR,
    dimension: str | None = None,
    difficulty: str | None = None,
) -> list[TaskDefinition]:
    """Load all tasks, optionally filtered by dimension and/or difficulty."""
    if not tasks_dir.exists():
        logger.warning("Tasks directory not found: %s", tasks_dir)
        return []

    tasks: list[TaskDefinition] = []
    for path in sorted(tasks_dir.rglob("*.yaml")):
        task = load_task(path)
        if dimension and task.dimension != dimension:
            continue
        if difficulty and task.difficulty != difficulty:
            continue
        tasks.append(task)

    logger.info("Loaded %d tasks from %s", len(tasks), tasks_dir)
    return tasks
