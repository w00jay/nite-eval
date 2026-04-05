"""Tests for task_loader module."""

from pathlib import Path

from nite_eval.task_loader import load_task, load_tasks

TASKS_DIR = Path("tasks")


def test_load_all_tasks():
    tasks = load_tasks(TASKS_DIR)
    assert len(tasks) == 15


def test_load_by_dimension():
    for dim, expected in [("research", 3), ("planning", 3), ("coding", 4), ("agentic", 5)]:
        tasks = load_tasks(TASKS_DIR, dimension=dim)
        assert len(tasks) == expected, f"{dim}: expected {expected}, got {len(tasks)}"


def test_load_by_difficulty():
    for diff, expected in [("easy", 4), ("medium", 6), ("hard", 5)]:
        tasks = load_tasks(TASKS_DIR, difficulty=diff)
        assert len(tasks) == expected, f"{diff}: expected {expected}, got {len(tasks)}"


def test_load_combined_filter():
    tasks = load_tasks(TASKS_DIR, dimension="agentic", difficulty="hard")
    assert len(tasks) == 2
    assert all(t.dimension == "agentic" and t.difficulty == "hard" for t in tasks)


def test_load_single_task():
    task = load_task(TASKS_DIR / "agentic" / "agentic_brain_easy_01.yaml")
    assert task.id == "agentic_brain_easy_01"
    assert task.dimension == "agentic"
    assert task.difficulty == "easy"
    assert len(task.tools) == 3
    assert "autoresearch" in task.user_message
    assert task.max_turns == 6
    assert task.timeout_seconds == 60


def test_task_has_mock_responses_or_test_suite():
    for task in load_tasks(TASKS_DIR):
        assert task.mock_responses or task.test_suite, f"{task.id} has neither mock_responses nor test_suite"


def test_scoring_weights_sum():
    for task in load_tasks(TASKS_DIR):
        weights = [cfg["weight"] for cfg in task.scoring.values() if "weight" in cfg]
        assert abs(sum(weights) - 1.0) < 0.01, f"{task.id}: weights sum to {sum(weights)}"


def test_judge_dimensions_property():
    task = load_task(TASKS_DIR / "research" / "research_mcp_easy_01.yaml")
    judge_dims = task.judge_dimensions
    assert "accuracy" in judge_dims
    assert "recommendation_quality" in judge_dims
    # checklist method should not be in judge_dimensions
    assert "coverage" not in judge_dims


def test_deterministic_dimensions_property():
    task = load_task(TASKS_DIR / "agentic" / "agentic_brain_easy_01.yaml")
    det_dims = task.deterministic_dimensions
    assert "tool_selection" in det_dims
    assert "distractor_avoidance" in det_dims
    # judge_rubric should not be in deterministic
    assert "synthesis" not in det_dims


def test_nonexistent_dir_returns_empty():
    tasks = load_tasks(Path("/nonexistent/dir"))
    assert tasks == []
