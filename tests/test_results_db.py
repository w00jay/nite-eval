"""Tests for results_db module."""

import tempfile

from nite_eval.results_db import ResultsDB


def _make_db() -> ResultsDB:
    tmp = tempfile.mktemp(suffix=".db")
    return ResultsDB(tmp)


def test_create_and_finish_run():
    with _make_db() as db:
        db.create_run("run-001", ["qwen3.5-9b", "gemma4-26b"])
        assert db.get_run_status("run-001") == "running"
        db.finish_run("run-001", "completed")
        assert db.get_run_status("run-001") == "completed"


def test_nonexistent_run():
    with _make_db() as db:
        assert db.get_run_status("nope") is None


def test_register_and_get_pending():
    with _make_db() as db:
        db.create_run("run-001", ["model-a"])
        db.register_tasks(
            "run-001",
            ["model-a"],
            [
                ("task1", "research", "easy"),
                ("task2", "agentic", "hard"),
            ],
        )
        pending = db.get_pending_tasks("run-001", "model-a")
        assert len(pending) == 2
        assert {p.task_id for p in pending} == {"task1", "task2"}


def test_checkpoint_flow():
    with _make_db() as db:
        db.create_run("run-001", ["model-a"])
        db.register_tasks(
            "run-001",
            ["model-a"],
            [
                ("task1", "research", "easy"),
                ("task2", "agentic", "hard"),
            ],
        )

        # Start and complete task1
        db.mark_task_running("run-001", "model-a", "task1")
        db.save_task_result(
            run_id="run-001",
            model_name="model-a",
            task_id="task1",
            final_response="The answer is 42.",
            total_turns=3,
            total_tool_calls=2,
            total_latency_ms=1500.0,
            reached_max_turns=False,
            weighted_score=0.75,
        )

        # task2 should still be pending
        pending = db.get_pending_tasks("run-001", "model-a")
        assert len(pending) == 1
        assert pending[0].task_id == "task2"


def test_save_and_retrieve_scores():
    with _make_db() as db:
        db.create_run("run-001", ["model-a"])
        db.register_tasks("run-001", ["model-a"], [("task1", "research", "easy")])
        db.mark_task_running("run-001", "model-a", "task1")

        db.save_score(
            run_id="run-001",
            model_name="model-a",
            task_id="task1",
            dimension="accuracy",
            method="judge_rubric",
            raw_score=3.0,
            normalized=0.5,
            weight=0.3,
            judge_model="reward-anything",
            reasoning="Mostly accurate with minor issues",
        )
        db.save_score(
            run_id="run-001",
            model_name="model-a",
            task_id="task1",
            dimension="coverage",
            method="checklist",
            raw_score=0.8,
            normalized=0.8,
            weight=0.4,
        )

        db.save_task_result(
            run_id="run-001",
            model_name="model-a",
            task_id="task1",
            final_response="response",
            total_turns=2,
            total_tool_calls=1,
            total_latency_ms=800.0,
            reached_max_turns=False,
            weighted_score=0.65,
        )

        scores = db.get_model_scores("run-001", "model-a")
        assert len(scores) == 1
        assert scores[0]["weighted_score"] == 0.65


def test_tool_call_recording():
    with _make_db() as db:
        db.create_run("run-001", ["model-a"])
        db.register_tasks("run-001", ["model-a"], [("task1", "agentic", "easy")])

        db.save_tool_calls(
            "run-001",
            "model-a",
            "task1",
            [
                {
                    "turn": 1,
                    "call_index": 0,
                    "tool_name": "web_search",
                    "arguments": {"query": "test"},
                    "result": {"content": "found"},
                },
                {
                    "turn": 2,
                    "call_index": 0,
                    "tool_name": "fetch_url",
                    "arguments": {"url": "http://x"},
                    "result": {"content": "page"},
                },
            ],
        )

        calls = db.get_tool_calls("run-001", "model-a", "task1")
        assert len(calls) == 2
        assert calls[0]["name"] == "web_search"
        assert calls[0]["arguments"] == {"query": "test"}
        assert calls[1]["name"] == "fetch_url"


def test_dimension_averages():
    with _make_db() as db:
        db.create_run("run-001", ["model-a"])
        db.register_tasks(
            "run-001",
            ["model-a"],
            [
                ("t1", "research", "easy"),
                ("t2", "research", "hard"),
                ("t3", "agentic", "easy"),
            ],
        )
        for tid, score in [("t1", 0.8), ("t2", 0.6), ("t3", 0.9)]:
            db.save_task_result(
                run_id="run-001",
                model_name="model-a",
                task_id=tid,
                final_response="r",
                total_turns=1,
                total_tool_calls=0,
                total_latency_ms=100.0,
                reached_max_turns=False,
                weighted_score=score,
            )

        avgs = db.get_dimension_averages("run-001", "model-a")
        assert abs(avgs["research"] - 0.7) < 0.01
        assert abs(avgs["agentic"] - 0.9) < 0.01


def test_run_summary():
    with _make_db() as db:
        db.create_run("run-001", ["model-a", "model-b"])
        db.register_tasks(
            "run-001",
            ["model-a", "model-b"],
            [
                ("t1", "research", "easy"),
            ],
        )
        db.save_task_result(
            run_id="run-001",
            model_name="model-a",
            task_id="t1",
            final_response="r",
            total_turns=1,
            total_tool_calls=0,
            total_latency_ms=100.0,
            reached_max_turns=False,
            weighted_score=0.8,
        )
        db.save_task_result(
            run_id="run-001",
            model_name="model-b",
            task_id="t1",
            final_response="r",
            total_turns=1,
            total_tool_calls=0,
            total_latency_ms=200.0,
            reached_max_turns=False,
            weighted_score=0.6,
        )

        summary = db.get_run_summary("run-001")
        assert summary["model-a"]["completed"] == 1
        assert summary["model-a"]["avg_score"] == 0.8
        assert summary["model-b"]["avg_score"] == 0.6


def test_failed_task():
    with _make_db() as db:
        db.create_run("run-001", ["model-a"])
        db.register_tasks("run-001", ["model-a"], [("t1", "coding", "hard")])
        db.save_task_result(
            run_id="run-001",
            model_name="model-a",
            task_id="t1",
            final_response="",
            total_turns=0,
            total_tool_calls=0,
            total_latency_ms=0,
            reached_max_turns=False,
            weighted_score=0.0,
            error="http_error: connection refused",
        )
        pending = db.get_pending_tasks("run-001", "model-a")
        assert len(pending) == 0  # failed is not pending


def test_resume_idempotent():
    """Registering tasks twice doesn't create duplicates (INSERT OR IGNORE)."""
    with _make_db() as db:
        db.create_run("run-001", ["model-a"])
        tasks = [("t1", "research", "easy"), ("t2", "agentic", "hard")]
        db.register_tasks("run-001", ["model-a"], tasks)
        db.register_tasks("run-001", ["model-a"], tasks)  # duplicate
        pending = db.get_pending_tasks("run-001", "model-a")
        assert len(pending) == 2


def test_dimension_with_no_completed_tasks_scores_zero():
    """A dimension whose tasks all failed must score 0, not vanish.

    get_dimension_averages filtered to status='completed', so a wholly failed
    dimension returned no key at all. The report rendered it as 0.00 via
    .get(d, 0) while compute_composite renormalised over the surviving
    dimensions — so in run-20260831-163006, ornith's four failed coding tasks
    were dropped and its composite read 0.80 instead of 0.60. Failing a
    dimension outright must not beat scoring badly in it.
    """
    with _make_db() as db:
        db.create_run("run-001", ["model-a"])
        db.register_tasks(
            "run-001",
            ["model-a"],
            [("t1", "research", "easy"), ("t2", "coding", "hard")],
        )
        db.save_task_result(
            run_id="run-001",
            model_name="model-a",
            task_id="t1",
            final_response="r",
            total_turns=1,
            total_tool_calls=0,
            total_latency_ms=100.0,
            reached_max_turns=False,
            weighted_score=0.8,
        )
        db.save_task_result(
            run_id="run-001",
            model_name="model-a",
            task_id="t2",
            final_response="",
            total_turns=1,
            total_tool_calls=0,
            total_latency_ms=100.0,
            reached_max_turns=False,
            weighted_score=0.0,
            error="truncated: finish_reason=length",
        )

        avgs = db.get_dimension_averages("run-001", "model-a")
        assert avgs["coding"] == 0.0
        assert abs(avgs["research"] - 0.8) < 0.01


def test_dimension_absent_from_run_is_not_reported():
    """A --dimension filtered run must not gain phantom zero dimensions."""
    with _make_db() as db:
        db.create_run("run-002", ["model-a"])
        db.register_tasks("run-002", ["model-a"], [("t1", "research", "easy")])
        db.save_task_result(
            run_id="run-002",
            model_name="model-a",
            task_id="t1",
            final_response="r",
            total_turns=1,
            total_tool_calls=0,
            total_latency_ms=100.0,
            reached_max_turns=False,
            weighted_score=0.8,
        )
        avgs = db.get_dimension_averages("run-002", "model-a")
        assert set(avgs) == {"research"}


def test_failed_tasks_count_as_zero_in_their_dimension():
    """A failed task must drag its dimension down, not be excluded from it.

    Filtering the average to completed tasks meant a model was scored only on
    the tasks it managed to finish, so failing more tasks raised the average of
    what was left. In run-20260831-173704 ornith's coding read 0.60 off the one
    coding task of four that completed.
    """
    with _make_db() as db:
        db.create_run("run-003", ["model-a"])
        db.register_tasks(
            "run-003",
            ["model-a"],
            [("t1", "coding", "easy"), ("t2", "coding", "hard"), ("t3", "coding", "medium")],
        )
        db.save_task_result(
            run_id="run-003",
            model_name="model-a",
            task_id="t1",
            final_response="r",
            total_turns=1,
            total_tool_calls=0,
            total_latency_ms=100.0,
            reached_max_turns=False,
            weighted_score=0.9,
        )
        for tid in ("t2", "t3"):
            db.save_task_result(
                run_id="run-003",
                model_name="model-a",
                task_id=tid,
                final_response="",
                total_turns=1,
                total_tool_calls=0,
                total_latency_ms=100.0,
                reached_max_turns=False,
                weighted_score=0.0,
                error="truncated: finish_reason=length",
            )

        avgs = db.get_dimension_averages("run-003", "model-a")
        assert abs(avgs["coding"] - 0.3) < 0.01  # 0.9 / 3, not 0.9 / 1


def test_pending_tasks_do_not_drag_a_dimension_down():
    """An interrupted run must not score unrun tasks as failures."""
    with _make_db() as db:
        db.create_run("run-004", ["model-a"])
        db.register_tasks(
            "run-004",
            ["model-a"],
            [("t1", "coding", "easy"), ("t2", "coding", "hard")],
        )
        db.save_task_result(
            run_id="run-004",
            model_name="model-a",
            task_id="t1",
            final_response="r",
            total_turns=1,
            total_tool_calls=0,
            total_latency_ms=100.0,
            reached_max_turns=False,
            weighted_score=0.9,
        )
        avgs = db.get_dimension_averages("run-004", "model-a")
        assert abs(avgs["coding"] - 0.9) < 0.01
