"""The report must say which dimensions carry a wider noise floor than the composite.

Coding tasks run against a real container, and the container's output enters the
conversation. An `ls -la` on turn 1 returns the directory's mtime, which is the
container's creation time, so two runs of the same model at temperature 0 see
different history from turn 2 onward and go on to write different code:
ornith-1.5 emitted 65986, 66666 and 65604 bytes of tool-call arguments across
three runs of one task. In one of them the code it happened to write hit a real
bug and both automated criteria scored 0 instead of 1.00 and 0.93.

So coding gaps are not comparable to the composite's 0.05 threshold, which was
calibrated for judge variance on mock-backed tasks.
"""

import tempfile

from nite_eval.report import generate_report
from nite_eval.results_db import ResultsDB


def _db_with_two_models() -> ResultsDB:
    db = ResultsDB(tempfile.mktemp(suffix=".db"))
    db.create_run("run-001", ["model-a", "model-b"])
    db.register_tasks(
        "run-001",
        ["model-a", "model-b"],
        [("t1", "coding", "easy"), ("t2", "research", "easy")],
    )
    for model, coding, research in (("model-a", 0.60, 0.80), ("model-b", 0.50, 0.79)):
        for task, score in (("t1", coding), ("t2", research)):
            db.save_task_result(
                run_id="run-001",
                model_name=model,
                task_id=task,
                final_response="r",
                total_turns=1,
                total_tool_calls=0,
                total_latency_ms=100.0,
                reached_max_turns=False,
                weighted_score=score,
            )
    return db


def test_wider_dimension_threshold_is_reported():
    with _db_with_two_models() as db:
        report = generate_report(
            db,
            "run-001",
            {"coding": 0.5, "research": 0.5},
            mdd=0.05,
            dimension_mdd={"coding": 0.15},
        )
    assert "0.15" in report
    assert "coding" in report.lower()


def test_no_extra_note_without_dimension_thresholds():
    """A run that declares none must read exactly as it did before."""
    with _db_with_two_models() as db:
        report = generate_report(db, "run-001", {"coding": 0.5, "research": 0.5}, mdd=0.05)
    assert "0.15" not in report


# --- tasks that never answered ---


def test_no_answer_tasks_are_surfaced_separately():
    """A task that never terminated scores near zero for the wrong reason.

    Qwopus3.8-27B-Flash lost most of a research dimension to one such task
    while looking merely mediocre in the summary table, and a dimension mean
    cannot distinguish "answered badly" from "never answered".
    """
    import tempfile

    from nite_eval.conversation_runner import NO_ANSWER_PREFIX
    from nite_eval.report import generate_report
    from nite_eval.results_db import ResultsDB

    db = ResultsDB(tempfile.mktemp(suffix=".db"))
    db.create_run("run-001", ["looper"])
    tasks = [("research_a_easy_01", "research", "easy"), ("research_b_easy_01", "research", "easy")]
    db.register_tasks("run-001", ["looper"], tasks)
    for task_id, final in (
        ("research_a_easy_01", f"{NO_ANSWER_PREFIX} every one of 5 turns ended in a tool call ]"),
        ("research_b_easy_01", "a real answer"),
    ):
        db.mark_task_running("run-001", "looper", task_id)
        db.save_task_result(
            run_id="run-001",
            model_name="looper",
            task_id=task_id,
            final_response=final,
            total_turns=5,
            total_tool_calls=14,
            total_latency_ms=1000.0,
            reached_max_turns=True,
            weighted_score=0.0,
        )
    with db:
        report = generate_report(db, "run-001")
    assert "## Tasks That Produced No Answer" in report
    assert "| looper | research_a_easy_01 | research | 5 | 14 | 0.00 |" in report
    # The task that did answer must not appear in this table.
    assert "research_b_easy_01" not in report.split("## Tasks That Produced No Answer")[1].split("\n## ")[0]


def test_no_answer_section_absent_when_every_task_answered():
    import tempfile

    from nite_eval.report import generate_report
    from nite_eval.results_db import ResultsDB

    db = ResultsDB(tempfile.mktemp(suffix=".db"))
    db.create_run("run-001", ["m"])
    db.register_tasks("run-001", ["m"], [("research_a_easy_01", "research", "easy")])
    db.mark_task_running("run-001", "m", "research_a_easy_01")
    db.save_task_result(
        run_id="run-001",
        model_name="m",
        task_id="research_a_easy_01",
        final_response="a real answer",
        total_turns=2,
        total_tool_calls=1,
        total_latency_ms=1000.0,
        reached_max_turns=False,
        weighted_score=0.8,
    )
    with db:
        report = generate_report(db, "run-001")
    assert "Tasks That Produced No Answer" not in report
