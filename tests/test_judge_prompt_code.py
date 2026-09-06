"""The judge prompt must show the code and say what to do when there is none.

Two defects, both found in run-20260905-235130:

1. The prompt had no `## Code` section at all, so coding criteria were scored
   from the model's prose summary.
2. The prompt had no anchor for an absent implementation. Given a detailed spec
   under `## Task` and a 22-character response, reward-anything narrated the
   spec back as though it were the model's work and scored it 4/5, three times,
   at confidence 1.0.
"""

from nite_eval.judge import JudgeClient

RUBRIC = "Is the code well structured?"
TASK = "Write /app/scan.ts exporting createHandler(deps). Return 200 on Claude failure."


def _judge():
    return JudgeClient(base_url="http://localhost:0", model="test-judge")


def test_code_appears_in_the_prompt_when_the_model_wrote_some():
    prompt = _judge()._build_prompt(
        "code_quality", RUBRIC, TASK, "I built it.", code_evidence="### /app/scan.ts (12 chars)\nexport const q=1"
    )
    assert "export const q=1" in prompt
    assert "/app/scan.ts" in prompt


def test_no_code_section_when_the_model_wrote_nothing():
    prompt = _judge()._build_prompt("code_quality", RUBRIC, TASK, "/app/scan.ts", code_evidence="")
    assert "## Code" not in prompt


def test_absent_work_is_explicitly_scored_one():
    """Without this anchor the judge scores the spec instead of the response."""
    prompt = _judge()._build_prompt("code_quality", RUBRIC, TASK, "/app/scan.ts", code_evidence="")
    low = prompt.lower()
    assert "score 1" in low
    # it must be told not to credit the task description
    assert "task description" in low or "specification" in low


def test_task_and_code_blocks_are_distinguishable():
    """The judge must not confuse the spec with the model's work."""
    prompt = _judge()._build_prompt(
        "code_quality", RUBRIC, TASK, "done", code_evidence="### /app/scan.ts (5 chars)\nhello"
    )
    assert prompt.index("## Task") < prompt.index("hello")
    assert "## Response to Evaluate" in prompt


def test_tool_evidence_and_code_evidence_coexist():
    prompt = _judge()._build_prompt(
        "no_hallucination",
        RUBRIC,
        TASK,
        "done",
        evidence="get_price({}) -> {'price': 123}",
        code_evidence="### /app/a.py (5 chars)\nx = 1",
    )
    assert "get_price" in prompt
    assert "x = 1" in prompt
