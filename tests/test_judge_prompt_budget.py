"""The judge prompt must fit the judge's context, whatever the model wrote.

The judges run at `--ctx-size 4096` and cannot be raised: both share the 3060,
which has under 1GB of headroom left (CLAUDE.md, "Current known risk"). With
1024 tokens reserved for the verdict the prompt has roughly 3000 tokens.

Before the budget existed the blocks were capped independently at 6000 chars
each, so a coding task could assemble task + code + response well past that.
`coding_mcp_hard_01` wrote 67181 characters of file content in one run, which
is exactly the case that would have overflowed.
"""

from nite_eval.judge import MAX_PROMPT_CHARS, JudgeClient


def _judge():
    return JudgeClient(base_url="http://localhost:0", model="test-judge")


def test_enormous_code_still_fits_the_context():
    prompt = _judge()._build_prompt(
        "code_quality", "rubric", "task " * 200, "summary " * 500, code_evidence="x" * 67_181
    )
    assert len(prompt) <= MAX_PROMPT_CHARS


def test_enormous_everything_still_fits():
    prompt = _judge()._build_prompt(
        "data_accuracy",
        "rubric " * 50,
        "task " * 500,
        "response " * 2000,
        evidence="e" * 40_000,
        code_evidence="c" * 40_000,
    )
    assert len(prompt) <= MAX_PROMPT_CHARS


def test_code_is_preserved_in_preference_to_the_prose_summary():
    """The summary is what misled the judge; the artifact is the substance."""
    prompt = _judge()._build_prompt("code_quality", "rubric", "task", "PROSE" * 4000, code_evidence="REALCODE" * 400)
    assert "REALCODE" in prompt
    assert len(prompt) <= MAX_PROMPT_CHARS


def test_small_prompts_are_left_alone():
    prompt = _judge()._build_prompt("code_quality", "rubric", "task", "short answer", code_evidence="x = 1")
    assert "short answer" in prompt
    assert "x = 1" in prompt
    assert "truncated" not in prompt.lower()
