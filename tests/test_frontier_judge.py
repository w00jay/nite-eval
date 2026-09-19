"""Frontier judge routing and API judge behavior (no network)."""

import pytest

from nite_eval.conversation_runner import ModelReply
from nite_eval.cost import CostTracker
from nite_eval.judge import ApiJudgeClient, JudgeClient, JudgeError, JudgeResult, build_judge

JUDGE_CFG = {
    "base_url": "http://127.0.0.1:9091/v1",
    "flow_judge_model": "flow-judge",
    "reward_anything_model": "reward-anything",
}


class ScriptedBackend:
    provider = "anthropic"
    model_id = "claude-opus-5"

    def __init__(self, replies):
        self._replies = list(replies)
        self.prompts: list[str] = []

    def generate(self, messages, max_tokens, tools=None, native_tools=False):  # noqa: ARG002
        self.prompts.append(messages[-1].content)
        return self._replies.pop(0)

    def close(self):
        pass


def test_api_judge_parses_score_and_bills_the_run_budget():
    tracker = CostTracker()
    backend = ScriptedBackend(
        [ModelReply(text='{"reasoning": "Solid but shallow.", "score": 3}', prompt_tokens=2000, completion_tokens=50)]
    )
    result = ApiJudgeClient(backend, tracker=tracker).evaluate("research", "rubric", "task", "answer")

    assert isinstance(result, JudgeResult)
    assert result.score == 3
    assert result.reasoning == "Solid but shallow."
    # Judge spend draws on the same budget as the models under test
    assert tracker.spent_usd == pytest.approx(2000 * 5 / 1e6 + 50 * 25 / 1e6)


def test_api_judge_reuses_the_local_rubric_prompt():
    backend = ScriptedBackend([ModelReply(text='{"score": 5}')])
    ApiJudgeClient(backend).evaluate("research", "RUBRIC-MARKER", "TASK-MARKER", "ANSWER-MARKER")
    prompt = backend.prompts[0]
    assert "RUBRIC-MARKER" in prompt and "TASK-MARKER" in prompt and "ANSWER-MARKER" in prompt


def test_api_judge_retries_empty_then_reports_error():
    backend = ScriptedBackend([ModelReply(text="  "), ModelReply(text=""), ModelReply(text="")])
    result = ApiJudgeClient(backend).evaluate("research", "r", "t", "m")
    assert isinstance(result, JudgeError)
    assert result.error == "empty_response"


def test_api_judge_keeps_more_evidence_than_the_local_judge():
    assert ApiJudgeClient.MAX_RESPONSE_CHARS > JudgeClient.MAX_RESPONSE_CHARS


def test_build_judge_without_frontier_stays_local():
    judge = build_judge(JUDGE_CFG)
    assert judge._select("research").model == "reward-anything"
    assert judge._select("reasoning_quality").model == "flow-judge"
    judge.close()


def test_build_judge_routes_named_dimensions_to_frontier(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    cfg = {
        **JUDGE_CFG,
        "frontier": {"provider": "anthropic", "api_model": "claude-opus-5", "dimensions": ["reasoning_quality"]},
    }
    judge = build_judge(cfg, CostTracker())
    assert judge._select("reasoning_quality").model == "claude-opus-5"
    assert judge._select("research").model == "reward-anything"  # everything else stays local
    judge.close()


def test_build_judge_wildcard_routes_every_dimension(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    cfg = {**JUDGE_CFG, "frontier": {"provider": "anthropic", "api_model": "claude-opus-5", "dimensions": ["*"]}}
    judge = build_judge(cfg)
    assert judge._select("research").model == "claude-opus-5"
    assert judge._select("practical_output").model == "claude-opus-5"
    judge.close()


def test_build_judge_rejects_a_local_frontier_block():
    cfg = {**JUDGE_CFG, "frontier": {"provider": "local", "api_model": "reward-anything"}}
    with pytest.raises(ValueError, match="non-local provider"):
        build_judge(cfg)
