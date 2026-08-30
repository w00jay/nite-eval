"""Tests for deterministic scoring methods."""

from nite_eval.scoring import (
    TaskScore,
    compute_composite,
    compute_dimension_score,
    score_checklist,
    score_contains_check,
    score_distractor_avoidance,
    score_sequence_match,
    score_subset_match,
)


def test_sequence_match_perfect():
    actual = [
        {"name": "dns_lookup", "arguments": {"domain": "example.com"}},
        {"name": "dns_lookup", "arguments": {"domain": "example.com"}},
        {"name": "port_check", "arguments": {"ip": "93.184.216.34", "port": 443}},
    ]
    expected = [
        {"name": "dns_lookup", "args": {"domain": "example.com"}},
        {"name": "dns_lookup", "args": {"domain": "example.com"}},
        {"name": "port_check", "args": {"ip": "93.184.216.34", "port": 443}},
    ]
    assert score_sequence_match(actual, expected) == 1.0


def test_sequence_match_partial():
    actual = [
        {"name": "dns_lookup", "arguments": {"domain": "example.com"}},
        {"name": "port_check", "arguments": {"ip": "93.184.216.34", "port": 443}},
    ]
    expected = [
        {"name": "dns_lookup", "args": {"domain": "example.com"}},
        {"name": "dns_lookup", "args": {"domain": "example.com"}},  # Missing retry
        {"name": "port_check", "args": {"ip": "93.184.216.34", "port": 443}},
    ]
    # Only 1 of 3 expected matched (dns_lookup matches, then port_check can't match 2nd dns_lookup)
    score = score_sequence_match(actual, expected)
    assert abs(score - 1 / 3) < 0.01


def test_sequence_match_wrong_order():
    actual = [
        {"name": "port_check", "arguments": {"ip": "1.2.3.4", "port": 80}},
        {"name": "dns_lookup", "arguments": {"domain": "example.com"}},
    ]
    expected = [
        {"name": "dns_lookup", "args": {"domain": "example.com"}},
        {"name": "port_check", "args": {"ip": "1.2.3.4", "port": 80}},
    ]
    # Only port_check can match after dns_lookup is missed
    score = score_sequence_match(actual, expected)
    assert score < 1.0


def test_subset_match_all_present():
    actual = [
        {"name": "get_price"},
        {"name": "get_news"},
        {"name": "get_macro"},
    ]
    expected = ["get_price", "get_news", "get_macro"]
    assert score_subset_match(actual, expected) == 1.0


def test_subset_match_partial():
    actual = [{"name": "get_price"}]
    expected = ["get_price", "get_news", "get_macro"]
    score = score_subset_match(actual, expected)
    assert abs(score - 1 / 3) < 0.01


def test_checklist_all_found():
    response = "The Docker gateway is Go-based and supports SSE transport. It has bearer auth."
    criteria = ["Go-based", "transport support", "auth"]
    assert score_checklist(response, criteria) == 1.0


def test_checklist_partial():
    response = "The Docker gateway is written in Golang with fast startup."
    criteria = ["Golang implementation", "transport support", "authentication approach"]
    score = score_checklist(response, criteria)
    assert 0.0 < score < 1.0


def test_contains_check():
    response = "Port 443 is open on 93.184.216.34"
    assert score_contains_check(response, ["443", "93.184.216.34"]) == 1.0
    assert score_contains_check(response, ["443", "1.2.3.4"]) == 0.5


def test_distractor_avoidance_clean():
    actual = [{"name": "dns_lookup"}, {"name": "port_check"}]
    assert score_distractor_avoidance(actual, ["send_email"]) == 1.0


def test_distractor_avoidance_fail():
    actual = [{"name": "dns_lookup"}, {"name": "send_email"}]
    assert score_distractor_avoidance(actual, ["send_email"]) == 0.0


def test_compute_dimension_score():
    tasks = [
        TaskScore(task_id="t1", dimension="agentic", scores=[], weighted_total=0.8),
        TaskScore(task_id="t2", dimension="agentic", scores=[], weighted_total=0.6),
        TaskScore(task_id="t3", dimension="coding", scores=[], weighted_total=0.9),
    ]
    assert compute_dimension_score(tasks, "agentic") == 0.7
    assert compute_dimension_score(tasks, "coding") == 0.9
    assert compute_dimension_score(tasks, "research") == 0.0


def test_composite_equal_weights():
    dims = {"research": 0.8, "planning": 0.6, "coding": 0.7, "agentic": 0.9}
    composite = compute_composite(dims)
    assert abs(composite - 0.75) < 0.01


def test_composite_custom_weights():
    dims = {"research": 0.8, "coding": 0.4}
    weights = {"research": 0.3, "coding": 0.7}
    composite = compute_composite(dims, weights)
    expected = (0.8 * 0.3 + 0.4 * 0.7) / 1.0
    assert abs(composite - expected) < 0.01


# --- Wave 2: scorers that replaced the free-1.0 deterministic fallback ---


def test_tool_args_match_scores_partial_credit():
    from nite_eval.scoring import score_tool_args_match

    calls = [
        {"name": "capture_thought", "arguments": {"content": "autoresearch and latency", "type": "idea"}},
    ]
    expected = [
        {
            "name": "capture_thought",
            "args_must_contain": {"content": ["autoresearch", "mcp gateway", "latency"], "type": "idea"},
        }
    ]
    # 3 of 4 requirements met (content: autoresearch + latency, type: idea; missing "mcp gateway")
    assert score_tool_args_match(calls, expected) == 0.75


def test_tool_args_match_zero_when_tool_never_called():
    from nite_eval.scoring import score_tool_args_match

    expected = [{"name": "capture_thought", "args_must_contain": {"content": ["x"]}}]
    assert score_tool_args_match([], expected) == 0.0


def test_tool_absence_penalises_calling_a_distractor():
    from nite_eval.scoring import score_distractor_avoidance

    clean = [{"name": "search_thoughts", "arguments": {}}]
    dirty = [{"name": "send_email", "arguments": {}}]
    assert score_distractor_avoidance(clean, ["send_email"]) == 1.0
    assert score_distractor_avoidance(dirty, ["send_email"]) == 0.0


def test_tool_ordering_requires_after_to_follow_before():
    from nite_eval.scoring import score_tool_ordering

    ordering = [["refresh_credentials", "call_mcp_tool"]]
    good = [
        {"name": "call_mcp_tool", "arguments": {}},
        {"name": "refresh_credentials", "arguments": {}},
        {"name": "call_mcp_tool", "arguments": {}},
    ]
    # Called the tool, refreshed, but never retried afterwards.
    bad = [
        {"name": "call_mcp_tool", "arguments": {}},
        {"name": "refresh_credentials", "arguments": {}},
    ]
    assert score_tool_ordering(good, ordering) == 1.0
    assert score_tool_ordering(bad, ordering) == 0.0


def test_judge_prompt_includes_tool_results_when_evidence_given():
    """Fact-checking criteria need the tool results as ground truth."""
    from nite_eval.judge import JudgeClient

    client = JudgeClient.__new__(JudgeClient)
    client.MAX_RESPONSE_CHARS = 10000
    prompt = JudgeClient._build_prompt(
        client,
        dimension="no_hallucination",
        rubric="check facts",
        task_description="analyse NVDA",
        model_response="RSI is 62",
        evidence='get_technical_indicators({}) -> {"rsi": 62}',
    )
    assert "Tool Results (ground truth)" in prompt
    assert '"rsi": 62' in prompt


def test_judge_prompt_omits_evidence_section_when_not_supplied():
    from nite_eval.judge import JudgeClient

    client = JudgeClient.__new__(JudgeClient)
    client.MAX_RESPONSE_CHARS = 10000
    prompt = JudgeClient._build_prompt(
        client,
        dimension="code_quality",
        rubric="rate it",
        task_description="write a parser",
        model_response="here is code",
    )
    assert "Tool Results" not in prompt


# --- Wave 2: checklist scoring by judge instead of substring matching ---


class _StubJudge:
    def __init__(self, met):
        self._met = met
        self.calls = []

    def evaluate_checklist(self, criteria, task_description, model_response):
        self.calls.append(criteria)
        return self._met


def test_checklist_uses_judge_verdicts():
    from nite_eval.scoring import score_checklist_with_judge

    judge = _StubJudge([True, False, True, False])
    score, details = score_checklist_with_judge(judge, ["a", "b", "c", "d"], "task", "response")
    assert score == 0.5
    assert details["unmet"] == ["b", "d"]
    # One call for the whole checklist, not one per criterion.
    assert len(judge.calls) == 1


def test_checklist_judge_failure_is_reported_not_silently_downgraded():
    from nite_eval.judge import JudgeError
    from nite_eval.scoring import score_checklist_with_judge

    judge = _StubJudge(JudgeError(error="boom", raw_response=""))
    score, details = score_checklist_with_judge(judge, ["a"], "task", "response")
    assert score == 0.0
    assert details["error"] == "boom"


def test_old_substring_checklist_would_have_passed_on_a_keyword():
    """Documents the behaviour that was replaced.

    `score_checklist` counted a criterion as met if any word longer than three
    characters appeared anywhere in the response.
    """
    from nite_eval.scoring import score_checklist

    assert score_checklist("our strategy is unrelated", ["Addresses embedding strategy"]) == 1.0
