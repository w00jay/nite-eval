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
