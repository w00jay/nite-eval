"""Tests for the Hermes AST comparator."""

from nite_eval.ast_comparator import (
    GoldCall,
    compare_calls,
)
from nite_eval.hermes_parser import extract_tool_calls


def _make_parsed(response: str):
    return extract_tool_calls(response)


# --- Single call tests ---


def test_perfect_single_call():
    parsed = _make_parsed('<tool_call>{"name": "web_search", "arguments": {"query": "MCP gateway"}}</tool_call>')
    gold = [GoldCall(name="web_search", arguments={"query": "MCP gateway"})]
    result = compare_calls(parsed, gold)
    assert result.name_accuracy == 1.0
    assert result.arg_name_accuracy == 1.0
    assert result.arg_value_accuracy == 1.0
    assert result.overall_score > 0.95


def test_wrong_function_name():
    parsed = _make_parsed('<tool_call>{"name": "fetch_url", "arguments": {"url": "http://example.com"}}</tool_call>')
    gold = [GoldCall(name="web_search", arguments={"query": "test"})]
    result = compare_calls(parsed, gold)
    assert result.name_accuracy == 0.0
    assert len(result.missing_calls) == 1
    assert len(result.extra_calls) == 1


def test_missing_required_arg():
    parsed = _make_parsed('<tool_call>{"name": "dns_lookup", "arguments": {}}</tool_call>')
    gold = [GoldCall(name="dns_lookup", arguments={"domain": "example.com"})]
    result = compare_calls(parsed, gold)
    assert result.name_accuracy == 1.0
    assert result.arg_name_accuracy == 0.0
    assert result.arg_value_accuracy == 0.0


def test_wrong_arg_value():
    parsed = _make_parsed('<tool_call>{"name": "dns_lookup", "arguments": {"domain": "wrong.com"}}</tool_call>')
    gold = [GoldCall(name="dns_lookup", arguments={"domain": "example.com"})]
    result = compare_calls(parsed, gold)
    assert result.name_accuracy == 1.0
    assert result.arg_name_accuracy == 1.0
    assert result.arg_value_accuracy == 0.0


def test_case_insensitive_string_match():
    parsed = _make_parsed('<tool_call>{"name": "web_search", "arguments": {"query": "MCP Gateway"}}</tool_call>')
    gold = [GoldCall(name="web_search", arguments={"query": "mcp gateway"})]
    result = compare_calls(parsed, gold)
    assert result.arg_value_accuracy == 1.0


def test_extra_args_tolerated():
    parsed = _make_parsed('<tool_call>{"name": "web_search", "arguments": {"query": "test", "limit": 5}}</tool_call>')
    gold = [GoldCall(name="web_search", arguments={"query": "test"})]
    result = compare_calls(parsed, gold)
    assert result.name_accuracy == 1.0
    assert result.arg_value_accuracy == 1.0
    assert result.call_comparisons[0].extra_args == ["limit"]


# --- Fuzzy matching ---


def test_fuzzy_arg_match():
    parsed = _make_parsed(
        '<tool_call>{"name": "web_search", "arguments": {"query": "docker mcp-gateway features and setup"}}</tool_call>'
    )
    gold = [
        GoldCall(
            name="web_search",
            arguments={"query": "docker mcp-gateway"},
            fuzzy_args=["query"],
        )
    ]
    result = compare_calls(parsed, gold)
    assert result.arg_value_accuracy == 1.0


def test_fuzzy_match_fails_when_missing_terms():
    parsed = _make_parsed(
        '<tool_call>{"name": "web_search", "arguments": {"query": "completely unrelated search"}}</tool_call>'
    )
    gold = [
        GoldCall(
            name="web_search",
            arguments={"query": "docker mcp-gateway"},
            fuzzy_args=["query"],
        )
    ]
    result = compare_calls(parsed, gold)
    assert result.arg_value_accuracy == 0.0


# --- Multi-call ordered ---


def test_ordered_sequence_perfect():
    parsed = _make_parsed(
        '<tool_call>{"name": "dns_lookup", "arguments": {"domain": "example.com"}}</tool_call>\n'
        '<tool_call>{"name": "dns_lookup", "arguments": {"domain": "example.com"}}</tool_call>\n'
        '<tool_call>{"name": "port_check", "arguments": {"ip": "93.184.216.34", "port": 443}}</tool_call>'
    )
    gold = [
        GoldCall(name="dns_lookup", arguments={"domain": "example.com"}),
        GoldCall(name="dns_lookup", arguments={"domain": "example.com"}),
        GoldCall(name="port_check", arguments={"ip": "93.184.216.34", "port": 443}),
    ]
    result = compare_calls(parsed, gold, ordered=True)
    assert result.sequence_score == 1.0
    assert result.count_score == 1.0
    assert result.overall_score > 0.95


def test_ordered_missing_middle_call():
    parsed = _make_parsed(
        '<tool_call>{"name": "dns_lookup", "arguments": {"domain": "example.com"}}</tool_call>\n'
        '<tool_call>{"name": "port_check", "arguments": {"ip": "93.184.216.34", "port": 443}}</tool_call>'
    )
    gold = [
        GoldCall(name="dns_lookup", arguments={"domain": "example.com"}),
        GoldCall(name="dns_lookup", arguments={"domain": "example.com"}),  # Missing retry
        GoldCall(name="port_check", arguments={"ip": "93.184.216.34", "port": 443}),
    ]
    result = compare_calls(parsed, gold, ordered=True)
    assert result.sequence_score < 1.0
    # Greedy forward: dns_lookup[0] matches, dns_lookup[1] not found, port_check pointer exhausted
    assert len(result.missing_calls) == 2


def test_ordered_extra_call():
    parsed = _make_parsed(
        '<tool_call>{"name": "dns_lookup", "arguments": {"domain": "example.com"}}</tool_call>\n'
        '<tool_call>{"name": "send_email", "arguments": '
        '{"to": "a@b.com", "subject": "hi", "body": "hello"}}</tool_call>\n'
        '<tool_call>{"name": "port_check", "arguments": {"ip": "93.184.216.34", "port": 443}}</tool_call>'
    )
    gold = [
        GoldCall(name="dns_lookup", arguments={"domain": "example.com"}),
        GoldCall(name="port_check", arguments={"ip": "93.184.216.34", "port": 443}),
    ]
    result = compare_calls(parsed, gold, ordered=True)
    assert result.sequence_score == 1.0  # Both expected calls found in order
    assert len(result.extra_calls) == 1
    assert result.extra_calls[0].name == "send_email"
    assert result.count_score < 1.0  # Penalized for extra call


# --- Multi-call unordered ---


def test_unordered_any_order():
    parsed = _make_parsed(
        '<tool_call>{"name": "port_check", "arguments": {"ip": "1.2.3.4", "port": 80}}</tool_call>\n'
        '<tool_call>{"name": "dns_lookup", "arguments": {"domain": "example.com"}}</tool_call>'
    )
    gold = [
        GoldCall(name="dns_lookup", arguments={"domain": "example.com"}),
        GoldCall(name="port_check", arguments={"ip": "1.2.3.4", "port": 80}),
    ]
    result = compare_calls(parsed, gold, ordered=False)
    assert result.name_accuracy == 1.0
    assert result.arg_value_accuracy == 1.0


# --- Edge cases ---


def test_no_calls_when_expected():
    parsed = _make_parsed("Here is your answer: the sky is blue.")
    gold = [GoldCall(name="web_search", arguments={"query": "sky color"})]
    result = compare_calls(parsed, gold)
    assert result.name_accuracy == 0.0
    assert result.count_score == 0.0
    assert len(result.missing_calls) == 1


def test_calls_when_none_expected():
    parsed = _make_parsed('<tool_call>{"name": "web_search", "arguments": {"query": "test"}}</tool_call>')
    gold: list[GoldCall] = []
    result = compare_calls(parsed, gold)
    assert result.count_score == 0.0
    assert len(result.extra_calls) == 1


def test_empty_both():
    parsed = _make_parsed("No tools needed here.")
    gold: list[GoldCall] = []
    result = compare_calls(parsed, gold)
    assert result.count_score == 1.0
    assert result.overall_score == 1.0


def test_numeric_arg_tolerance():
    parsed = _make_parsed('<tool_call>{"name": "set_temp", "arguments": {"value": 72.0}}</tool_call>')
    gold = [GoldCall(name="set_temp", arguments={"value": 72})]
    result = compare_calls(parsed, gold)
    assert result.arg_value_accuracy == 1.0


def test_nested_dict_args():
    parsed = _make_parsed(
        '<tool_call>{"name": "query_inventory", "arguments": '
        '{"filters": {"wine_type": "red", "status": "in_cellar"}}}</tool_call>'
    )
    gold = [
        GoldCall(
            name="query_inventory",
            arguments={"filters": {"wine_type": "red", "status": "in_cellar"}},
        )
    ]
    result = compare_calls(parsed, gold)
    assert result.arg_value_accuracy == 1.0


def test_list_args():
    parsed = _make_parsed(
        '<tool_call>{"name": "get_indicators", "arguments": {"indicators": ["rsi", "macd", "bollinger"]}}</tool_call>'
    )
    gold = [
        GoldCall(
            name="get_indicators",
            arguments={"indicators": ["rsi", "macd", "bollinger"]},
        )
    ]
    result = compare_calls(parsed, gold)
    assert result.arg_value_accuracy == 1.0


def test_required_args_subset():
    """When only some args are required, only check those."""
    parsed = _make_parsed('<tool_call>{"name": "search", "arguments": {"query": "test"}}</tool_call>')
    gold = [
        GoldCall(
            name="search",
            arguments={"query": "test", "limit": 10},
            required_args=["query"],  # Only query is required
        )
    ]
    result = compare_calls(parsed, gold)
    assert result.arg_name_accuracy == 1.0
    assert result.arg_value_accuracy == 1.0


def test_overall_score_range():
    """Overall score should always be between 0 and 1."""
    # Perfect
    parsed = _make_parsed('<tool_call>{"name": "test", "arguments": {"a": 1}}</tool_call>')
    gold = [GoldCall(name="test", arguments={"a": 1})]
    result = compare_calls(parsed, gold)
    assert 0.0 <= result.overall_score <= 1.0

    # Terrible
    parsed2 = _make_parsed('<tool_call>{"name": "wrong", "arguments": {"b": 2}}</tool_call>')
    result2 = compare_calls(parsed2, gold)
    assert 0.0 <= result2.overall_score <= 1.0
