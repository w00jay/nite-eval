"""Tests for the deterministic mock tool environment."""

from nite_eval.mock_tools import MockToolEnv


def _sample_mock_responses() -> dict:
    return {
        "web_search": [
            {
                "match": {"query_contains": "docker mcp"},
                "response": {"content": {"results": [{"title": "Docker MCP Gateway"}]}},
            },
            {
                "match": {"query_contains": "microsoft mcp"},
                "response": {"content": {"results": [{"title": "Microsoft MCP Gateway"}]}},
            },
        ],
        "dns_lookup": [
            {
                "match": {"domain": "example.com"},
                "sequence": [
                    {"error": "DNS timeout, please retry"},
                    {"content": {"ip": "93.184.216.34", "ttl": 3600}},
                ],
            },
        ],
        "summarize": [
            {
                "match": {"topic_contains": "any"},
                "response": {"content": {"status": "saved"}},
            },
        ],
    }


def test_exact_match():
    env = MockToolEnv.from_task_yaml(_sample_mock_responses())
    result = env.call("dns_lookup", {"domain": "example.com"})
    # First call in sequence returns error
    assert "error" in result


def test_contains_match():
    env = MockToolEnv.from_task_yaml(_sample_mock_responses())
    result = env.call("web_search", {"query": "docker mcp-gateway features"})
    assert "content" in result
    assert result["content"]["results"][0]["title"] == "Docker MCP Gateway"


def test_contains_match_case_insensitive():
    env = MockToolEnv.from_task_yaml(_sample_mock_responses())
    result = env.call("web_search", {"query": "DOCKER MCP stuff"})
    assert "content" in result


def test_sequence_responses():
    env = MockToolEnv.from_task_yaml(_sample_mock_responses())

    # First call: error
    r1 = env.call("dns_lookup", {"domain": "example.com"})
    assert "error" in r1
    assert "timeout" in r1["error"]

    # Second call: success
    r2 = env.call("dns_lookup", {"domain": "example.com"})
    assert "content" in r2
    assert r2["content"]["ip"] == "93.184.216.34"

    # Third call: repeats last in sequence
    r3 = env.call("dns_lookup", {"domain": "example.com"})
    assert "content" in r3


def test_any_match():
    env = MockToolEnv.from_task_yaml(_sample_mock_responses())
    result = env.call("summarize", {"topic": "anything at all"})
    assert result["content"]["status"] == "saved"


def test_no_match():
    env = MockToolEnv.from_task_yaml(_sample_mock_responses())
    result = env.call("web_search", {"query": "completely unrelated"})
    assert "error" in result


def test_unknown_tool():
    env = MockToolEnv.from_task_yaml(_sample_mock_responses())
    result = env.call("nonexistent_tool", {})
    assert "error" in result


def test_call_log():
    env = MockToolEnv.from_task_yaml(_sample_mock_responses())
    env.call("web_search", {"query": "docker mcp"})
    env.call("dns_lookup", {"domain": "example.com"})

    log = env.get_call_log()
    assert len(log) == 2
    assert log[0]["name"] == "web_search"
    assert log[1]["name"] == "dns_lookup"
    assert log[0]["call_number"] == 1
    assert log[1]["call_number"] == 1  # First call to dns_lookup


def test_reset():
    env = MockToolEnv.from_task_yaml(_sample_mock_responses())
    env.call("web_search", {"query": "docker mcp"})
    assert len(env.get_call_log()) == 1

    env.reset()
    assert len(env.get_call_log()) == 0
    assert env.call_counts == {}
