"""Tests for Hermes-format tool call parsing and validation."""

from nite_eval.hermes_parser import (
    extract_tool_calls,
    format_tool_definitions,
    format_tool_response,
    validate_tool_calls,
)

SAMPLE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_url",
            "parameters": {
                "type": "object",
                "properties": {"url": {"type": "string"}},
                "required": ["url"],
            },
        },
    },
]


def test_extract_single_tool_call():
    response = '<tool_call>{"name": "web_search", "arguments": {"query": "test"}}</tool_call>'
    parsed = extract_tool_calls(response)
    assert len(parsed.tool_calls) == 1
    assert parsed.tool_calls[0].name == "web_search"
    assert parsed.tool_calls[0].arguments == {"query": "test"}
    assert not parsed.errors


def test_extract_multiple_tool_calls():
    response = (
        '<tool_call>{"name": "web_search", "arguments": {"query": "a"}}</tool_call>\n'
        '<tool_call>{"name": "fetch_url", "arguments": {"url": "http://example.com"}}</tool_call>'
    )
    parsed = extract_tool_calls(response)
    assert len(parsed.tool_calls) == 2
    assert parsed.tool_calls[0].name == "web_search"
    assert parsed.tool_calls[1].name == "fetch_url"


def test_extract_with_scratch_pad():
    response = (
        "<scratch_pad>I need to search for this first.</scratch_pad>\n"
        '<tool_call>{"name": "web_search", "arguments": {"query": "test"}}</tool_call>'
    )
    parsed = extract_tool_calls(response)
    assert parsed.scratch_pad == "I need to search for this first."
    assert len(parsed.tool_calls) == 1


def test_extract_with_surrounding_text():
    response = (
        "Let me look that up for you.\n"
        '<tool_call>{"name": "web_search", "arguments": {"query": "test"}}</tool_call>\n'
        "I'll check the results."
    )
    parsed = extract_tool_calls(response)
    assert len(parsed.tool_calls) == 1
    assert "look that up" in parsed.text
    assert "check the results" in parsed.text


def test_trailing_comma_fix():
    response = '<tool_call>{"name": "web_search", "arguments": {"query": "test",}}</tool_call>'
    parsed = extract_tool_calls(response)
    assert len(parsed.tool_calls) == 1
    assert not parsed.errors


def test_whitespace_variance():
    response = '<tool_call>\n  {"name": "web_search", "arguments": {"query": "test"}}  \n</tool_call>'
    parsed = extract_tool_calls(response)
    assert len(parsed.tool_calls) == 1


def test_args_before_name():
    response = '<tool_call>{"arguments": {"query": "test"}, "name": "web_search"}</tool_call>'
    parsed = extract_tool_calls(response)
    assert len(parsed.tool_calls) == 1
    assert parsed.tool_calls[0].name == "web_search"


def test_empty_arguments():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_status",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    response = '<tool_call>{"name": "get_status", "arguments": {}}</tool_call>'
    parsed = extract_tool_calls(response)
    assert len(parsed.tool_calls) == 1
    errors = validate_tool_calls(parsed, tools)
    assert not errors


def test_malformed_json_error():
    response = "<tool_call>not valid json at all</tool_call>"
    parsed = extract_tool_calls(response)
    assert len(parsed.tool_calls) == 0
    assert len(parsed.errors) == 1
    assert parsed.errors[0]["error"] == "malformed_json"


def test_missing_name_error():
    response = '<tool_call>{"arguments": {"query": "test"}}</tool_call>'
    parsed = extract_tool_calls(response)
    assert len(parsed.tool_calls) == 0
    assert len(parsed.errors) == 1
    assert parsed.errors[0]["error"] == "missing_name"


def test_no_tool_calls():
    response = "Here is your answer: the sky is blue."
    parsed = extract_tool_calls(response)
    assert len(parsed.tool_calls) == 0
    assert not parsed.errors
    assert "sky is blue" in parsed.text


def test_validate_unknown_function():
    response = '<tool_call>{"name": "unknown_tool", "arguments": {}}</tool_call>'
    parsed = extract_tool_calls(response)
    errors = validate_tool_calls(parsed, SAMPLE_TOOLS)
    assert len(errors) == 1
    assert errors[0].error == "unknown_function"


def test_validate_missing_required_param():
    response = '<tool_call>{"name": "web_search", "arguments": {}}</tool_call>'
    parsed = extract_tool_calls(response)
    errors = validate_tool_calls(parsed, SAMPLE_TOOLS)
    assert len(errors) == 1
    assert errors[0].error == "missing_required_params"
    assert "query" in errors[0].details["missing"]


def test_validate_type_mismatch():
    response = '<tool_call>{"name": "web_search", "arguments": {"query": 42}}</tool_call>'
    parsed = extract_tool_calls(response)
    errors = validate_tool_calls(parsed, SAMPLE_TOOLS)
    assert len(errors) == 1
    assert errors[0].error == "type_mismatch"


def test_validate_valid_call():
    response = '<tool_call>{"name": "web_search", "arguments": {"query": "hello"}}</tool_call>'
    parsed = extract_tool_calls(response)
    errors = validate_tool_calls(parsed, SAMPLE_TOOLS)
    assert not errors


def test_format_tool_definitions():
    formatted = format_tool_definitions(SAMPLE_TOOLS)
    assert "<tools>" in formatted
    assert "</tools>" in formatted
    assert "web_search" in formatted
    assert "<tool_call>" in formatted  # Includes format instructions
    assert "function calling" in formatted.lower()


def test_format_tool_response():
    formatted = format_tool_response("web_search", {"results": ["a", "b"]})
    assert "<tool_response>" in formatted
    assert "web_search" in formatted


# --- Gemma/Harmony-format parsing (gemma4-26b-a4b) ---


def test_gemma_simple_string_delim():
    """Gemma variant that uses <|"|> as string delimiter."""
    response = '<|tool_call>call:get_price_data{period:<|"|>1mo<|"|>,symbol:<|"|>NVDA<|"|>}<tool_call|>'
    parsed = extract_tool_calls(response)
    assert len(parsed.tool_calls) == 1
    assert parsed.tool_calls[0].name == "get_price_data"
    assert parsed.tool_calls[0].arguments == {"period": "1mo", "symbol": "NVDA"}


def test_gemma_json_string_delim():
    """Gemma variant that uses standard " for strings."""
    response = '<|tool_call>call:run_code{command: "ls -R"}<tool_call|>'
    parsed = extract_tool_calls(response)
    assert len(parsed.tool_calls) == 1
    assert parsed.tool_calls[0].name == "run_code"
    assert parsed.tool_calls[0].arguments == {"command": "ls -R"}


def test_gemma_array_argument():
    """Array with Gemma string delims."""
    response = (
        "<|tool_call>call:get_technical_indicators"
        '{indicators:[<|"|>rsi<|"|>,<|"|>macd<|"|>,<|"|>bollinger<|"|>],'
        'symbol:<|"|>NVDA<|"|>}<tool_call|>'
    )
    parsed = extract_tool_calls(response)
    assert len(parsed.tool_calls) == 1
    tc = parsed.tool_calls[0]
    assert tc.name == "get_technical_indicators"
    assert tc.arguments["symbol"] == "NVDA"
    assert tc.arguments["indicators"] == ["rsi", "macd", "bollinger"]


def test_gemma_nested_object():
    response = (
        "<|tool_call>call:call_mcp_tool"
        '{arguments:{server:<|"|>notion<|"|>,tool:<|"|>search<|"|>},'
        'server:<|"|>notion<|"|>}<tool_call|>'
    )
    parsed = extract_tool_calls(response)
    assert len(parsed.tool_calls) == 1
    tc = parsed.tool_calls[0]
    assert tc.name == "call_mcp_tool"
    assert tc.arguments["server"] == "notion"
    assert tc.arguments["arguments"] == {"server": "notion", "tool": "search"}


def test_gemma_multiple_calls():
    response = (
        '<|tool_call>call:web_search{query:<|"|>A<|"|>}<tool_call|>'
        "<|tool_response>\n<|channel>thought\nthinking<channel|>"
        '<|tool_call>call:web_search{query:<|"|>B<|"|>}<tool_call|>'
    )
    parsed = extract_tool_calls(response)
    assert len(parsed.tool_calls) == 2
    assert parsed.tool_calls[0].arguments == {"query": "A"}
    assert parsed.tool_calls[1].arguments == {"query": "B"}
    # Channel block captured as scratch-pad fallback
    assert parsed.scratch_pad == "thought\nthinking"


def test_gemma_hermes_priority():
    """When both formats present, Hermes wins (Gemma is fallback only)."""
    response = (
        '<tool_call>{"name": "web_search", "arguments": {"query": "hermes"}}</tool_call>'
        '<|tool_call>call:web_search{query:<|"|>gemma<|"|>}<tool_call|>'
    )
    parsed = extract_tool_calls(response)
    assert len(parsed.tool_calls) == 1
    assert parsed.tool_calls[0].arguments == {"query": "hermes"}
