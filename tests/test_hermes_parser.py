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


def test_gemma_unwraps_hermes_style_wrapping():
    """Gemma sometimes emits call:fn{arguments:{...}} — unwrap the single key."""
    response = '<|tool_call>call:web_search{arguments:{query:<|"|>abc<|"|>}}<tool_call|>'
    parsed = extract_tool_calls(response)
    assert len(parsed.tool_calls) == 1
    assert parsed.tool_calls[0].arguments == {"query": "abc"}


def test_gemma_preserves_arguments_as_real_param():
    """When 'arguments' is one of several keys, it's a real parameter — no unwrap."""
    response = (
        "<|tool_call>call:call_mcp_tool"
        '{arguments:{server:<|"|>notion<|"|>,tool:<|"|>search<|"|>},'
        'server:<|"|>notion<|"|>}<tool_call|>'
    )
    parsed = extract_tool_calls(response)
    assert len(parsed.tool_calls) == 1
    tc = parsed.tool_calls[0]
    assert tc.arguments["server"] == "notion"
    assert tc.arguments["arguments"] == {"server": "notion", "tool": "search"}


def test_gemma_hermes_priority():
    """When both formats present, Hermes wins (Gemma is fallback only)."""
    response = (
        '<tool_call>{"name": "web_search", "arguments": {"query": "hermes"}}</tool_call>'
        '<|tool_call>call:web_search{query:<|"|>gemma<|"|>}<tool_call|>'
    )
    parsed = extract_tool_calls(response)
    assert len(parsed.tool_calls) == 1
    assert parsed.tool_calls[0].arguments == {"query": "hermes"}


def test_function_key_alias():
    """qwen3.8 emits {"function": name} instead of the Hermes {"name": ...}."""
    response = (
        '<tool_call>\n{"function": "web_search", "arguments": {"query": "Vivino AI"}}\n</tool_call>\n'
        '<tool_call>\n{"function": "get_price_data", "arguments": {"symbol": "NVDA"}}\n</tool_call>'
    )
    parsed = extract_tool_calls(response)
    assert not parsed.errors
    assert [tc.name for tc in parsed.tool_calls] == ["web_search", "get_price_data"]
    assert parsed.tool_calls[0].arguments == {"query": "Vivino AI"}


def test_function_key_nested_openai_style():
    response = '<tool_call>{"function": {"name": "web_search", "arguments": {"query": "x"}}}</tool_call>'
    parsed = extract_tool_calls(response)
    assert not parsed.errors
    assert parsed.tool_calls[0].name == "web_search"
    assert parsed.tool_calls[0].arguments == {"query": "x"}


def test_name_key_wins_over_function_key():
    response = '<tool_call>{"name": "real", "function": "decoy", "arguments": {}}</tool_call>'
    parsed = extract_tool_calls(response)
    assert parsed.tool_calls[0].name == "real"


def test_dropped_key_quote_is_repaired():
    """qwen3.8 reproducibly drops the opening quote of the key after the name."""
    from nite_eval.hermes_parser import extract_tool_calls

    raw = '<tool_call>\n{"name": "write_file",\narguments": {"path": "/a.go", "content": "package main"}}\n</tool_call>'
    result = extract_tool_calls(raw)

    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "write_file"
    assert result.tool_calls[0].arguments["path"] == "/a.go"
    assert result.repaired == 1
    assert result.errors == []


def test_repair_does_not_corrupt_strings_containing_the_pattern():
    """A `foo":` sequence inside escaped source must be left alone."""
    from nite_eval.hermes_parser import extract_tool_calls

    # The content contains `key":` inside a Go string literal.
    raw = (
        '<tool_call>\n{"name": "write_file", "arguments": '
        '{"path": "/a.go", "content": "s := \\"key\\": value\\nx := 1"}}\n</tool_call>'
    )
    result = extract_tool_calls(raw)

    assert len(result.tool_calls) == 1
    assert result.repaired == 0
    assert '\\"' not in result.tool_calls[0].arguments["content"]
    assert '"key": value' in result.tool_calls[0].arguments["content"]


def test_well_formed_calls_are_not_counted_as_repaired():
    from nite_eval.hermes_parser import extract_tool_calls

    raw = '<tool_call>\n{"name": "search", "arguments": {"query": "test"}}\n</tool_call>'
    result = extract_tool_calls(raw)

    assert len(result.tool_calls) == 1
    assert result.repaired == 0


def test_valid_json_containing_code_is_not_corrupted():
    """Regression: _fix_json appended braces counted from inside string values.

    coding_mcp_hard_01 failed at turn 30 of 32 with malformed_json on a payload
    that json.loads accepts. The tool call carried Go source, so the naive
    `count("{") > count("}")` check saw an imbalance and appended spurious
    closing braces, producing "Extra data".
    """
    from nite_eval.hermes_parser import extract_tool_calls

    go_source = 'func h(w http.ResponseWriter) {\\n\\twriteJSON(w, map[string]any{\\\"error\\\": \\\"x\\\"})\\n'
    raw = (
        '<tool_call>\n{"name": "run_code", "arguments": {"command": "cat > a.go <<EOF\\n'
        + go_source
        + '"}}\n</tool_call>'
    )
    result = extract_tool_calls(raw)

    assert result.errors == []
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "run_code"
    assert result.repaired == 0, "valid JSON must not be counted as repaired"


def test_unbalanced_braces_inside_strings_are_not_counted():
    from nite_eval.hermes_parser import _count_unclosed_braces

    # Three opening braces inside a string, none structural.
    assert _count_unclosed_braces('{"a": "if x { y { z {"}') == 0
    # A genuinely unclosed structural brace.
    assert _count_unclosed_braces('{"a": {"b": 1}') == 1
    # Escaped quotes must not end the string early.
    assert _count_unclosed_braces('{"a": "he said \\"{\\" loudly"}') == 0


def test_genuinely_truncated_json_still_gets_closing_braces():
    from nite_eval.hermes_parser import _fix_json

    fixed, _ = _fix_json('{"name": "f", "arguments": {"path": "/a"')
    import json as _json

    assert _json.loads(fixed)["arguments"]["path"] == "/a"


# --- Ornith XML-format parsing (ornith-1.5-35b-a3b) ---
#
# Ornith's chat template hardcodes an XML tool-call format:
#   <tool_call><function=NAME><parameter=KEY>value</parameter></function></tool_call>
# The body is not JSON, so the Hermes path raises JSONDecodeError. Without a
# fallback every Ornith tool call is discarded as malformed_json and the model
# is scored on a tool-less answer.

XML_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer"},
                    "fuzzy": {"type": "boolean"},
                    "tags": {"type": "array"},
                    "opts": {"type": "object"},
                },
                "required": ["query"],
            },
        },
    },
]


def test_xml_single_call():
    response = (
        "<tool_call>\n<function=search>\n<parameter=query>\nrust ownership\n</parameter>\n</function>\n</tool_call>"
    )
    parsed = extract_tool_calls(response, XML_TOOLS)
    assert len(parsed.tool_calls) == 1
    assert parsed.tool_calls[0].name == "search"
    assert parsed.tool_calls[0].arguments == {"query": "rust ownership"}
    assert parsed.errors == []


def test_xml_multiline_value_preserved():
    """Multi-line values must survive verbatim — this is how files get written."""
    body = 'package auth\n\nimport (\n\t"fmt"\n)\n'
    response = (
        f"<tool_call>\n<function=write_file>\n<parameter=path>\nauth.go\n</parameter>\n"
        f"<parameter=content>\n{body}</parameter>\n</function>\n</tool_call>"
    )
    parsed = extract_tool_calls(response, XML_TOOLS)
    assert parsed.tool_calls[0].arguments["path"] == "auth.go"
    assert parsed.tool_calls[0].arguments["content"] == body.rstrip("\n")


def test_xml_typed_coercion_against_schema():
    """XML carries no types; the declared schema type decides."""
    response = (
        "<tool_call>\n<function=search>\n"
        "<parameter=query>\nx\n</parameter>\n"
        "<parameter=limit>\n5\n</parameter>\n"
        "<parameter=fuzzy>\ntrue\n</parameter>\n"
        '<parameter=tags>\n["a", "b"]\n</parameter>\n'
        '<parameter=opts>\n{"deep": true}\n</parameter>\n'
        "</function>\n</tool_call>"
    )
    parsed = extract_tool_calls(response, XML_TOOLS)
    args = parsed.tool_calls[0].arguments
    assert args["limit"] == 5
    assert args["fuzzy"] is True
    assert args["tags"] == ["a", "b"]
    assert args["opts"] == {"deep": True}
    assert validate_tool_calls(parsed, XML_TOOLS) == []


def test_xml_string_param_is_never_coerced():
    """A JSON-looking string param stays a string — coding tasks write JSON files."""
    response = (
        "<tool_call>\n<function=write_file>\n"
        "<parameter=path>\nconfig.json\n</parameter>\n"
        '<parameter=content>\n{"a": 1}\n</parameter>\n'
        "</function>\n</tool_call>"
    )
    parsed = extract_tool_calls(response, XML_TOOLS)
    assert parsed.tool_calls[0].arguments["content"] == '{"a": 1}'
    assert validate_tool_calls(parsed, XML_TOOLS) == []


def test_xml_without_schema_keeps_strings():
    """No tools passed: stay a string rather than guessing a type."""
    response = "<tool_call>\n<function=search>\n<parameter=limit>\n5\n</parameter>\n</function>\n</tool_call>"
    parsed = extract_tool_calls(response)
    assert parsed.tool_calls[0].arguments["limit"] == "5"


def test_xml_multiple_calls():
    response = (
        "<tool_call>\n<function=search>\n<parameter=query>\na\n</parameter>\n</function>\n</tool_call>\n"
        "<tool_call>\n<function=search>\n<parameter=query>\nb\n</parameter>\n</function>\n</tool_call>"
    )
    parsed = extract_tool_calls(response, XML_TOOLS)
    assert [tc.arguments["query"] for tc in parsed.tool_calls] == ["a", "b"]


def test_xml_zero_arg_call():
    response = "<tool_call>\n<function=list_files>\n</function>\n</tool_call>"
    parsed = extract_tool_calls(response, XML_TOOLS)
    assert parsed.tool_calls[0].name == "list_files"
    assert parsed.tool_calls[0].arguments == {}


def test_xml_stripped_from_text():
    """Raw XML must not leak into the answer the judge sees."""
    response = (
        "Let me look that up.\n<tool_call>\n<function=search>\n"
        "<parameter=query>\nx\n</parameter>\n</function>\n</tool_call>"
    )
    parsed = extract_tool_calls(response, XML_TOOLS)
    assert "<function=" not in parsed.text
    assert parsed.text == "Let me look that up."


def test_xml_without_tool_call_wrapper():
    """Models drop the wrapper; the call should still be found."""
    response = "<function=search>\n<parameter=query>\nunwrapped\n</parameter>\n</function>"
    parsed = extract_tool_calls(response, XML_TOOLS)
    assert len(parsed.tool_calls) == 1
    assert parsed.tool_calls[0].arguments == {"query": "unwrapped"}


def test_hermes_still_wins_over_xml():
    """A valid Hermes call must not be reparsed by the XML path."""
    response = '<tool_call>{"name": "search", "arguments": {"query": "hermes"}}</tool_call>'
    parsed = extract_tool_calls(response, XML_TOOLS)
    assert len(parsed.tool_calls) == 1
    assert parsed.tool_calls[0].arguments == {"query": "hermes"}


def test_xml_malformed_still_errors():
    """Junk inside <tool_call> that is neither JSON nor XML stays an error."""
    response = "<tool_call>\nnot json and not xml\n</tool_call>"
    parsed = extract_tool_calls(response, XML_TOOLS)
    assert parsed.tool_calls == []
    assert parsed.errors[0]["error"] == "malformed_json"


# --- Ornith dropped name-value quote (ornith-1.5-35b-a3b) ---
#
# Measured on the live model at temperature 0, thinking on, 8 prompts: 4 came
# back as valid Hermes and 4 dropped the opening quote of the *value* after
# "name". That is one character away from qwen3.8's defect, which drops the
# opening quote of the following *key* — different position, so
# _repair_dropped_key_quote does not catch it.

ORNITH_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_thoughts",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}, "limit": {"type": "integer"}},
                "required": ["query"],
            },
        },
    },
]


def test_ornith_dropped_name_value_quote():
    """Verbatim sample from the live model."""
    response = (
        '<tool_call>\n{"name": search_thoughts", "arguments": '
        '{"query": "local LLM benchmarking", "limit": 5}}\n</tool_call>'
    )
    parsed = extract_tool_calls(response, ORNITH_TOOLS)
    assert len(parsed.tool_calls) == 1
    assert parsed.tool_calls[0].name == "search_thoughts"
    assert parsed.tool_calls[0].arguments == {"query": "local LLM benchmarking", "limit": 5}
    assert parsed.errors == []


def test_ornith_repair_is_counted():
    """The repair must stay visible in the report, not be silently absorbed."""
    response = '<tool_call>{"name": search_thoughts", "arguments": {"query": "x"}}</tool_call>'
    parsed = extract_tool_calls(response, ORNITH_TOOLS)
    assert parsed.repaired == 1


def test_valid_hermes_is_not_counted_as_repaired():
    response = '<tool_call>{"name": "search_thoughts", "arguments": {"query": "x"}}</tool_call>'
    parsed = extract_tool_calls(response, ORNITH_TOOLS)
    assert parsed.repaired == 0
    assert parsed.tool_calls[0].name == "search_thoughts"


def test_dropped_name_quote_repair_is_string_aware():
    """An identifier-then-quote sequence inside a string value must survive."""
    response = (
        '<tool_call>{"name": "capture_thought", "arguments": '
        '{"content": "he wrote map[string]any{x\\": 1} then stopped"}}</tool_call>'
    )
    parsed = extract_tool_calls(response, ORNITH_TOOLS)
    assert parsed.repaired == 0
    assert parsed.tool_calls[0].arguments["content"] == 'he wrote map[string]any{x": 1} then stopped'


def test_qwen38_key_quote_repair_still_works():
    """The pre-existing qwen3.8 repair must not regress."""
    response = '<tool_call>\n{"name": "write_file",\narguments": {"content": "package auth"}}\n</tool_call>'
    parsed = extract_tool_calls(response)
    assert len(parsed.tool_calls) == 1
    assert parsed.tool_calls[0].name == "write_file"
    assert parsed.repaired >= 1
