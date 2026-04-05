"""Parse and validate Hermes-format tool calls from model output.

Handles edge cases from real-world Hermes implementations:
- Arguments before name key in JSON
- Empty arguments for zero-arg functions
- Whitespace variance inside tags
- Malformed JSON with trailing commas
- Multiple tool calls in a single response
- Scratch pad reasoning blocks
"""

import json
import re
from dataclasses import dataclass, field

TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
SCRATCH_PAD_RE = re.compile(r"<scratch_pad>\s*(.*?)\s*</scratch_pad>", re.DOTALL)
TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")


@dataclass
class ToolCall:
    name: str
    arguments: dict
    raw: str


@dataclass
class ParsedResponse:
    tool_calls: list[ToolCall] = field(default_factory=list)
    scratch_pad: str | None = None
    text: str = ""
    errors: list[dict] = field(default_factory=list)


def _fix_json(raw: str) -> str:
    """Attempt to fix common JSON malformations."""
    return TRAILING_COMMA_RE.sub(r"\1", raw.strip())


def extract_tool_calls(response: str) -> ParsedResponse:
    """Extract all tool calls and scratch pad from a Hermes-format response."""
    result = ParsedResponse()

    scratch_match = SCRATCH_PAD_RE.search(response)
    if scratch_match:
        result.scratch_pad = scratch_match.group(1).strip()

    # Text is everything outside tool_call and scratch_pad tags
    text = TOOL_CALL_RE.sub("", response)
    text = SCRATCH_PAD_RE.sub("", text)
    result.text = text.strip()

    matches = TOOL_CALL_RE.findall(response)
    for raw_match in matches:
        try:
            parsed = json.loads(_fix_json(raw_match))
        except json.JSONDecodeError:
            result.errors.append({"error": "malformed_json", "raw": raw_match})
            continue

        name = parsed.get("name")
        if not name:
            result.errors.append({"error": "missing_name", "raw": raw_match})
            continue

        arguments = parsed.get("arguments", {})
        if not isinstance(arguments, dict):
            result.errors.append({"error": "invalid_arguments_type", "raw": raw_match})
            continue

        result.tool_calls.append(ToolCall(name=name, arguments=arguments, raw=raw_match))

    return result


@dataclass
class ValidationError:
    tool_call: ToolCall
    error: str
    details: dict = field(default_factory=dict)


def validate_tool_calls(
    parsed: ParsedResponse,
    available_tools: list[dict],
) -> list[ValidationError]:
    """Validate extracted tool calls against tool schemas.

    Args:
        parsed: Output from extract_tool_calls
        available_tools: List of OpenAI-format tool definitions
            [{"type": "function", "function": {"name": ..., "parameters": ...}}]

    Returns:
        List of validation errors (empty if all valid).
    """
    tool_map: dict[str, dict] = {}
    for tool in available_tools:
        func = tool.get("function", {})
        tool_map[func["name"]] = func

    errors: list[ValidationError] = []
    for tc in parsed.tool_calls:
        if tc.name not in tool_map:
            errors.append(ValidationError(tc, "unknown_function", {"available": list(tool_map.keys())}))
            continue

        func_def = tool_map[tc.name]
        params = func_def.get("parameters", {})
        required = params.get("required", [])
        missing = [r for r in required if r not in tc.arguments]
        if missing:
            errors.append(ValidationError(tc, "missing_required_params", {"missing": missing}))

        # Type checking for provided arguments
        properties = params.get("properties", {})
        for arg_name, arg_value in tc.arguments.items():
            if arg_name not in properties:
                continue  # Extra args are allowed (models may add reasoning)
            expected_type = properties[arg_name].get("type")
            if expected_type and not _type_matches(arg_value, expected_type):
                errors.append(
                    ValidationError(
                        tc,
                        "type_mismatch",
                        {"param": arg_name, "expected": expected_type, "got": type(arg_value).__name__},
                    )
                )

    return errors


def _type_matches(value: object, json_type: str) -> bool:
    """Check if a Python value matches a JSON schema type."""
    match json_type:
        case "string":
            return isinstance(value, str)
        case "integer":
            return isinstance(value, int) and not isinstance(value, bool)
        case "number":
            return isinstance(value, int | float) and not isinstance(value, bool)
        case "boolean":
            return isinstance(value, bool)
        case "array":
            return isinstance(value, list)
        case "object":
            return isinstance(value, dict)
        case _:
            return True  # Unknown type, don't validate


HERMES_TOOL_PREAMBLE = """\
You are a function calling AI model. You are provided with function signatures \
within <tools></tools> XML tags. You may call one or more functions to assist \
with the user query. Don't make assumptions about what values to plug into \
functions. Here are the available tools:"""

HERMES_TOOL_INSTRUCTIONS = """
For each function call, return a JSON object with function name and arguments \
within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>"""


def format_tool_definitions(tools: list[dict]) -> str:
    """Format tool definitions with full Hermes-format instructions."""
    return (
        HERMES_TOOL_PREAMBLE + "\n<tools>\n" + json.dumps(tools, indent=2) + "\n</tools>\n" + HERMES_TOOL_INSTRUCTIONS
    )


def format_tool_response(name: str, content: dict) -> str:
    """Format a tool response for injection into the conversation."""
    return "<tool_response>" + json.dumps({"name": name, "content": content}) + "</tool_response>"
