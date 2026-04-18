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

# Gemma's chat template emits a Harmony-like format:
#   <|tool_call>call:FUNC{key:<|"|>val<|"|>,key:[<|"|>a<|"|>,<|"|>b<|"|>]}<tool_call|>
# with `<|channel>thought...<channel|>` blocks as scratchpad.
# Some variants use "..." for strings instead of <|"|>...<|"|>.
GEMMA_TOOL_CALL_RE = re.compile(r"<\|tool_call>(.*?)<tool_call\|>", re.DOTALL)
GEMMA_CHANNEL_RE = re.compile(r"<\|channel>(.*?)<channel\|>", re.DOTALL)
GEMMA_CALL_PREFIX_RE = re.compile(r"\s*call:([A-Za-z_][A-Za-z0-9_]*)\s*(\{.*\})\s*$", re.DOTALL)
GEMMA_STRING_DELIM = '<|"|>'


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
    fixed = TRAILING_COMMA_RE.sub(r"\1", raw.strip())

    # Fix missing closing braces — count open vs close
    open_braces = fixed.count("{")
    close_braces = fixed.count("}")
    if open_braces > close_braces:
        fixed += "}" * (open_braces - close_braces)

    return fixed


def _quote_bare_keys(s: str) -> str:
    """Quote bare identifiers that appear as JSON keys.

    Walks the string tracking inside-string state (delimiter `"`, respecting
    backslash escapes). Any `[A-Za-z_]\\w*` immediately followed by `:` and
    found *outside* a string is wrapped in quotes. Needed because Gemma's
    tool-call args look like `{key:"value"}` — valid-ish but missing key quotes.
    """
    out: list[str] = []
    i = 0
    in_string = False
    ident_re = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)(\s*:)")
    while i < len(s):
        ch = s[i]
        if in_string:
            if ch == "\\" and i + 1 < len(s):
                out.append(ch)
                out.append(s[i + 1])
                i += 2
                continue
            if ch == '"':
                in_string = False
            out.append(ch)
            i += 1
            continue
        if ch == '"':
            in_string = True
            out.append(ch)
            i += 1
            continue
        m = ident_re.match(s, i)
        if m and (not out or out[-1] not in ('"',)):
            ident, colon = m.group(1), m.group(2)
            out.append(f'"{ident}"{colon}')
            i = m.end()
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def _extract_gemma_tool_calls(response: str) -> tuple[list[ToolCall], list[dict]]:
    """Extract Gemma-format tool calls. Returns (calls, errors)."""
    calls: list[ToolCall] = []
    errors: list[dict] = []
    for raw in GEMMA_TOOL_CALL_RE.findall(response):
        m = GEMMA_CALL_PREFIX_RE.match(raw)
        if not m:
            errors.append({"error": "malformed_gemma_call", "raw": raw})
            continue
        name = m.group(1)
        args_raw = m.group(2)
        # Normalize Gemma string delimiters to standard "
        args_norm = args_raw.replace(GEMMA_STRING_DELIM, '"')
        # Quote bare keys so JSON can parse
        args_json = _quote_bare_keys(args_norm)
        try:
            args = json.loads(_fix_json(args_json))
        except json.JSONDecodeError:
            errors.append({"error": "malformed_gemma_json", "raw": args_raw})
            continue
        if not isinstance(args, dict):
            errors.append({"error": "invalid_arguments_type", "raw": args_raw})
            continue
        # Gemma sometimes mixes Hermes-style wrapping into its own format:
        #   call:web_search{arguments:{query:"..."}}
        # If the outer dict has exactly one key "arguments" containing a dict,
        # unwrap it so mocks see the real args. Preserves legitimate cases
        # like call:call_mcp_tool{arguments:{...},server:"..."} where
        # "arguments" is a real parameter alongside others.
        if len(args) == 1 and isinstance(args.get("arguments"), dict):
            args = args["arguments"]
        calls.append(ToolCall(name=name, arguments=args, raw=raw))
    return calls, errors


def extract_tool_calls(response: str) -> ParsedResponse:
    """Extract all tool calls and scratch pad from a Hermes-format response.

    Falls back to Gemma/Harmony-style `<|tool_call>call:FUNC{…}<tool_call|>`
    when no Hermes tool calls are present — lets gemma4 models participate
    in tool-driven tasks instead of being silently handicapped.
    """
    result = ParsedResponse()

    scratch_match = SCRATCH_PAD_RE.search(response)
    if scratch_match:
        result.scratch_pad = scratch_match.group(1).strip()

    # Text is everything outside tool_call, scratch_pad, and Gemma-channel tags
    text = TOOL_CALL_RE.sub("", response)
    text = SCRATCH_PAD_RE.sub("", text)
    text = GEMMA_TOOL_CALL_RE.sub("", text)
    text = GEMMA_CHANNEL_RE.sub("", text)
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

    # Gemma fallback only runs when the Hermes path found nothing — avoids
    # double-parsing in the mixed-format case (unlikely but safe).
    if not result.tool_calls and GEMMA_TOOL_CALL_RE.search(response):
        gemma_calls, gemma_errors = _extract_gemma_tool_calls(response)
        result.tool_calls.extend(gemma_calls)
        result.errors.extend(gemma_errors)
        if result.scratch_pad is None:
            channel_match = GEMMA_CHANNEL_RE.search(response)
            if channel_match:
                result.scratch_pad = channel_match.group(1).strip()

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
