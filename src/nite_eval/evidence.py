"""What the judge is allowed to see besides the model's closing text.

Two separate things, deliberately kept apart:

- `build_tool_evidence` — every tool call and its result, as ground truth for
  criteria that ask whether the response's facts match what the tools actually
  returned (no_hallucination, data_accuracy, data_threading). Without it those
  criteria were unanswerable and fell through to a free 1.0.

- `build_code_evidence` — the files the model actually wrote. Coding criteria
  were scored from `conv.final_response` alone, and code is written through
  `write_file` tool calls, so the judges had never seen a line of it. They were
  scoring the model's prose summary of its own work: on
  `coding_wine_medium_01`, nine tasks where the file did not exist scored a
  judge average of 0.75, response length irrelevant — 28 characters and 4593
  characters alike.
"""

import json

# A call is a file write if it carries somewhere to put it and something to put
# there. Matching on shape rather than on the name `write_file` keeps this
# working for a task that names its tool differently, which task YAMLs are free
# to do.
_PATH_KEYS = ("path", "file_path", "filename", "file", "filepath")
_CONTENT_KEYS = ("content", "file_content", "text", "body")

DEFAULT_MAX_CHARS = 24_000

# What the judge is told when the task offered a way to write files and the
# model wrote none. Omitting the section instead was measured against a live
# reward-anything on the real coding_wine_medium_01 non-answer and did not
# work: code_quality fell 4.00 -> 1.67, but error_handling held at 3.67 and
# edge_case_handling at 4.00, the reasoning still describing an implementation
# that was never written. An absent section is not a signal — the judge fills
# the gap from the task specification. The absence has to be stated.
NO_FILES_WRITTEN = (
    "NONE. The model did not create any files. There is no implementation to "
    "score, so every criterion about the code scores 1."
)


def _iter_tool_responses(conv):
    for turn in conv.turns:
        yield from turn.tool_responses


def _first_present(args: dict, keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = args.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def build_tool_evidence(conv) -> str:
    """Every tool call and result, one per line, for fact-checking criteria."""
    lines = []
    for tr in _iter_tool_responses(conv):
        args = json.dumps(tr.get("arguments", {}))
        result = json.dumps(tr.get("result", {}))
        lines.append(f"{tr['name']}({args}) -> {result}")
    return "\n".join(lines)


def declares_file_writing_tool(tools) -> bool:
    """Did the task offer the model any way to write a file?

    Distinguishes "wrote nothing and could have" — which the judge must be told
    about — from "wrote nothing because there was no such tool", which is the
    normal case for research and planning and must stay silent.
    """
    for tool in tools or []:
        fn = tool.get("function", tool) if isinstance(tool, dict) else {}
        props = ((fn.get("parameters") or {}).get("properties") or {}) if isinstance(fn, dict) else {}
        if any(k in props for k in _CONTENT_KEYS) and any(k in props for k in _PATH_KEYS):
            return True
    return False


def build_code_evidence(conv, max_chars: int = DEFAULT_MAX_CHARS, tools=None) -> str:
    """The final contents of every file the model wrote, in write order.

    When the model wrote nothing, returns an explicit statement of that fact if
    the task offered a file-writing tool, and "" if it did not. The explicit
    statement is load-bearing: see NO_FILES_WRITTEN.

    A path written more than once keeps its last version — a model that writes
    then fixes should be judged on the fix — while holding its original
    position, so the judge reads files in the order work started on them.
    """
    order: list[str] = []
    latest: dict[str, str] = {}
    writes: dict[str, int] = {}

    for tr in _iter_tool_responses(conv):
        args = tr.get("arguments") or {}
        if not isinstance(args, dict):
            continue
        content = _first_present(args, _CONTENT_KEYS)
        if content is None:
            continue
        path = _first_present(args, _PATH_KEYS)
        if path is None:
            continue
        if path not in latest:
            order.append(path)
        latest[path] = content
        writes[path] = writes.get(path, 0) + 1

    if not order:
        return NO_FILES_WRITTEN if declares_file_writing_tool(tools) else ""

    # Budget is shared across files so one enormous file cannot crowd out the
    # others entirely; every file keeps at least its path and a slice.
    per_file = max(400, max_chars // len(order))
    blocks = []
    for path in order:
        body = latest[path]
        note = f"{len(body)} chars"
        if writes[path] > 1:
            note += f", final of {writes[path]} writes"
        if len(body) > per_file:
            body = body[:per_file] + f"\n\n[... truncated, {len(latest[path]) - per_file} chars not shown ...]"
        blocks.append(f"### {path} ({note})\n{body}")
    return "\n\n".join(blocks)
