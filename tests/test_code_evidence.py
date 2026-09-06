"""The coding judges must see the code, not the model's summary of it.

Regression tests for the defect found in run-20260905-235130: every
judge_rubric criterion on a coding task was scored from `conv.final_response`
alone. Code is written through `write_file` tool calls, which appear in neither
`final_response` nor the tool-results evidence, so the judges had never seen a
line of code — on failing runs or winning ones. They scored the model's prose
description of its own work.

Measured consequence: nine tasks where the file did not exist at all, seven of
them scoring exactly 0.75 on every judge criterion, response length irrelevant
(28 chars and 4593 chars alike).
"""

from nite_eval.evidence import build_code_evidence, build_tool_evidence


class FakeTurn:
    def __init__(self, tool_responses):
        self.tool_responses = tool_responses


class FakeConv:
    def __init__(self, turns):
        self.turns = turns


def _conv(*calls):
    return FakeConv([FakeTurn(list(calls))])


def test_written_file_reaches_the_judge():
    conv = _conv(
        {"name": "write_file", "arguments": {"path": "/app/a.ts", "content": "export const x = 1;"}, "result": {}},
    )
    ev = build_code_evidence(conv)
    assert "/app/a.ts" in ev
    assert "export const x = 1;" in ev


WRITE_TOOL = [{"function": {"name": "write_file", "parameters": {"properties": {"path": {}, "content": {}}}}}]
NO_WRITE_TOOL = [{"function": {"name": "search", "parameters": {"properties": {"query": {}}}}}]


def test_no_writes_states_the_absence_when_a_write_tool_existed():
    """qwen3.6 on coding_wine_medium_01: zero tool calls, 28-char response.

    Omitting the section here was measured and does not work — against a live
    reward-anything, code_quality fell 4.00 -> 1.67 but error_handling held at
    3.67 and edge_case_handling at 4.00, still describing code that was never
    written. The judge fills an absent section from the task spec, so the
    absence has to be spelled out.
    """
    ev = build_code_evidence(_conv(), tools=WRITE_TOOL)
    assert "NONE" in ev
    assert "scores 1" in ev

    ev2 = build_code_evidence(
        _conv({"name": "run_code", "arguments": {"command": "ls"}, "result": {}}), tools=WRITE_TOOL
    )
    assert "NONE" in ev2


def test_no_writes_stays_silent_when_no_write_tool_was_offered():
    """Research and planning tasks write no files by design."""
    assert build_code_evidence(_conv(), tools=NO_WRITE_TOOL) == ""
    assert build_code_evidence(_conv(), tools=None) == ""


def test_later_write_supersedes_earlier_one():
    """A model that writes then fixes should be judged on the fixed version."""
    conv = _conv(
        {"name": "write_file", "arguments": {"path": "/app/a.ts", "content": "BROKEN"}, "result": {}},
        {"name": "write_file", "arguments": {"path": "/app/a.ts", "content": "FIXED"}, "result": {}},
    )
    ev = build_code_evidence(conv)
    assert "FIXED" in ev
    assert "BROKEN" not in ev


def test_multiple_files_all_present_and_ordered():
    conv = _conv(
        {"name": "write_file", "arguments": {"path": "/app/b.ts", "content": "second"}, "result": {}},
        {"name": "write_file", "arguments": {"path": "/app/a.ts", "content": "first"}, "result": {}},
    )
    ev = build_code_evidence(conv)
    assert "/app/a.ts" in ev and "/app/b.ts" in ev
    # first written appears first — the judge reads it in the order it was built
    assert ev.index("/app/b.ts") < ev.index("/app/a.ts")


def test_non_writing_tools_are_excluded():
    conv = _conv(
        {"name": "run_code", "arguments": {"command": "go test ./..."}, "result": {"stdout": "ok"}},
        {"name": "read_file", "arguments": {"path": "/app/README.md"}, "result": {"content": "docs"}},
        {"name": "write_file", "arguments": {"path": "/app/a.go", "content": "package main"}, "result": {}},
    )
    ev = build_code_evidence(conv)
    assert "package main" in ev
    assert "go test" not in ev
    assert "docs" not in ev


def test_alternate_argument_names_are_recognised():
    """Not every task names the parameters path/content."""
    conv = _conv(
        {"name": "write_file", "arguments": {"file_path": "/app/a.py", "content": "x = 1"}, "result": {}},
    )
    assert "/app/a.py" in build_code_evidence(conv)
    assert "x = 1" in build_code_evidence(conv)


def test_oversized_content_is_truncated_but_path_survives():
    conv = _conv(
        {"name": "write_file", "arguments": {"path": "/app/big.ts", "content": "x" * 50_000}, "result": {}},
    )
    ev = build_code_evidence(conv, max_chars=1000)
    assert "/app/big.ts" in ev
    assert len(ev) < 2000
    assert "truncated" in ev.lower()


def test_write_without_content_is_skipped_not_crashed():
    conv = _conv({"name": "write_file", "arguments": {"path": "/app/a.ts"}, "result": {}})
    assert build_code_evidence(conv) == ""
    assert "NONE" in build_code_evidence(conv, tools=WRITE_TOOL)


def test_tool_evidence_still_works_and_is_separate():
    """The pre-existing fact-checking evidence must not change shape."""
    conv = _conv({"name": "get_price", "arguments": {"sym": "AAPL"}, "result": {"price": 123}})
    ev = build_tool_evidence(conv)
    assert "get_price" in ev
    assert "123" in ev
    # a lookup is not a code artifact
    assert build_code_evidence(conv) == ""
