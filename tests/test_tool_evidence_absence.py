"""A judge told nothing about missing tool calls invents the tool calls.

`build_code_evidence` learned this the expensive way: omitting the section when
no files were written left `error_handling` at 3.67 and `edge_case_handling` at
4.00, the reasoning describing an implementation that was never written. Stating
the absence sent all three to 1.00. The constant NO_FILES_WRITTEN exists for
that reason.

`build_tool_evidence` had the same shape and not the same fix — it returned ""
for a conversation with no tool calls, which reaches the judge as an absent
section rather than as evidence of absence.

The defect is latent today: only agentic_artemis_medium_01 and
agentic_finance_hard_01 carry EVIDENCE_DIMENSIONS criteria, and no agentic run
has ever completed with zero tool calls (0 of 255). It goes live the moment a
criterion on a research or planning task takes tool evidence — planning
completes with no tool calls 22.8% of the time.
"""

from nite_eval.evidence import NO_TOOLS_CALLED, build_tool_evidence


class FakeTurn:
    def __init__(self, tool_responses):
        self.tool_responses = tool_responses


class FakeConv:
    def __init__(self, turns):
        self.turns = turns


def _conv(*calls):
    return FakeConv([FakeTurn(list(calls))])


SEARCH_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
        },
    }
]


def test_no_calls_on_a_task_that_offered_tools_states_the_absence():
    """The case the fix exists for: the model could have looked and did not."""
    ev = build_tool_evidence(_conv(), tools=SEARCH_TOOL)
    assert ev == NO_TOOLS_CALLED
    assert "NONE" in ev


def test_the_statement_names_the_consequence_not_just_the_absence():
    """ "No tools" alone leaves the judge free to assume the facts check out.

    NO_FILES_WRITTEN works because it says what the absence means for scoring,
    not merely that something is missing.
    """
    assert "1" in NO_TOOLS_CALLED
    assert "grounded" in NO_TOOLS_CALLED.lower()


def test_no_calls_on_a_task_with_no_tools_stays_silent():
    """Not every task offers tools. Announcing an absence there is noise."""
    assert build_tool_evidence(_conv(), tools=[]) == ""
    assert build_tool_evidence(_conv(), tools=None) == ""


def test_calls_present_keeps_the_original_shape():
    """Regression: the evidence format is what the fact-checking rubrics read."""
    conv = _conv({"name": "get_price", "arguments": {"sym": "AAPL"}, "result": {"price": 123}})
    ev = build_tool_evidence(conv, tools=SEARCH_TOOL)
    assert "get_price" in ev
    assert "123" in ev
    assert "NONE" not in ev


def test_tools_argument_is_optional_for_existing_callers():
    """The old one-argument call must keep working and keep its old behaviour."""
    conv = _conv({"name": "get_price", "arguments": {}, "result": {"price": 1}})
    assert "get_price" in build_tool_evidence(conv)
    assert build_tool_evidence(_conv()) == ""


def test_a_turn_with_tools_declared_but_only_failed_calls_still_counts_as_called():
    """A call that errored is still evidence the model looked."""
    conv = _conv({"name": "web_search", "arguments": {"query": "x"}, "result": {"error": "boom"}})
    ev = build_tool_evidence(conv, tools=SEARCH_TOOL)
    assert "web_search" in ev
    assert ev != NO_TOOLS_CALLED
