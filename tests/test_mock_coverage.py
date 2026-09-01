"""A call that no mock answers must be visible, not just logged.

An unmatched call is returned to the model as an error and warned about in the
log, and recorded nowhere else. The model then burns turns retrying, answers
without the data, and scores lower — with nothing in the report to say the
fixture was the problem rather than the model.

That silently penalises any model whose phrasing differs from whoever wrote the
mocks. ornith-1.5 called `query_inventory({'filters': {'wine_type': 'sparkling'}})`
on agentic_wine_medium_01, a sensible query against the declared schema, and got
an error because the fixture only covers white, rosé and red. Earlier, with the
schema reaching it only as prose, it called `query_inventory({})` five times
across three turns — that task has seven mocks and no catch-all, unlike every
other tool in it.

Counting the misses does not change any score. It makes the gap legible so a
low score can be attributed to the model or to the fixture.
"""

from nite_eval.mock_tools import MockToolEnv

MOCKS = {
    "query_inventory": [
        {"match": {"filters": {"wine_type": "white"}}, "response": {"content": {"bottles": ["b001"]}}},
    ],
    "get_taste_profile": [{"match": {}, "response": {"content": {"likes": "dry"}}}],
}


def test_matched_calls_record_no_miss():
    env = MockToolEnv.from_task_yaml(MOCKS)
    env.call("query_inventory", {"filters": {"wine_type": "white"}})
    env.call("get_taste_profile", {})
    assert env.unmatched_calls == []


def test_unmatched_arguments_are_recorded():
    """The sparkling case: valid against the schema, absent from the fixture."""
    env = MockToolEnv.from_task_yaml(MOCKS)
    result = env.call("query_inventory", {"filters": {"wine_type": "sparkling"}})
    assert "error" in result  # behaviour unchanged
    assert len(env.unmatched_calls) == 1
    miss = env.unmatched_calls[0]
    assert miss["name"] == "query_inventory"
    assert miss["arguments"] == {"filters": {"wine_type": "sparkling"}}
    assert miss["reason"] == "no_matching_mock"


def test_call_with_no_arguments_is_recorded():
    """A tool with no catch-all rejects the most natural first call."""
    env = MockToolEnv.from_task_yaml(MOCKS)
    env.call("query_inventory", {})
    assert [m["reason"] for m in env.unmatched_calls] == ["no_matching_mock"]


def test_undefined_tool_is_recorded_separately():
    """fetch_url and search_news are declared in tasks and never mocked.

    A different gap from an unmatched argument set, and worth telling apart:
    one is missing fixture data, the other a tool nobody wrote mocks for.
    """
    env = MockToolEnv.from_task_yaml(MOCKS)
    env.call("fetch_url", {"url": "https://example.com"})
    assert len(env.unmatched_calls) == 1
    assert env.unmatched_calls[0]["reason"] == "no_mock_for_tool"


def test_repeated_misses_all_count():
    """Five identical unanswered calls is the signal, not one."""
    env = MockToolEnv.from_task_yaml(MOCKS)
    for _ in range(5):
        env.call("query_inventory", {})
    assert len(env.unmatched_calls) == 5


def test_reset_clears_misses():
    env = MockToolEnv.from_task_yaml(MOCKS)
    env.call("query_inventory", {})
    env.reset()
    assert env.unmatched_calls == []
