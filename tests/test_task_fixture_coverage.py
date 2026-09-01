"""The specific calls that models actually made must be answerable.

Each case below is a real call from run-20260831-212124 or run-20260901-015444
that the fixture could not answer, found by the unmatched_mock_calls counter
rather than by reading logs. Pinning them here stops a future fixture edit
quietly reintroducing the gap — the failure mode is invisible without this,
since an unanswered call only depresses a score.
"""

from pathlib import Path

import pytest
import yaml

from nite_eval.mock_tools import MockToolEnv

TASKS = Path(__file__).resolve().parent.parent / "tasks"


def _env(relpath: str) -> MockToolEnv:
    spec = yaml.safe_load((TASKS / relpath).read_text())
    return MockToolEnv.from_task_yaml(spec["mock_responses"])


@pytest.mark.parametrize(
    "arguments",
    [
        # ornith, twice across separate runs — a wine type the cellar lacks.
        {"filters": {"wine_type": "sparkling"}},
        # ornith on the prompt-text path, five times across three turns.
        {},
        # A filter combination nobody wrote a mock for.
        {"filters": {"wine_type": "red", "region": "Tuscany"}},
    ],
)
def test_query_inventory_answers_any_filter(arguments):
    env = _env("agentic/agentic_wine_medium_01.yaml")
    result = env.call("query_inventory", arguments)
    assert "content" in result, result
    assert env.unmatched_calls == []


def test_query_inventory_catch_all_does_not_shadow_real_data():
    """The catch-all must stay last: the matcher takes the first mock that fits."""
    env = _env("agentic/agentic_wine_medium_01.yaml")
    result = env.call("query_inventory", {"filters": {"wine_type": "white", "status": "in_cellar"}})
    assert len(result["content"]["bottles"]) == 3


@pytest.mark.parametrize(
    "query",
    [
        "NVDA earnings sentiment",  # the original, ticker form
        "Nvidia stock analyst price target AI",  # ornith's actual call
        "nvidia outlook",  # matching is case-insensitive
    ],
)
def test_search_news_accepts_ticker_and_company_name(query):
    env = _env("agentic/agentic_finance_hard_01.yaml")
    result = env.call("search_news", {"query": query, "days_back": 14})
    assert result["content"]["articles"], result
    assert env.unmatched_calls == []


def test_fetch_url_is_answerable():
    """Declared in tools: with no mocks — qwen3.6 spent 4 calls on it."""
    env = _env("research/research_finance_hard_01.yaml")
    result = env.call("fetch_url", {"url": "https://arxiv.org/abs/2403.07815"})
    assert "content" in result, result
    assert env.unmatched_calls == []


def test_every_declared_tool_is_mocked_or_a_named_distractor():
    """A tool a task offers must be callable — unless not calling it is the point.

    agentic_brain_easy_01 declares send_email with no mocks deliberately: it
    carries a tool_absence criterion worth 0.2 and a sequence_match that says
    "NOT send_email". Mocking it would defeat the task. So the rule is that an
    unmocked tool must be named by a scoring criterion, which is what separates
    a designed trap from fetch_url, which was simply forgotten.

    Sandbox-backed coding tasks are exempt: their tools run against a real
    container and carry no mock_responses at all.
    """
    missing: list[str] = []
    for path in sorted(TASKS.rglob("*.yaml")):
        spec = yaml.safe_load(path.read_text())
        mocks = spec.get("mock_responses")
        if not mocks:
            continue  # sandbox task
        scoring_text = yaml.safe_dump(spec.get("scoring", {}))
        declared = {t["function"]["name"] for t in spec.get("tools", [])}
        for tool in sorted(declared - set(mocks)):
            if tool not in scoring_text:
                missing.append(f"{path.name}: {tool}")
    assert missing == [], f"tools declared with neither mocks nor a scoring reason: {missing}"
