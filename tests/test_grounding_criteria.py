"""Every `grounding` criterion, on every task that has one.

A canary only proves retrieval while three things hold, and none is visible when
reading the YAML:

  * it is reachable from the mocks, or it is weight the best model cannot score,
  * it is absent from everything the model is handed, or it proves nothing,
  * a single plausible tool call reaches at least one, or the criterion measures
    thoroughness rather than grounding.

Parameterised over the task files, so a task added later is covered without
editing this. The string-type check for contains_check criteria lives in
test_research_grounding_canaries.py and already spans every task.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from nite_eval.mock_tools import MockToolEnv

TASKS = Path(__file__).resolve().parent.parent / "tasks"


def _grounding_tasks() -> list[tuple[str, dict]]:
    out = []
    for p in sorted(TASKS.rglob("*.yaml")):
        spec = yaml.safe_load(p.read_text())
        for name, cfg in (spec.get("scoring") or {}).items():
            if cfg.get("method") == "contains_check":
                out.append((f"{spec['id']}/{name}", spec))
    return out


def _cases() -> list[tuple[str, dict, str]]:
    cases = []
    for label, spec in _grounding_tasks():
        crit = next(c for n, c in spec["scoring"].items() if c.get("method") == "contains_check")
        for canary in crit["criteria"]:
            cases.append((label, spec, canary))
    return cases


GROUNDING = _grounding_tasks()
CASES = _cases()


def test_at_least_one_task_uses_grounding():
    """Guards against the parameterised tests below silently covering nothing."""
    assert GROUNDING, "no task declares a contains_check criterion"


@pytest.mark.parametrize("label,spec,canary", CASES, ids=[f"{c[0]}:{c[2]}" for c in CASES])
def test_canary_is_reachable_from_the_fixtures(label, spec, canary):
    fixtures = yaml.dump(spec.get("mock_responses") or {})
    assert canary in fixtures, f"{label}: {canary!r} is scored but appears in no mock response"


@pytest.mark.parametrize("label,spec,canary", CASES, ids=[f"{c[0]}:{c[2]}" for c in CASES])
def test_canary_is_absent_from_everything_the_model_is_handed(label, spec, canary):
    handed = " ".join(
        [
            spec.get("user_message", ""),
            spec.get("system_prompt", ""),
            spec.get("description", ""),
            yaml.dump(spec.get("tools") or []),
        ]
    )
    assert canary not in handed, f"{label}: {canary!r} is in the prompt, so citing it proves nothing"


@pytest.mark.parametrize("label,spec", GROUNDING, ids=[g[0] for g in GROUNDING])
def test_the_catch_all_carries_a_canary(label, spec):
    """A model that makes one plausible call should be able to score something.

    Otherwise the criterion rewards making many calls rather than making one.
    """
    crit = next(c for n, c in spec["scoring"].items() if c.get("method") == "contains_check")
    env = MockToolEnv.from_task_yaml(spec["mock_responses"])
    reached = set()
    for tool, mocks in spec["mock_responses"].items():
        for mock in mocks:
            args = {k.removesuffix("_contains"): "any" for k in (mock.get("match") or {})}
            result = env.call(tool, args)
            text = yaml.dump(result)
            reached |= {c for c in crit["criteria"] if c in text}
    assert reached, f"{label}: no single tool call reaches any canary"
