"""research_finance_hard_01 must reward retrieval, not recall.

With `tools:` stripped the task scored HIGHER than grounded — 0.787 against
0.674 over 4 models — because every figure in its fixtures was a real public
fact and none of its criteria asked where the answer came from. The fix seeds
facts that exist nowhere but the task file and gives them weight through a
deterministic `contains_check`.

That only works while three things hold, and none of them is self-evident from
reading the YAML:

  * the canaries are reachable from the mocks (else nobody can score it),
  * they are absent from the prompt (else they prove nothing),
  * they are strings (else the run dies partway through with AttributeError).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from nite_eval.mock_tools import MockToolEnv
from nite_eval.scoring import score_contains_check

TASKS = Path(__file__).resolve().parent.parent / "tasks"
TASK_PATH = TASKS / "research" / "research_finance_hard_01.yaml"
TASK = yaml.safe_load(TASK_PATH.read_text())
CANARIES = TASK["scoring"]["grounding"]["criteria"]


def _all_task_files() -> list[Path]:
    return sorted(TASKS.rglob("*.yaml"))


@pytest.mark.parametrize("path", _all_task_files(), ids=lambda p: p.stem)
def test_contains_check_criteria_are_strings(path: Path):
    """score_contains_check calls .lower() on each criterion.

    An unquoted 0.847 is a YAML float, and the AttributeError does not surface
    until that task is scored — deep into a run, after the GPU time is spent.
    """
    spec = yaml.safe_load(path.read_text())
    for name, cfg in (spec.get("scoring") or {}).items():
        if cfg.get("method") != "contains_check":
            continue
        criteria = cfg["criteria"]
        assert isinstance(criteria, list), f"{path.name}/{name}"
        for c in criteria:
            assert isinstance(c, str), f"{path.name}/{name}: {c!r} is {type(c).__name__}, quote it"


def test_weights_still_sum_to_one():
    total = sum(cfg["weight"] for cfg in TASK["scoring"].values())
    assert total == pytest.approx(1.0), TASK["scoring"]


def test_grounding_is_deterministic():
    """It must not cost a judge call, and must not be judge-routed by name."""
    assert TASK["scoring"]["grounding"]["method"] == "contains_check"


# Reachability and prompt-absence are checked for every grounding task in
# test_grounding_criteria.py, parameterised over the task files.


def test_a_single_generic_search_grounds_partially():
    """The catch-all has to carry a canary.

    A model that searches once and writes its answer should be able to score
    something, or the criterion measures thoroughness rather than grounding.
    """
    env = MockToolEnv.from_task_yaml(TASK["mock_responses"])
    result = env.call("web_search", {"query": "transformer time series forecasting overview"})
    assert "content" in result, result
    text = yaml.dump(result)
    assert any(c in text for c in CANARIES), text


def test_the_real_public_figures_are_deliberately_still_there():
    """Overwriting them would measure deference to a tool, not research.

    Pinned so a later edit does not quietly turn this into a trust-the-fixture
    task without saying so.
    """
    fixtures = yaml.dump(TASK["mock_responses"])
    for real in ("200M", "710M", "rank 8-16"):
        assert real in fixtures, f"{real!r} was removed — see the comment above `scoring:`"


def test_an_answer_from_memory_scores_zero():
    from_memory = (
        "TimesFM is a 200M parameter decoder-only foundation model pre-trained on "
        "100B time points. Chronos is T5-based, 20M to 710M. For PEFT, LoRA rank "
        "8-16 is sufficient. On a 24GB GPU all of these fit comfortably."
    )
    assert score_contains_check(from_memory, CANARIES) == 0.0


def test_a_grounded_answer_scores_the_fraction_it_cites():
    grounded = (
        "PatchTST tops SwingBench-24 at MASE 0.796, with TimesFM at 0.847 zero-shot. "
        "Fine-tuning on the FIN-27 corpus peaked at 11.3 GB on a single 24GB card."
    )
    assert score_contains_check(grounded, CANARIES) == 1.0

    partial = "SwingBench-24 is the relevant benchmark here, and TimesFM leads the zero-shot entries."
    assert score_contains_check(partial, CANARIES) == pytest.approx(0.25)
