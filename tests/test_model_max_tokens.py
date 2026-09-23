"""Per-model max_tokens resolution.

Until 2026-09-22 a `max_tokens:` under a model entry was never read — the only
resolution was task-then-global, so the key documented in `eval_config.yaml`'s
frontier examples was inert. That is the same silent-no-op shape as a
`chat_template_kwargs` key the template does not contain: it looks configured
and does nothing.

Precedence is model > task > global, so a model entry can *lower* a task's
budget as well as raise it. Bonsai 2 is the case that motivated it: its two
coding failures each fill the 32768-token coding budget with a loop, and a
per-model ceiling bounds that without touching any other model's budget.
"""

from nite_eval.orchestrator import resolve_max_tokens

EVAL_CFG = {"max_tokens": 2048}


def test_task_wins_when_model_is_silent():
    assert resolve_max_tokens(32768, {}, EVAL_CFG) == 32768


def test_global_used_when_task_and_model_are_silent():
    assert resolve_max_tokens(None, {}, EVAL_CFG) == 2048


def test_built_in_default_when_nothing_is_configured():
    assert resolve_max_tokens(None, {}, {}) == 2048


def test_model_overrides_task():
    """The bonsai case: cap a looping model below the coding budget."""
    assert resolve_max_tokens(32768, {"max_tokens": 8192}, EVAL_CFG) == 8192


def test_model_overrides_global():
    assert resolve_max_tokens(None, {"max_tokens": 8192}, EVAL_CFG) == 8192


def test_model_may_raise_above_the_task_budget():
    """Frontier entries use this to give API models more room than locals."""
    assert resolve_max_tokens(2048, {"max_tokens": 8192}, EVAL_CFG) == 8192


def test_zero_is_treated_as_unset_at_every_level():
    """Matches the `or` semantics the task-level read already had."""
    assert resolve_max_tokens(0, {"max_tokens": 0}, EVAL_CFG) == 2048
