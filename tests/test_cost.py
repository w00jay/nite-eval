"""Pricing and budget-cap behavior."""

import pytest

from nite_eval.cost import BudgetExceededError, CostTracker, PriceBook


def test_known_anthropic_price():
    # 1M in + 1M out on Opus 5 = $5 + $25
    assert PriceBook().cost_usd("claude-opus-5", 1_000_000, 1_000_000) == pytest.approx(30.0)


def test_unpriced_model_returns_none_not_zero():
    assert PriceBook().cost_usd("some-unlisted-model", 100, 100) is None


def test_local_provider_is_free_even_when_unlisted():
    assert PriceBook().cost_usd("qwen3.6-35b-a3b", 5000, 5000, provider="local") == 0.0
    assert PriceBook().cost_usd("qwen3.6-35b-a3b", 5000, 5000, provider="llama.cpp") == 0.0


def test_config_overrides_and_adds_prices():
    book = PriceBook({"gpt-x": {"input_per_mtok": 2.0, "output_per_mtok": 8.0}})
    assert book.cost_usd("gpt-x", 1_000_000, 1_000_000) == pytest.approx(10.0)


def test_tracker_accumulates_per_model():
    tracker = CostTracker()
    tracker.record("claude-opus-5", 1_000_000, 0)
    tracker.record("claude-opus-5", 1_000_000, 0)
    assert tracker.spent_usd == pytest.approx(10.0)
    assert tracker.by_model["claude-opus-5"] == pytest.approx(10.0)


def test_tracker_flags_unpriced_models_without_charging():
    tracker = CostTracker()
    assert tracker.record("mystery-model", 1_000_000, 1_000_000) == 0.0
    assert tracker.spent_usd == 0.0
    assert "mystery-model" in tracker.unpriced_models


def test_cap_raises_once_reached():
    tracker = CostTracker(cap_usd=4.0)
    tracker.record("claude-opus-5", 500_000, 0)  # $2.50
    tracker.check()  # under cap, no raise
    tracker.record("claude-opus-5", 500_000, 0)  # $5.00 total
    with pytest.raises(BudgetExceededError, match=r"\$5.00 reached cap \$4.00"):
        tracker.check()


def test_remaining_budget():
    assert CostTracker().remaining_usd() is None
    tracker = CostTracker(cap_usd=10.0)
    tracker.record("claude-opus-5", 1_000_000, 0)
    assert tracker.remaining_usd() == pytest.approx(5.0)
