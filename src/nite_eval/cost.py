"""Token pricing and run budget enforcement for API-backed models.

Local models are free; API models are not. Every generation is priced from a
per-model $/MTok table and accumulated against an optional hard cap that aborts
the run rather than silently spending.

Prices below are Anthropic first-party list rates (as published 2026-06-24).
No OpenAI or third-party rates are hardcoded — they are not verified here, so
those models are reported as UNPRICED (excluded from the total and the cap)
until rates are supplied under `cost.prices` in eval_config.yaml.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelPrice:
    input_per_mtok: float
    output_per_mtok: float


DEFAULT_PRICES: dict[str, ModelPrice] = {
    "claude-fable-5": ModelPrice(10.00, 50.00),
    "claude-opus-5": ModelPrice(5.00, 25.00),
    "claude-opus-4-8": ModelPrice(5.00, 25.00),
    "claude-opus-4-7": ModelPrice(5.00, 25.00),
    "claude-opus-4-6": ModelPrice(5.00, 25.00),
    "claude-sonnet-5": ModelPrice(3.00, 15.00),
    "claude-sonnet-4-6": ModelPrice(3.00, 15.00),
    "claude-haiku-4-5": ModelPrice(1.00, 5.00),
}

FREE_PROVIDERS = frozenset({"local", "llama.cpp", "vllm"})


class BudgetExceededError(RuntimeError):
    """Raised when a run's accumulated API spend hits the configured cap."""


class PriceBook:
    """Resolves a model ID to a price, with config overrides taking priority."""

    def __init__(self, overrides: dict[str, dict[str, float]] | None = None):
        self._prices = dict(DEFAULT_PRICES)
        for model_id, spec in (overrides or {}).items():
            self._prices[model_id] = ModelPrice(
                input_per_mtok=float(spec["input_per_mtok"]),
                output_per_mtok=float(spec["output_per_mtok"]),
            )

    def price_for(self, model_id: str) -> ModelPrice | None:
        return self._prices.get(model_id)

    def cost_usd(self, model_id: str, prompt_tokens: int, completion_tokens: int, provider: str = "") -> float | None:
        """Cost in USD, or None if the model has no known price.

        None is distinct from 0.0: local models genuinely cost nothing, while an
        unpriced API model means the run's total is an undercount.
        """
        if provider in FREE_PROVIDERS:
            return 0.0
        price = self.price_for(model_id)
        if price is None:
            return None
        return (prompt_tokens * price.input_per_mtok + completion_tokens * price.output_per_mtok) / 1_000_000


@dataclass
class CostTracker:
    """Accumulates spend across a run and enforces an optional hard cap."""

    price_book: PriceBook = field(default_factory=PriceBook)
    cap_usd: float | None = None
    spent_usd: float = 0.0
    by_model: dict[str, float] = field(default_factory=dict)
    unpriced_models: set[str] = field(default_factory=set)

    def record(self, model_id: str, prompt_tokens: int, completion_tokens: int, provider: str = "") -> float:
        """Price one generation, add it to the running total, and return its cost."""
        cost = self.price_book.cost_usd(model_id, prompt_tokens, completion_tokens, provider)
        if cost is None:
            if model_id not in self.unpriced_models:
                logger.warning(
                    "No price configured for %s — its spend is not counted toward the budget cap. "
                    "Set cost.prices['%s'] in eval_config.yaml.",
                    model_id,
                    model_id,
                )
            self.unpriced_models.add(model_id)
            return 0.0
        self.spent_usd += cost
        self.by_model[model_id] = self.by_model.get(model_id, 0.0) + cost
        return cost

    def check(self) -> None:
        """Raise BudgetExceededError if the cap has been reached."""
        if self.cap_usd is not None and self.spent_usd >= self.cap_usd:
            raise BudgetExceededError(f"API spend ${self.spent_usd:.2f} reached cap ${self.cap_usd:.2f}")

    def remaining_usd(self) -> float | None:
        if self.cap_usd is None:
            return None
        return max(0.0, self.cap_usd - self.spent_usd)
