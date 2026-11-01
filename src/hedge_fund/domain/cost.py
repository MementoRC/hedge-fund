"""LLM cost types: per-model token usage, the user's price table, the run-cost estimate,
the pre-research budget plan and what a cycle spent (Layer 7c)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum


@dataclass(frozen=True)
class ModelUsage:
    """Tokens one model was billed for over ``calls`` LLM calls; every count is >= 0.

    ``input_tokens`` includes the cached ones (LangChain's convention).
    """

    model: str
    calls: int
    input_tokens: int
    output_tokens: int
    cache_write_tokens: int
    cache_read_tokens: int


@dataclass(frozen=True)
class ModelPrice:
    """USD per million tokens, each finite and >= 0."""

    input: float
    output: float
    cache_write: float
    cache_read: float


@dataclass(frozen=True)
class PriceTable:
    """Per-model prices, and the per-run cost assumed without history (None: no fallback)."""

    models: Mapping[str, ModelPrice] = field(default_factory=dict)
    fallback_run_cost: float | None = None


class EstimateSource(StrEnum):
    """Where the per-run cost estimate came from."""

    HISTORY = "history"
    FALLBACK = "fallback"


@dataclass(frozen=True)
class Estimate:
    """The assumed USD cost of one research run, and over how many stored runs."""

    cost: float
    source: EstimateSource
    n_samples: int


@dataclass(frozen=True)
class BudgetPlan:
    """K and N cut, in that order, so ``projected = n × k × estimate.cost`` fits ``budget``.

    ``skipped`` is True when no name fits (research is not run); ``reasons`` explain each cut.
    """

    budget: float
    estimate: Estimate
    requested_n: int
    requested_k: int
    n: int
    k: int
    projected: float
    skipped: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class Spend:
    """What this cycle's research runs cost; reused research made no runs.

    ``total`` sums the known run costs; ``unknown_runs`` could not be priced; ``usage`` is
    summed per model, sorted by model; ``over_budget`` is None without a budget.
    """

    total: float
    runs: int
    unknown_runs: int
    usage: tuple[ModelUsage, ...]
    over_budget: bool | None


class BudgetError(ValueError):
    """The budget cannot be planned: a bad budget, or no history and no fallback cost."""


class ResearchRunError(Exception):
    """A research run failed, possibly after billed LLM calls; ``__cause__`` is the failure.

    The message is the cause's ``"<Type>: <message>"``; ``usage`` is what the run used before
    it failed (None when it was not captured).
    """

    def __init__(self, message: str, usage: tuple[ModelUsage, ...] | None) -> None:
        super().__init__(message)
        self.usage = usage
