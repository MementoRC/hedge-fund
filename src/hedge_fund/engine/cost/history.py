"""The per-run cost estimate: the median of stored run costs, else the table's fallback."""

from __future__ import annotations

import itertools
import statistics
from collections.abc import Iterable, Sequence

from hedge_fund.domain.cost import BudgetError, Estimate, EstimateSource, PriceTable
from hedge_fund.domain.research import ResearchResult
from hedge_fund.engine.cost.pricing import price


def history_costs(
    results: Iterable[ResearchResult], table: PriceTable, window: int = 30
) -> tuple[float, ...]:
    """The known costs of the runs in ``results`` (newest first), at most ``window`` of them.

    Valid and errored runs both count (both were billed). Dropped: a run whose cost is
    unknown (no usage, or an unpriced model), and a run with no calls (``usage == ()``) or a
    priced cost <= 0. Those failed before any LLM call (e.g. a bad API key), and counting
    them would drag the median to 0 and switch the governor off. ``results`` is read
    lazily, so the caller's loads stop once ``window`` costs are found.
    """
    costs = (
        price(record.usage, table)
        for result in results
        for name in result.names
        for record in name.runs
        if record.usage
    )
    billed = (cost for cost in costs if cost is not None and cost > 0)
    return tuple(itertools.islice(billed, window))


def estimate(history: Sequence[float], fallback: float | None) -> Estimate:
    """The median of ``history`` (HISTORY); else ``fallback`` (FALLBACK, 0 samples).

    Raises BudgetError when there is neither.
    """
    if history:
        return Estimate(
            cost=statistics.median(history),
            source=EstimateSource.HISTORY,
            n_samples=len(history),
        )
    if fallback is not None:
        return Estimate(cost=fallback, source=EstimateSource.FALLBACK, n_samples=0)
    raise BudgetError(
        "no run cost history and no fallback_run_cost in the price table: "
        "price the models in prices.toml or set fallback_run_cost"
    )
