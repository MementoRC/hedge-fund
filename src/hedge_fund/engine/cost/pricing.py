"""Price LLM usage with the user's table, and sum what a cycle's research runs spent."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

from hedge_fund.domain.cost import ModelUsage, PriceTable, Spend
from hedge_fund.domain.research import RunRecord

TOKENS_PER_UNIT = 1_000_000  # prices are USD per million tokens


def price(usage: Sequence[ModelUsage] | None, table: PriceTable) -> float | None:
    """USD cost of ``usage``; None when it was not captured or a model is not in ``table``.

    LangChain's ``input_tokens`` includes the cached ones, so only
    ``max(0, input - cache_read - cache_write)`` is priced at ``input``; cache reads, cache
    writes and output are priced at their own rates. A model named ``"unknown"`` is never
    guessed.
    """
    if usage is None:
        return None
    total = 0.0
    for row in usage:
        rates = table.models.get(row.model)
        if rates is None:
            return None
        uncached = max(0, row.input_tokens - row.cache_read_tokens - row.cache_write_tokens)
        total += (
            uncached * rates.input
            + row.output_tokens * rates.output
            + row.cache_write_tokens * rates.cache_write
            + row.cache_read_tokens * rates.cache_read
        ) / TOKENS_PER_UNIT
    return total


def merge_usage(rows: Iterable[ModelUsage]) -> tuple[ModelUsage, ...]:
    """``rows`` summed per model (calls and every token count), sorted by model."""
    totals: dict[str, ModelUsage] = {}
    for row in rows:
        seen = totals.get(row.model)
        totals[row.model] = (
            row
            if seen is None
            else ModelUsage(
                model=row.model,
                calls=seen.calls + row.calls,
                input_tokens=seen.input_tokens + row.input_tokens,
                output_tokens=seen.output_tokens + row.output_tokens,
                cache_write_tokens=seen.cache_write_tokens + row.cache_write_tokens,
                cache_read_tokens=seen.cache_read_tokens + row.cache_read_tokens,
            )
        )
    return tuple(totals[model] for model in sorted(totals))


def spend(runs: Iterable[RunRecord], table: PriceTable, budget: float | None) -> Spend:
    """What ``runs`` cost: known costs summed, unknown ones counted, usage summed per model.

    Errored runs count like valid ones (both were billed). ``over_budget`` is
    ``total > budget``, None without a budget; it is reported, never enforced.
    """
    records = list(runs)
    costs = [price(record.usage, table) for record in records]
    known = [cost for cost in costs if cost is not None]
    total = math.fsum(known)
    return Spend(
        total=total,
        runs=len(records),
        unknown_runs=len(costs) - len(known),
        usage=merge_usage(row for record in records for row in record.usage or ()),
        over_budget=None if budget is None else total > budget,
    )
