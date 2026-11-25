"""Tests for pricing LLM usage and summing a cycle's spend."""

from __future__ import annotations

from datetime import date

import pytest

from hedge_fund.domain.cost import ModelPrice, ModelUsage, PriceTable, Spend
from hedge_fund.domain.research import Rating, RunRecord
from hedge_fund.engine.cost import merge_usage, price, spend

OPUS = ModelPrice(input=15.0, output=75.0, cache_write=18.75, cache_read=1.5)
SONNET = ModelPrice(input=3.0, output=15.0, cache_write=3.75, cache_read=0.3)
TABLE = PriceTable(models={"opus": OPUS, "sonnet": SONNET})


def _usage(
    model: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_write: int = 0,
    cache_read: int = 0,
    *,
    calls: int = 1,
) -> ModelUsage:
    return ModelUsage(
        model=model,
        calls=calls,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_write_tokens=cache_write,
        cache_read_tokens=cache_read,
    )


def _record(usage: tuple[ModelUsage, ...] | None, error: str | None = None) -> RunRecord:
    return RunRecord(
        ticker="AAA",
        as_of=date(2026, 9, 30),
        run_index=0,
        rating=None if error else Rating.BUY,
        decision="",
        reports={},
        seconds=1.0,
        error=error,
        usage=usage,
    )


# --- price -------------------------------------------------------------------------------


def test_input_and_output_are_priced_per_million_tokens() -> None:
    assert price((_usage("opus", 1_000_000, 100_000),), TABLE) == pytest.approx(15.0 + 7.5)


def test_cached_tokens_leave_input_and_are_priced_at_their_own_rates() -> None:
    usage = (_usage("sonnet", 1_000_000, cache_write=200_000, cache_read=300_000),)
    # 500k uncached x 3.0 + 200k written x 3.75 + 300k read x 0.3, per million
    assert price(usage, TABLE) == pytest.approx(1.5 + 0.75 + 0.09)


def test_uncached_input_is_floored_at_zero() -> None:
    usage = (_usage("sonnet", 100, cache_write=200, cache_read=300),)
    assert price(usage, TABLE) == pytest.approx((200 * 3.75 + 300 * 0.3) / 1e6)


def test_models_are_summed() -> None:
    usage = (_usage("opus", 1_000_000), _usage("sonnet", 0, 1_000_000))
    assert price(usage, TABLE) == pytest.approx(15.0 + 15.0)


def test_no_calls_cost_nothing() -> None:
    assert price((), TABLE) == 0.0


@pytest.mark.parametrize(
    "usage",
    [
        pytest.param(None, id="not-captured"),
        pytest.param((_usage("haiku", 10),), id="unpriced-model"),
        pytest.param((_usage("opus", 10), _usage("unknown", 10)), id="unknown-model"),
    ],
)
def test_uncaptured_usage_or_an_unpriced_model_is_an_unknown_cost(
    usage: tuple[ModelUsage, ...] | None,
) -> None:
    assert price(usage, TABLE) is None


# --- merge_usage -------------------------------------------------------------------------


def test_merge_usage_sums_per_model_sorted_by_model() -> None:
    rows = [
        _usage("sonnet", 10, 1, 2, 3),
        _usage("opus", 5, 6, 7, 8),
        _usage("sonnet", 1, 1, 1, 1, calls=2),
    ]
    assert merge_usage(rows) == (
        _usage("opus", 5, 6, 7, 8),
        _usage("sonnet", 11, 2, 3, 4, calls=3),
    )


def test_merge_usage_of_nothing_is_empty() -> None:
    assert merge_usage([]) == ()


# --- spend -------------------------------------------------------------------------------


def test_spend_totals_known_costs_and_counts_unknown_runs() -> None:
    runs = [
        _record((_usage("opus", 1_000_000),)),
        _record(None),
        _record((_usage("haiku", 10),)),
        _record((_usage("opus", 0, 100_000),), error="RuntimeError: boom"),  # billed too
    ]
    out = spend(runs, TABLE, budget=100.0)
    assert out.total == pytest.approx(15.0 + 7.5)
    assert (out.runs, out.unknown_runs, out.over_budget) == (4, 2, False)
    assert out.usage == (_usage("haiku", 10), _usage("opus", 1_000_000, 100_000, calls=2))


@pytest.mark.parametrize(
    ("budget", "over"),
    [
        pytest.param(None, None, id="no-budget"),
        pytest.param(15.0, False, id="equal"),
        pytest.param(14.99, True, id="over"),
    ],
)
def test_over_budget_is_reported_only_with_a_budget(
    budget: float | None, over: bool | None
) -> None:
    out = spend([_record((_usage("opus", 1_000_000),))], TABLE, budget=budget)
    assert out.over_budget is over


def test_spend_over_no_runs_is_zero() -> None:
    assert spend([], TABLE, budget=None) == Spend(
        total=0.0, runs=0, unknown_runs=0, usage=(), over_budget=None
    )
