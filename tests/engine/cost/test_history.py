"""Tests for the run-cost history and the per-run estimate."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import date

import pytest

from hedge_fund.domain.cost import (
    BudgetError,
    Estimate,
    EstimateSource,
    ModelPrice,
    ModelUsage,
    PriceTable,
)
from hedge_fund.domain.research import NameResearch, Rating, ResearchResult, RunRecord, Verdict
from hedge_fund.domain.shortlist import Candidate, Side
from hedge_fund.engine.cost import estimate, history_costs

TABLE = PriceTable(models={"m": ModelPrice(input=1.0, output=0.0, cache_write=0.0, cache_read=0.0)})
CANDIDATE = Candidate(ticker="AAA", sector="Tech", side=Side.LONG, depth=1, composite=1.0)
NEWER, OLDER = date(2026, 9, 29), date(2026, 9, 28)


def _run(index: int, cost: float | None, error: str | None) -> RunRecord:
    """A run costing ``cost`` USD under TABLE (1 USD per million input tokens); None: unknown."""
    usage = None if cost is None else (ModelUsage("m", 1, round(cost * 1_000_000), 0, 0, 0),)
    rating = None if error else Rating.BUY
    return RunRecord("AAA", NEWER, index, rating, "", {}, 1.0, error, usage)


def _result(day: date, *costs: float | None, error_at: int | None = None) -> ResearchResult:
    runs = tuple(
        _run(index, cost, "RuntimeError: x" if index == error_at else None)
        for index, cost in enumerate(costs)
    )
    name = NameResearch(CANDIDATE, runs, 0, None, None, None, Verdict.INSUFFICIENT)
    return ResearchResult(as_of=day, names=(name,))


# --- history_costs -----------------------------------------------------------------------


def test_costs_follow_the_results_order_and_drop_unknown_runs() -> None:
    results = [_result(NEWER, 1.0, None, 2.0), _result(OLDER, 3.0)]
    assert history_costs(results, TABLE) == (1.0, 2.0, 3.0)


def test_errored_runs_are_included() -> None:
    assert history_costs([_result(OLDER, 1.0, 4.0, error_at=1)], TABLE) == (1.0, 4.0)


def test_at_most_window_costs_are_kept() -> None:
    results = [_result(NEWER, 1.0, 2.0), _result(OLDER, 3.0)]
    assert history_costs(results, TABLE, window=2) == (1.0, 2.0)
    assert len(history_costs([_result(OLDER, *[1.0] * 31)], TABLE)) == 30  # default window


def test_reading_stops_once_the_window_is_full() -> None:
    def results() -> Iterator[ResearchResult]:
        yield _result(NEWER, 1.0, 2.0)
        raise AssertionError("read past the window")

    assert history_costs(results(), TABLE, window=2) == (1.0, 2.0)


def test_no_results_no_costs() -> None:
    assert history_costs([], TABLE) == ()


def test_runs_that_cost_nothing_are_skipped() -> None:
    # A run that failed before any LLM call (bad API key): no calls, or zero tokens.
    no_calls = RunRecord("AAA", NEWER, 2, None, "", {}, 0.0, "RuntimeError: bad API key", ())
    runs = (_run(0, 0.0, "RuntimeError: bad API key"), _run(1, 1.0, None), no_calls)
    name = NameResearch(CANDIDATE, runs, 1, None, None, None, Verdict.INSUFFICIENT)
    assert history_costs([ResearchResult(as_of=OLDER, names=(name,))], TABLE) == (1.0,)


def test_runs_priced_by_an_unknown_model_drop_out_of_the_history() -> None:
    other = RunRecord(
        "AAA", NEWER, 1, Rating.BUY, "", {}, 1.0, None, (ModelUsage("other", 1, 5, 0, 0, 0),)
    )
    name = NameResearch(
        CANDIDATE, (_run(0, 1.0, None), other), 0, None, None, None, Verdict.INSUFFICIENT
    )
    assert history_costs([ResearchResult(as_of=OLDER, names=(name,))], TABLE) == (1.0,)


# --- estimate ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("history", "cost"),
    [
        pytest.param((3.0, 1.0, 2.0), 2.0, id="odd"),
        pytest.param((1.0, 4.0), 2.5, id="even"),
        pytest.param((0.7,), 0.7, id="one"),
    ],
)
def test_the_estimate_is_the_median_of_the_history(history: tuple[float, ...], cost: float) -> None:
    assert estimate(history, fallback=9.0) == Estimate(
        cost=cost, source=EstimateSource.HISTORY, n_samples=len(history)
    )


def test_without_history_the_fallback_is_used() -> None:
    assert estimate((), fallback=0.8) == Estimate(
        cost=0.8, source=EstimateSource.FALLBACK, n_samples=0
    )


def test_without_history_or_fallback_it_is_a_budget_error() -> None:
    with pytest.raises(BudgetError, match="no run cost history and no fallback_run_cost"):
        estimate((), fallback=None)
