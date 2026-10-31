"""Tests for observe: one researched name scored against SPY after a horizon."""

from __future__ import annotations

import math
from datetime import date

import pandas as pd
import pytest

from hedge_fund.domain.research import NameResearch, Verdict
from hedge_fund.domain.shortlist import Candidate, Side
from hedge_fund.domain.verification import ExclusionReason, Observation, ObservationStatus
from hedge_fund.engine.verification.observe import observe

DAYS = pd.bdate_range("2026-09-07", periods=10)  # Mon 09-07 .. Fri 09-18
CAL = [stamp.date() for stamp in DAYS]
SPY = pd.Series([100.0 + step for step in range(10)], index=DAYS)  # 5-day return 5% from day 0
UP = pd.Series([100.0, 101, 102, 103, 104, 110, 111, 112, 113, 120], index=DAYS)
DOWN = pd.Series([100.0, 99, 98, 97, 96, 90, 89, 88, 87, 80], index=DAYS)


def _name(
    side: Side = Side.LONG,
    mean: float | None = 1.0,
    verdict: Verdict = Verdict.CONFIRM,
    valid: int = 3,
) -> NameResearch:
    return NameResearch(
        candidate=Candidate("AAA", "Tech", side, 1, 0.5),
        runs=(),
        valid=valid,
        mean=mean,
        std=0.5,
        agreement=0.8,
        verdict=verdict,
    )


def _observe(
    name: NameResearch | None = None,
    as_of: date = CAL[0],
    closes: pd.Series = UP,
    spy: pd.Series = SPY,
    horizon: int = 5,
    executed: bool = False,
    calendar: list[date] = CAL,
) -> Observation:
    return observe(name or _name(), as_of, closes, spy, calendar, horizon, executed)


def test_a_scored_observation_carries_the_research_and_the_returns() -> None:
    row = _observe()
    assert (row.as_of, row.ticker, row.sector, row.side) == (CAL[0], "AAA", "Tech", Side.LONG)
    assert (row.verdict, row.mean, row.std, row.agreement, row.valid) == (
        Verdict.CONFIRM,
        1.0,
        0.5,
        0.8,
        3,
    )
    assert (row.horizon, row.base_date, row.end_date) == (5, CAL[0], CAL[5])
    assert row.name_return == pytest.approx(0.10)
    assert row.spy_return == pytest.approx(0.05)
    assert row.aligned_excess == pytest.approx(0.05)
    assert (row.hit, row.executed, row.status, row.reason) == (
        True,
        False,
        ObservationStatus.SCORED,
        None,
    )


@pytest.mark.parametrize(
    ("side", "mean", "closes", "aligned", "hit"),
    [
        pytest.param(Side.LONG, 1.0, UP, 0.05, True, id="long-bullish-beat"),
        pytest.param(Side.LONG, 1.0, DOWN, -0.15, False, id="long-bullish-lagged"),
        pytest.param(Side.LONG, -1.0, DOWN, -0.15, True, id="long-bearish-lagged"),
        pytest.param(Side.LONG, -1.0, UP, 0.05, False, id="long-bearish-beat"),
        pytest.param(Side.SHORT, 1.0, DOWN, 0.15, True, id="short-sell-lagged"),
        pytest.param(Side.SHORT, 1.0, UP, -0.05, False, id="short-sell-beat"),
        pytest.param(Side.SHORT, -1.0, UP, -0.05, True, id="short-buy-beat"),
        pytest.param(Side.SHORT, -1.0, DOWN, 0.15, False, id="short-buy-lagged"),
    ],
)
def test_the_hit_is_the_sign_of_the_aligned_mean_against_the_aligned_excess(
    side: Side, mean: float, closes: pd.Series, aligned: float, hit: bool
) -> None:
    row = _observe(_name(side=side, mean=mean), closes=closes)
    assert row.status is ObservationStatus.SCORED
    assert row.aligned_excess == pytest.approx(aligned)
    assert row.hit is hit


def test_a_mean_of_zero_is_scored_without_a_hit() -> None:
    row = _observe(_name(mean=0.0, verdict=Verdict.NEUTRAL))
    assert row.status is ObservationStatus.SCORED
    assert row.hit is None
    assert row.aligned_excess == pytest.approx(0.05)


@pytest.mark.parametrize("mean", [1.0, -1.0])
def test_a_zero_excess_with_a_call_is_a_miss(mean: float) -> None:
    row = _observe(_name(mean=mean), closes=SPY)
    assert row.aligned_excess == pytest.approx(0.0)
    assert row.hit is False


def test_a_weekend_as_of_uses_the_previous_trading_day_as_the_base() -> None:
    row = _observe(as_of=date(2026, 9, 12))  # Saturday; base Friday 09-11, end exactly 09-18
    assert (row.base_date, row.end_date) == (CAL[4], CAL[9])
    assert row.status is ObservationStatus.SCORED
    assert row.name_return == pytest.approx(120 / 104 - 1)
    assert row.spy_return == pytest.approx(109 / 104 - 1)


def test_an_end_exactly_on_the_last_calendar_date_is_scored() -> None:
    row = _observe(as_of=CAL[4])
    assert (row.end_date, row.status) == (CAL[9], ObservationStatus.SCORED)


@pytest.mark.parametrize(
    ("as_of", "base"),
    [(CAL[5], CAL[5]), (date(2026, 9, 25), CAL[9])],
    ids=["past-the-last-date", "as-of-after-the-calendar"],
)
def test_an_end_past_the_last_calendar_date_is_pending(as_of: date, base: date) -> None:
    row = _observe(as_of=as_of)
    assert (row.status, row.reason, row.base_date, row.end_date) == (
        ObservationStatus.PENDING,
        None,
        base,
        None,
    )
    assert (row.name_return, row.spy_return, row.aligned_excess, row.hit) == (None,) * 4


def _with(position: int, value: float) -> pd.Series:
    series = UP.copy()
    series.iloc[position] = value
    return series


@pytest.mark.parametrize(
    "closes",
    [
        pytest.param(_with(0, math.nan), id="nan-at-base"),
        pytest.param(_with(5, math.nan), id="nan-at-end"),
        pytest.param(_with(5, 0.0), id="zero-at-end"),
        pytest.param(_with(5, math.inf), id="inf-at-end"),
        pytest.param(pd.Series([math.nan] * 10, index=DAYS), id="all-nan"),
    ],
)
def test_a_name_without_a_close_at_base_or_end_is_excluded_for_no_price(
    closes: pd.Series,
) -> None:
    row = _observe(closes=closes)
    assert (row.status, row.reason) == (ObservationStatus.EXCLUDED, ExclusionReason.NO_PRICE)
    assert row.end_date == CAL[5]
    assert (row.name_return, row.spy_return, row.aligned_excess, row.hit) == (None,) * 4


@pytest.mark.parametrize(
    ("verdict", "mean", "as_of"),
    [
        pytest.param(Verdict.INSUFFICIENT, None, CAL[0], id="insufficient-verdict"),
        pytest.param(Verdict.INSUFFICIENT, 0.5, CAL[0], id="insufficient-with-a-mean"),
        pytest.param(Verdict.CONFIRM, None, CAL[9], id="no-mean-not-pending"),
        pytest.param(Verdict.INSUFFICIENT, None, date(2026, 9, 4), id="before-the-calendar"),
    ],
)
def test_insufficient_research_is_excluded_without_a_price_lookup(
    verdict: Verdict, mean: float | None, as_of: date
) -> None:
    nothing = pd.Series([math.nan] * 10, index=DAYS)
    row = _observe(_name(mean=mean, verdict=verdict, valid=1), as_of=as_of, closes=nothing)
    assert (row.status, row.reason) == (
        ObservationStatus.EXCLUDED,
        ExclusionReason.INSUFFICIENT_RESEARCH,
    )
    assert (row.base_date, row.end_date, row.name_return, row.hit) == (None, None, None, None)
    assert (row.mean, row.valid) == (mean, 1)


def test_executed_is_passed_through() -> None:
    assert _observe(executed=True).executed is True


def test_an_as_of_before_the_calendar_raises() -> None:
    with pytest.raises(ValueError, match="no trading day on or before 2026-09-04"):
        _observe(as_of=date(2026, 9, 4))


@pytest.mark.parametrize("short", ["closes", "spy"])
def test_closes_and_spy_need_one_close_per_calendar_day(short: str) -> None:
    kwargs: dict[str, pd.Series] = {short: UP.iloc[:9]}
    with pytest.raises(ValueError, match="one close per calendar day"):
        _observe(**kwargs)


def test_a_missing_benchmark_close_raises() -> None:
    with pytest.raises(ValueError, match="no benchmark close on 2026-09-07"):
        _observe(spy=_with_nan_spy())


def _with_nan_spy() -> pd.Series:
    series = SPY.copy()
    series.iloc[0] = math.nan
    return series
