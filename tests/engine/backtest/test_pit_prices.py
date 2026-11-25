"""Point-in-time prices: splits after as_of undone, dividends up to as_of back-adjusted."""

from __future__ import annotations

import math
from datetime import date
from fractions import Fraction

import numpy as np
import pandas as pd
import pytest

from hedge_fund.domain.backtest import Dividend
from hedge_fund.domain.ledger import Split
from hedge_fund.engine.backtest.pit_prices import closes_as_of, dividend_factors, unwound

DAYS = pd.bdate_range("2024-01-01", periods=5)  # Monday 1 to Friday 5 January 2024
FROZEN = pd.Series([50.0, 51.0, 51.0, 52.0, 53.0], index=DAYS)  # raw 100, 102, 51, 52, 53
SPLITS = (Split("AAA", date(2024, 1, 3), Fraction(2)),)
DIVIDENDS = (Dividend("AAA", date(2024, 1, 5), 1.04),)  # f = 1 - 1.04 / 52 = 0.98


def _values(series: pd.Series) -> list[float]:
    return [round(value, 10) for value in series.tolist()]


def test_before_the_split_the_rows_are_on_the_pre_split_basis() -> None:
    seen = closes_as_of(FROZEN, "AAA", SPLITS, DIVIDENDS, date(2024, 1, 2))
    assert list(seen.index) == list(DAYS[:2])
    assert _values(seen) == [100.0, 102.0]


def test_on_the_split_ex_date_the_split_counts_and_a_later_dividend_does_not() -> None:
    seen = closes_as_of(FROZEN, "AAA", SPLITS, DIVIDENDS, date(2024, 1, 4))
    assert _values(seen) == [50.0, 51.0, 51.0, 52.0]


def test_on_the_dividend_ex_date_the_earlier_rows_are_back_adjusted() -> None:
    seen = closes_as_of(FROZEN, "AAA", SPLITS, DIVIDENDS, date(2024, 1, 5))
    assert _values(seen) == [49.0, 49.98, 49.98, 50.96, 53.0]


def test_other_tickers_events_are_ignored() -> None:
    splits = (Split("BBB", date(2024, 1, 3), Fraction(2)),)
    dividends = (Dividend("BBB", date(2024, 1, 5), 1.0),)
    seen = closes_as_of(FROZEN, "AAA", splits, dividends, date(2024, 1, 5))
    assert _values(seen) == FROZEN.tolist()


def test_a_dividend_on_the_first_row_has_nothing_to_adjust() -> None:
    first = (Dividend("AAA", date(2024, 1, 1), 1.0),)
    assert dividend_factors(FROZEN, "AAA", first, date(2024, 1, 5)).tolist() == [1.0] * 5


def test_the_prior_close_skips_missing_rows() -> None:
    gappy = FROZEN.copy()
    gappy.iloc[3] = np.nan  # no close on 2024-01-04: the prior close is 2024-01-03's 51
    dividend = (Dividend("AAA", date(2024, 1, 5), 1.02),)  # f = 1 - 1.02 / 51 = 0.98
    factors = dividend_factors(gappy, "AAA", dividend, date(2024, 1, 5))
    assert _values(factors) == [0.98, 0.98, 0.98, 0.98, 1.0]


def test_a_dividend_not_below_the_prior_close_is_rejected() -> None:
    huge = (Dividend("AAA", date(2024, 1, 5), 52.0),)
    with pytest.raises(ValueError, match="not below the prior close"):
        dividend_factors(FROZEN, "AAA", huge, date(2024, 1, 5))


def test_unwound_scales_opens_by_the_later_splits() -> None:
    opens = pd.Series([49.0, 50.5, 50.0, 51.5, 52.5], index=DAYS)
    assert _values(unwound(opens, "AAA", SPLITS, date(2024, 1, 2))) == [98.0, 101.0]


def _events(
    rng: np.random.Generator, raw: pd.Series, window: pd.DatetimeIndex
) -> tuple[tuple[Split, ...], tuple[Dividend, ...]]:
    """One split and two dividends (0.5% of the prior raw close) ex-dated inside ``window``."""
    picks = sorted(rng.choice(len(window), size=3, replace=False))
    split_day, *dividend_days = (window[i] for i in picks)
    ratio = (Fraction(2), Fraction(3), Fraction(1, 4))[int(rng.integers(3))]
    splits = (Split("AAA", split_day.date(), ratio),)
    dividends = tuple(
        Dividend("AAA", day.date(), 0.005 * float(raw.loc[raw.index < day].iloc[-1]))
        for day in dividend_days
    )
    return splits, dividends


def _vendor_view(
    raw: pd.Series, splits: tuple[Split, ...], dividends: tuple[Dividend, ...]
) -> tuple[pd.Series, str, tuple[Split, ...], tuple[Dividend, ...]]:
    """closes_as_of's inputs as yfinance serves them after every event: closes and dividend
    amounts divided by the ratios of the splits ex-dated after their day."""

    def later(on: date) -> float:
        return math.prod(float(split.ratio) for split in splits if split.ex_date > on)

    closes = pd.Series([value / later(day.date()) for day, value in raw.items()], index=raw.index)
    frozen = tuple(Dividend(d.ticker, d.ex_date, d.amount / later(d.ex_date)) for d in dividends)
    return closes, "AAA", splits, frozen


@pytest.mark.parametrize("seed", range(25))
def test_no_event_after_as_of_changes_what_as_of_sees(seed: int) -> None:
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2024-01-01", periods=60)
    raw = pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, 60))), index=days)
    as_of = days[39].date()
    early_splits, early_dividends = _events(rng, raw, days[1:40])
    late_splits, late_dividends = _events(rng, raw, days[40:])
    before = closes_as_of(*_vendor_view(raw, early_splits, early_dividends), as_of)
    after = closes_as_of(
        *_vendor_view(raw, early_splits + late_splits, early_dividends + late_dividends), as_of
    )
    pd.testing.assert_series_equal(after, before, rtol=1e-12)
    assert after.iloc[-1] == pytest.approx(raw.iloc[39], rel=1e-12)
