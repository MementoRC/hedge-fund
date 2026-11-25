"""trading_days (the SPY close dates) and trading_date (rolling a date onto them)."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from hedge_fund.engine.attribution.dating import trading_date, trading_days

NAN = float("nan")
SPY = pd.Series(
    [600.0, 601.0, 602.0, NAN, 603.0, 604.0],
    index=pd.to_datetime(
        ["2026-09-28", "2026-09-29", "2026-09-30", "2026-10-01", "2026-10-02", "2026-10-05"]
    ),
)
ALL = [
    date(2026, 9, 28),
    date(2026, 9, 29),
    date(2026, 9, 30),
    date(2026, 10, 2),
    date(2026, 10, 5),
]


def test_trading_days_are_the_finite_spy_dates_in_the_window() -> None:
    assert trading_days(SPY, date(2026, 9, 29), date(2026, 10, 5)) == ALL[1:]


def test_the_bounds_need_not_be_trading_days() -> None:
    assert trading_days(SPY, date(2026, 10, 3), date(2026, 10, 4)) == []
    assert trading_days(SPY, date(2026, 9, 26), date(2026, 9, 28)) == [date(2026, 9, 28)]


def test_non_positive_closes_are_not_trading_days() -> None:
    spy = SPY.copy()
    spy.iloc[0] = 0.0
    assert trading_days(spy, date(2026, 9, 1), date(2026, 10, 31)) == ALL[1:]


def test_an_unsorted_series_gives_sorted_days() -> None:
    assert trading_days(SPY.iloc[::-1], date(2026, 9, 1), date(2026, 10, 31)) == ALL


CALENDAR = [date(2026, 9, 28), date(2026, 9, 29), date(2026, 10, 1), date(2026, 10, 5)]


@pytest.mark.parametrize(
    ("on", "expected"),
    [
        pytest.param(date(2026, 9, 29), date(2026, 9, 29), id="a-trading-day-stays"),
        pytest.param(date(2026, 9, 30), date(2026, 10, 1), id="a-holiday-rolls-forward"),
        pytest.param(date(2026, 10, 3), date(2026, 10, 5), id="saturday-to-monday"),
        pytest.param(date(2026, 9, 1), date(2026, 9, 28), id="before-the-calendar"),
        pytest.param(date(2026, 10, 6), None, id="after-the-calendar"),
    ],
)
def test_trading_date(on: date, expected: date | None) -> None:
    assert trading_date(on, CALENDAR) == expected
