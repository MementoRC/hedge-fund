"""Trading days, windows and monthly rebalance days from the store's calendar."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from hedge_fund.engine.backtest.calendar import in_window, rebalance_days, trading_days


def test_trading_days_are_the_index_dates():
    index = pd.DatetimeIndex(["2024-01-02", "2024-01-03", "2024-02-01"])
    assert trading_days(index) == (date(2024, 1, 2), date(2024, 1, 3), date(2024, 2, 1))


def test_in_window_is_inclusive():
    days = (date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4))
    assert in_window(days, date(2024, 1, 3), date(2024, 1, 4)) == days[1:]


def test_rebalance_days_are_the_first_day_of_each_month():
    days = (
        date(2024, 1, 2),
        date(2024, 1, 3),
        date(2024, 2, 1),
        date(2024, 2, 2),
        date(2025, 2, 3),
    )
    assert rebalance_days(days) == frozenset({date(2024, 1, 2), date(2024, 2, 1), date(2025, 2, 3)})


def test_a_mid_month_window_waits_for_the_next_month():
    days = (date(2024, 1, 2), date(2024, 1, 16), date(2024, 1, 17), date(2024, 2, 1))
    window = in_window(days, date(2024, 1, 16), date(2024, 2, 1))
    assert rebalance_days(days) & set(window) == {date(2024, 2, 1)}


def test_trading_days_must_be_sorted_and_unique():
    with pytest.raises(ValueError, match="ascending"):
        trading_days(pd.DatetimeIndex(["2024-01-03", "2024-01-02"]))
