"""The benchmark curve: SPY's vendor-adjusted closes on the window days. Pure."""

from __future__ import annotations

import dataclasses
from datetime import date

import pandas as pd
import pytest

from hedge_fund.engine.backtest.benchmark import BENCHMARK, benchmark_curve
from tests.factories import make_frozen_data


def test_the_curve_is_spy_closes_on_each_window_day() -> None:
    data = make_frozen_data()
    window = [stamp.date() for stamp in data.closes.index[:3]]

    curve = benchmark_curve(data, window, window[-1])

    assert [point.date for point in curve] == window
    assert [point.nav for point in curve] == pytest.approx(list(data.closes[BENCHMARK].iloc[:3]))


def test_a_store_without_spy_is_refused() -> None:
    data = make_frozen_data()
    data = dataclasses.replace(
        data, opens=data.opens.drop(columns="SPY"), closes=data.closes.drop(columns="SPY")
    )

    with pytest.raises(ValueError, match="the store has no SPY closes"):
        benchmark_curve(data, [date(2024, 1, 2)], date(2024, 1, 2))


def test_a_window_day_without_a_close_is_refused() -> None:
    data = make_frozen_data()
    saturday = pd.Timestamp("2024-01-06").date()

    with pytest.raises(ValueError, match=r"SPY has no close on 2024-01-06"):
        benchmark_curve(data, [saturday], saturday)
