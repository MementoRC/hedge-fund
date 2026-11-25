"""The benchmark equity curve of a frozen store. Pure."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date

import pandas as pd

from hedge_fund.domain.backtest import FrozenData
from hedge_fund.domain.risk import NavHistory, NavPoint
from hedge_fund.engine.backtest.pit_prices import closes_as_of

BENCHMARK = "SPY"


def benchmark_curve(
    data: FrozenData, window: Sequence[date], end: date, ticker: str = BENCHMARK
) -> NavHistory:
    """``ticker``'s vendor-adjusted closes as seen at ``end``, one point per ``window`` day.

    Raises ValueError when the store has no such ticker or a window day has no close.
    """
    if ticker not in data.closes.columns:
        raise ValueError(f"the store has no {ticker} closes")
    series = closes_as_of(data.closes[ticker], ticker, data.splits, data.dividends, as_of=end)
    points = []
    for day in window:
        value = series.get(pd.Timestamp(day))
        if value is None or pd.isna(value):
            raise ValueError(f"{ticker} has no close on {day}")
        points.append(NavPoint(date=day, nav=float(value)))
    return tuple(points)
