"""Fetch the benchmark and wire research, prices and risk into a target portfolio."""

from __future__ import annotations

from datetime import date

import pandas as pd

from hedge_fund.ports.data import DataProvider
from hedge_fund.services.snapshots import _calendar_window

BENCHMARK = "SPY"


def fetch_benchmark(
    provider: DataProvider, as_of: date, *, lookback_days: int, ticker: str = BENCHMARK
) -> pd.Series:
    """Adjusted closes of ``ticker`` up to ``as_of``, NaN rows dropped.

    The calendar start covers ``lookback_days + 1`` trading rows (one extra close for the
    first return). Raises ValueError when no close is left; the provider returns an all-NaN
    column for an unknown ticker rather than an empty frame.
    """
    frame = provider.closes([ticker], as_of - _calendar_window(lookback_days + 1), as_of)
    closes = frame[ticker].dropna()
    closes = closes.loc[closes.index <= pd.Timestamp(as_of)]
    if closes.empty:
        raise ValueError(f"no {ticker!r} closes up to {as_of}")
    return closes
