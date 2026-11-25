"""The trading-day calendar (the benchmark's close dates) and rolling a date onto it. Pure."""

from __future__ import annotations

import math
from bisect import bisect_left
from collections.abc import Sequence
from datetime import date

import pandas as pd


def trading_days(benchmark_closes: pd.Series, start: date, end: date) -> list[date]:
    """Sorted dates of ``benchmark_closes``' finite, positive closes with ``start <= d <= end``.

    The bounds need not be trading days.
    """
    index = pd.DatetimeIndex(benchmark_closes.index)
    values = benchmark_closes.to_numpy(dtype="float64")
    days = {
        stamp.date()
        for stamp, value in zip(index, values, strict=True)
        if math.isfinite(value) and value > 0
    }
    return sorted(day for day in days if start <= day <= end)


def trading_date(on: date, calendar: Sequence[date]) -> date | None:
    """The first day of the sorted ``calendar`` on or after ``on``; None after its last day.

    A non-trading day rolls to the next trading day; a date before the calendar rolls to
    its first day.
    """
    position = bisect_left(calendar, on)
    return calendar[position] if position < len(calendar) else None
