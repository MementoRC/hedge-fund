"""The store's trading calendar: days, a window of them, and monthly rebalance days. Pure."""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from datetime import date

import pandas as pd


def trading_days(index: pd.DatetimeIndex) -> tuple[date, ...]:
    """The index's dates, which must be strictly ascending."""
    days = tuple(stamp.date() for stamp in index)
    for before, after in itertools.pairwise(days):
        if not before < after:
            raise ValueError(f"trading days must be strictly ascending, got {after} after {before}")
    return days


def in_window(days: Sequence[date], start: date, end: date) -> tuple[date, ...]:
    """The days with ``start <= day <= end``."""
    return tuple(day for day in days if start <= day <= end)


def rebalance_days(days: Sequence[date]) -> frozenset[date]:
    """The first trading day of each (year, month) in ``days`` (the full store calendar)."""
    firsts: dict[tuple[int, int], date] = {}
    for day in days:
        firsts.setdefault((day.year, day.month), day)
    return frozenset(firsts.values())
