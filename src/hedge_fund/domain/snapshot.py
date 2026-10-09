"""Point-in-time price snapshot: the factor engine's only input."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date

import pandas as pd


@dataclass(frozen=True, eq=False)
class PriceSnapshot:
    """Close prices (dates x tickers) known as of ``as_of``, with a sector per ticker.

    Frozen only prevents field reassignment; the engine must never mutate ``closes``.
    """

    as_of: date
    closes: pd.DataFrame
    sectors: Mapping[str, str]

    def __post_init__(self) -> None:
        closes = self.closes
        if closes.empty:
            raise ValueError("closes is empty")
        if not isinstance(closes.index, pd.DatetimeIndex):
            raise ValueError("closes index must be a DatetimeIndex")
        if not closes.index.is_monotonic_increasing:
            raise ValueError("closes index must be sorted ascending")
        if closes.index.has_duplicates:
            raise ValueError("closes index has duplicate dates")
        if closes.columns.has_duplicates:
            raise ValueError("closes has duplicate tickers")
        if closes.index[-1].date() > self.as_of:
            raise ValueError("closes has rows after as_of")
        missing = sorted(str(t) for t in closes.columns if t not in self.sectors)
        if missing:
            raise ValueError(f"tickers without sector: {missing}")
