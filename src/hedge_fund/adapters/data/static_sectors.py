"""Sector decorator that answers from a shipped universe before asking the vendor."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date

import pandas as pd

from hedge_fund.ports.data import DataProvider
from hedge_fund.universe import Universe


class StaticSectorProvider:
    """Universe sectors first; `inner.sectors()` only for tickers outside the universe."""

    def __init__(self, inner: DataProvider, universe: Universe) -> None:
        self._inner = inner
        self._universe = universe

    def closes(self, tickers: Sequence[str], start: date, end: date) -> pd.DataFrame:
        return self._inner.closes(tickers, start, end)

    def sectors(self, tickers: Sequence[str]) -> dict[str, str | None]:
        requested = list(tickers)
        static = self._universe.sectors
        misses = [ticker for ticker in requested if ticker not in static]
        delegated = self._inner.sectors(misses) if misses else {}
        return {
            ticker: static[ticker] if ticker in static else delegated.get(ticker)
            for ticker in requested
        }
