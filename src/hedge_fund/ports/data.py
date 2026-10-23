"""DataProvider port: adjusted daily closes and GICS sectors for tickers."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from typing import Protocol

import pandas as pd


class DataProviderError(Exception):
    """A vendor or transport failure while fetching data."""


class DataProvider(Protocol):
    """Source of adjusted closes and sectors. Implementations live in adapters/data/."""

    def closes(self, tickers: Sequence[str], start: date, end: date) -> pd.DataFrame:
        """Dividend- and split-adjusted closes for start..end inclusive.

        Returns a tz-naive, sorted, unique DatetimeIndex and one column per requested
        ticker, in request order. A ticker with no data is an all-NaN column, not an
        error. Raises DataProviderError on vendor or transport failure.
        """

    def sectors(self, tickers: Sequence[str]) -> dict[str, str | None]:
        """GICS sector name per requested ticker, or None when unknown."""
