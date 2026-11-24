"""MacroProvider port: daily macro series (FRED) by observation date."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from typing import Protocol

import pandas as pd


class MacroProviderError(Exception):
    """A vendor, transport or parse failure while fetching macro series, or a fetched frame the
    caller rejects as unusable (empty, a required series missing or without a finite value
    before the cut)."""


class MacroProvider(Protocol):
    """Source of macro series. Implementations live in adapters/data/."""

    def series(self, ids: Sequence[str], start: date, end: date) -> pd.DataFrame:
        """Observations of ``ids`` dated start..end inclusive.

        Returns a tz-naive, sorted, unique DatetimeIndex of observation dates (unnamed) and
        one float column per requested id, in request order, with NaN for a missing value.
        Dates are raw: the point-in-time cut is the engine's. Raises MacroProviderError on a
        vendor, transport or parse failure.
        """
