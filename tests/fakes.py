"""In-memory DataProvider and yfinance-shaped frames for data-layer tests."""

from collections.abc import Mapping, Sequence
from datetime import date
from typing import Any

import pandas as pd


class FakeProvider:
    """DataProvider serving a canned closes frame and sector map, recording every call.

    Tests may replace `frame` between calls to simulate vendor restatements.
    """

    def __init__(self, frame: pd.DataFrame, sectors: Mapping[str, str | None]) -> None:
        self.frame = frame
        self.sector_map = dict(sectors)
        self.closes_calls: list[tuple[tuple[str, ...], date, date]] = []
        self.sectors_calls: list[tuple[str, ...]] = []

    def closes(self, tickers: Sequence[str], start: date, end: date) -> pd.DataFrame:
        requested = tuple(tickers)
        self.closes_calls.append((requested, start, end))
        window = self.frame.loc[pd.Timestamp(start) : pd.Timestamp(end)]
        return window.reindex(columns=list(requested))

    def sectors(self, tickers: Sequence[str]) -> dict[str, str | None]:
        requested = tuple(tickers)
        self.sectors_calls.append(requested)
        return {ticker: self.sector_map.get(ticker) for ticker in requested}


class RecordingDownload:
    """Stand-in for yf.download: returns a fixed frame (or raises) and records each call."""

    def __init__(self, result: pd.DataFrame | None = None, error: Exception | None = None) -> None:
        self.result = result
        self.error = error
        self.calls: list[tuple[list[str], dict[str, Any]]] = []

    def __call__(self, tickers: list[str], **kwargs: Any) -> pd.DataFrame | None:
        self.calls.append((list(tickers), kwargs))
        if self.error is not None:
            raise self.error
        return self.result


def yf_multi(closes: pd.DataFrame) -> pd.DataFrame:
    """yf.download shape for several tickers: (Price, Ticker) MultiIndex columns."""
    return pd.concat({"Close": closes, "Open": closes * 0.99}, axis=1, names=["Price", "Ticker"])


def yf_single(closes: pd.Series) -> pd.DataFrame:
    """Flat frame (one column per price field) as yfinance returns with multi_level_index=False.

    Not yfinance's default, which is a (Price, Ticker) MultiIndex even for a single ticker.
    """
    return pd.DataFrame({"Open": closes * 0.99, "Close": closes, "Volume": 1000.0})


def assert_closes_equal(actual: pd.DataFrame, expected: pd.DataFrame) -> None:
    """Frame equality ignoring index frequency and datetime resolution (parquet may change both)."""
    pd.testing.assert_frame_equal(
        actual.set_axis(pd.DatetimeIndex(actual.index).as_unit("ns")),
        expected.set_axis(pd.DatetimeIndex(expected.index).as_unit("ns")),
        check_freq=False,
    )
