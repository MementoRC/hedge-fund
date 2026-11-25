"""One ticker's daily Open, Close and Dividends from yfinance, for the backtest snapshot."""

from __future__ import annotations

from collections.abc import Callable
from datetime import date, timedelta
from typing import Any

import pandas as pd
import yfinance as yf

from hedge_fund.adapters.data._clock import market_today
from hedge_fund.ports.data import DataProviderError

HISTORY_COLUMNS: tuple[str, ...] = ("Open", "Close", "Dividends")


class YFinanceHistory:
    """``auto_adjust=False`` daily history: split-adjusted to the fetch date, no dividends.

    Dividend amounts are on the same split basis as the prices. The index becomes tz-naive
    dates; bars dated on or after the clock's today (New York's date by default) are
    dropped, so an unfinished session never enters the store. ``ticker`` is the yf.Ticker
    factory, injectable so tests never touch the network.
    """

    def __init__(
        self,
        ticker: Callable[[str], Any] = yf.Ticker,
        clock: Callable[[], date] = market_today,
    ) -> None:
        self._ticker = ticker
        self._clock = clock

    def __call__(self, ticker: str, start: date, end: date) -> pd.DataFrame | None:
        """HISTORY_COLUMNS for start..end inclusive, or None when yfinance has no bars.

        Raises DataProviderError on a vendor failure or a frame without HISTORY_COLUMNS.
        """
        try:
            raw = self._ticker(ticker).history(
                start=start, end=end + timedelta(days=1), auto_adjust=False, actions=True
            )
        except Exception as exc:
            raise DataProviderError(f"yfinance history failed for {ticker}: {exc}") from exc
        if raw is None or raw.empty:
            return None
        missing = [column for column in HISTORY_COLUMNS if column not in raw.columns]
        if missing:
            raise DataProviderError(f"yfinance history for {ticker} lacks {missing}")
        index = pd.DatetimeIndex(raw.index)
        if index.tz is not None:
            index = index.tz_localize(None)
        frame = raw.loc[:, list(HISTORY_COLUMNS)].set_axis(index.normalize(), axis=0)
        frame = frame[~frame.index.duplicated(keep="last")].sort_index()
        frame = frame.loc[frame.index < pd.Timestamp(self._clock())].astype("float64")
        return None if frame.empty else frame.rename_axis(index=None, columns=None)
