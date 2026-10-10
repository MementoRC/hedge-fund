"""DataProvider backed by yfinance (Yahoo Finance)."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import date, timedelta
from typing import Any

import pandas as pd
import yfinance as yf

from hedge_fund.ports.data import DataProviderError

YAHOO_TO_GICS: Mapping[str, str] = {
    "Technology": "Information Technology",
    "Financial Services": "Financials",
    "Healthcare": "Health Care",
    "Consumer Cyclical": "Consumer Discretionary",
    "Consumer Defensive": "Consumer Staples",
    "Communication Services": "Communication Services",
    "Industrials": "Industrials",
    "Energy": "Energy",
    "Utilities": "Utilities",
    "Real Estate": "Real Estate",
    "Basic Materials": "Materials",
}

Download = Callable[..., pd.DataFrame | None]
InfoFetcher = Callable[[str], Mapping[str, Any] | None]


class YFinanceProvider:
    """Adjusted daily closes and GICS sectors from Yahoo Finance.

    Vendor calls are injectable so tests never touch the network. yfinance reports
    per-ticker failures as empty data, which surfaces here as all-NaN columns.
    Bars dated on or after the clock's today are dropped, even after the close, so a
    session's unfinished bar is never returned.
    """

    def __init__(
        self,
        download: Download = yf.download,
        info_fetcher: InfoFetcher = lambda ticker: yf.Ticker(ticker).info,
        clock: Callable[[], date] = date.today,
    ) -> None:
        self._download = download
        self._info_fetcher = info_fetcher
        self._clock = clock

    def closes(self, tickers: Sequence[str], start: date, end: date) -> pd.DataFrame:
        requested = list(tickers)
        if not requested:
            return pd.DataFrame(index=pd.DatetimeIndex([]))
        try:
            raw = self._download(
                requested,
                start=start,
                end=end + timedelta(days=1),
                auto_adjust=True,
                actions=False,
                progress=False,
            )
        except Exception as exc:
            raise DataProviderError(f"yfinance download failed for {requested}: {exc}") from exc
        closes = _normalize_closes(raw, requested)
        return closes.loc[closes.index < pd.Timestamp(self._clock())]

    def sectors(self, tickers: Sequence[str]) -> dict[str, str | None]:
        result: dict[str, str | None] = {}
        for ticker in tickers:
            try:
                info = self._info_fetcher(ticker) or {}
            except Exception as exc:
                raise DataProviderError(f"yfinance info failed for {ticker}: {exc}") from exc
            sector = info.get("sector")
            result[ticker] = YAHOO_TO_GICS.get(sector) if isinstance(sector, str) else None
        return result


def _normalize_closes(raw: pd.DataFrame | None, tickers: list[str]) -> pd.DataFrame:
    """Turn any yf.download result into a dates x tickers frame in request order."""
    if raw is None or raw.empty:
        return pd.DataFrame(index=pd.DatetimeIndex([]), columns=tickers, dtype="float64")
    if isinstance(raw.columns, pd.MultiIndex):
        closes = raw["Close"]
    elif len(tickers) == 1:
        closes = raw[["Close"]].set_axis(tickers, axis=1)
    else:
        raise DataProviderError(f"unexpected flat yfinance frame for {len(tickers)} tickers")
    index = pd.DatetimeIndex(closes.index)
    if index.tz is not None:
        index = index.tz_localize(None)
    closes = closes.set_axis(index.normalize(), axis=0)
    closes = closes[~closes.index.duplicated(keep="last")].sort_index()
    return closes.reindex(columns=tickers).astype("float64").rename_axis(index=None, columns=None)
