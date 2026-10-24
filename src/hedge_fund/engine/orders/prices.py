"""Order sizing prices: each ticker's last finite close, refused when missing or stale.

Pure: ``as_of`` comes from the caller, no clock is read and ``closes`` is never mutated.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date, timedelta

import numpy as np
import pandas as pd


def order_prices(
    closes: pd.DataFrame, tickers: Iterable[str], *, as_of: date, max_age_days: int = 7
) -> dict[str, tuple[float, date]]:
    """Ticker -> (last finite close, that close's date) for every ticker in ``tickers``.

    A ticker that is not a column of ``closes``, whose column has no finite value, or whose
    last finite close is dated before ``as_of - max_age_days`` calendar days has no price
    (a close exactly ``max_age_days`` old is accepted). If any ticker has no price, one
    ValueError names every such ticker, sorted, each stale one with its last close date.
    """
    oldest = as_of - timedelta(days=max_age_days)
    prices: dict[str, tuple[float, date]] = {}
    problems: dict[str, str] = {}
    for ticker in tickers:
        found = _last_close(closes, ticker)
        if found is None:
            problems[ticker] = "no finite close"
        elif found[1] < oldest:
            problems[ticker] = f"last close {found[1]}"
        else:
            prices[ticker] = found
    if problems:
        named = ", ".join(f"{ticker} ({problem})" for ticker, problem in sorted(problems.items()))
        raise ValueError(
            f"no usable order price (missing, or last close before {oldest}) for: {named}"
        )
    return prices


def _last_close(closes: pd.DataFrame, ticker: str) -> tuple[float, date] | None:
    """The finite close with the latest date, or None (absent column, no finite value)."""
    if ticker not in closes.columns:
        return None
    column = closes[ticker]
    finite = column[np.isfinite(column.to_numpy(dtype="float64"))].sort_index()
    if finite.empty:
        return None
    return float(finite.iloc[-1]), finite.index[-1].date()
