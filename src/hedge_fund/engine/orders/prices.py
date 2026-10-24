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
    """Ticker -> (last usable close, that close's date) for every ticker in ``tickers``.

    A usable close is finite, positive and dated on or before ``as_of``: closes after
    ``as_of`` and NaN, inf, zero or negative closes are skipped, so a future price is never
    returned. A ticker that is not a column of ``closes``, that has no usable close, or whose
    last usable close is dated before ``as_of - max_age_days`` calendar days has no price
    (a close exactly ``max_age_days`` old is accepted). If any ticker has no price, one
    ValueError names every such ticker, sorted, each stale one with its last close date.
    """
    oldest = as_of - timedelta(days=max_age_days)
    prices: dict[str, tuple[float, date]] = {}
    problems: dict[str, str] = {}
    for ticker in tickers:
        found = _last_close(closes, ticker, as_of)
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


def _last_close(closes: pd.DataFrame, ticker: str, as_of: date) -> tuple[float, date] | None:
    """The finite positive close with the latest date <= as_of, or None (none qualifies)."""
    if ticker not in closes.columns:
        return None
    column = closes[ticker]
    values = column.to_numpy(dtype="float64")
    on_or_before = np.asarray(column.index.date <= as_of)
    usable = column[np.isfinite(values) & (values > 0) & on_or_before].sort_index()
    if usable.empty:
        return None
    return float(usable.iloc[-1]), usable.index[-1].date()
