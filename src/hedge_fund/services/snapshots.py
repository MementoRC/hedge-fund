"""Build validated PriceSnapshots from a DataProvider."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd

from hedge_fund.domain.snapshot import PriceSnapshot
from hedge_fund.ports.data import DataProvider

LOOKBACK_ROWS = 300
CALENDAR_WINDOW = timedelta(days=450)
NO_SECTOR = "no_sector"


@dataclass(frozen=True)
class SnapshotBuild:
    """A snapshot plus the tickers dropped while building it (ticker -> reason, sorted)."""

    snapshot: PriceSnapshot
    unresolved: Mapping[str, str]


def build_snapshot(
    provider: DataProvider,
    tickers: Sequence[str],
    as_of: date,
    lookback: int = LOOKBACK_ROWS,
) -> SnapshotBuild:
    """Fetch closes and sectors for `tickers` and return a snapshot ending at `as_of`.

    Tickers without a sector are dropped into `unresolved`. Tickers without prices stay
    as all-NaN columns; the factor engine excludes them as insufficient_history.
    """
    requested = _normalize(tickers)
    if not requested:
        raise ValueError("no tickers requested")
    closes = provider.closes(requested, as_of - CALENDAR_WINDOW, as_of)
    closes = closes.loc[closes.index <= pd.Timestamp(as_of)].tail(lookback)
    found = provider.sectors(requested)
    sectors: dict[str, str] = {}
    for ticker in requested:
        sector = found.get(ticker)
        if sector is not None:
            sectors[ticker] = sector
    unresolved = {ticker: NO_SECTOR for ticker in sorted(requested) if ticker not in sectors}
    closes = closes.reindex(columns=list(sectors))
    if closes.empty:
        raise ValueError("no price data for the requested tickers at as_of")
    return SnapshotBuild(
        snapshot=PriceSnapshot(as_of=as_of, closes=closes, sectors=sectors),
        unresolved=unresolved,
    )


def _normalize(tickers: Sequence[str]) -> list[str]:
    """Strip, upper-case and de-duplicate, keeping first occurrence order."""
    seen: dict[str, None] = {}
    for raw in tickers:
        ticker = raw.strip().upper()
        if ticker:
            seen.setdefault(ticker, None)
    return list(seen)
