"""Ledger records in the units of the split-adjusted closes: fetch the splits, then restate."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date

from hedge_fund.domain.ledger import Fill, OpeningEntry
from hedge_fund.engine.ledger.splits import apply_splits, split_window
from hedge_fund.ports.data import DataProvider


def restate_for_splits(
    prices: DataProvider, opening: Sequence[OpeningEntry], fills: Sequence[Fill], end: date
) -> tuple[tuple[OpeningEntry, ...], tuple[Fill, ...]]:
    """``opening`` and ``fills`` in post-split units for the splits up to ``end``.

    Asks ``prices`` for the splits of the recorded tickers after the earliest record date and
    applies them; no records means no request. A DataProviderError propagates, as does the
    ValueError of ``apply_splits``.
    """
    window = split_window(opening, fills)
    if window is None:
        return tuple(opening), tuple(fills)
    tickers, start = window
    return apply_splits(opening, fills, prices.splits(tickers, start, end))
