"""Restate ledger records in post-split units.

Pure; nothing here is stored. Inputs are not mutated. Prices are split-adjusted, so records
dated before a split's ex-date are scaled to match: shares times ``ratio``, price over it.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence
from datetime import date
from fractions import Fraction

from hedge_fund.domain.ledger import Fill, OpeningEntry, Split
from hedge_fund.engine.execution.window import MARKET_TZ


def apply_splits(
    opening: Sequence[OpeningEntry], fills: Sequence[Fill], splits: Sequence[Split]
) -> tuple[tuple[OpeningEntry, ...], tuple[Fill, ...]]:
    """``opening`` and ``fills`` restated in post-split units, in input order.

    A record of a split's ticker is scaled by every split whose ``ex_date`` is after the
    record's date (``OpeningEntry.on``; the New York date of ``Fill.filled_at``). Fills
    without ``filled_at`` are unchanged. ``requested_shares`` is informational and is floored
    when restated. Raises ValueError for a ``ratio <= 0`` or when a restated opening or
    ``filled_shares`` count is not a whole number.
    """
    for split in splits:
        if split.ratio <= 0:
            raise ValueError(f"split ratio for {split.ticker} on {split.ex_date} must be > 0")
    return (
        tuple(_opening(entry, splits) for entry in opening),
        tuple(_fill(fill, splits) for fill in fills),
    )


def split_window(
    opening: Sequence[OpeningEntry], fills: Sequence[Fill]
) -> tuple[list[str], date] | None:
    """The sorted tickers and the earliest record date that ``apply_splits`` could restate.

    Opening entries count by ``on``, fills with a ``filled_at`` by its New York date. None
    when there is no such record.
    """
    days = [(entry.ticker, entry.on) for entry in opening]
    days += [
        (fill.ticker, fill.filled_at.astimezone(MARKET_TZ).date())
        for fill in fills
        if fill.filled_at is not None
    ]
    if not days:
        return None
    return sorted({ticker for ticker, _ in days}), min(day for _, day in days)


def _factor(ticker: str, on: date, splits: Sequence[Split]) -> Fraction:
    factor = Fraction(1)
    for split in splits:
        if split.ticker == ticker and on < split.ex_date:
            factor *= split.ratio
    return factor


def _shares(shares: int, factor: Fraction, record: str) -> int:
    scaled = shares * factor
    if scaled.denominator != 1:
        raise ValueError(f"{record}: {shares} shares become {scaled} after splits, not whole")
    return int(scaled)


def _opening(entry: OpeningEntry, splits: Sequence[Split]) -> OpeningEntry:
    factor = _factor(entry.ticker, entry.on, splits)
    if factor == 1:
        return entry
    return dataclasses.replace(
        entry,
        shares=_shares(entry.shares, factor, f"opening {entry.ticker}"),
        price=float(entry.price / factor),
    )


def _fill(fill: Fill, splits: Sequence[Split]) -> Fill:
    if fill.filled_at is None:
        return fill
    factor = _factor(fill.ticker, fill.filled_at.astimezone(MARKET_TZ).date(), splits)
    if factor == 1:
        return fill
    record = f"{fill.ticker} {fill.client_order_id}"
    return dataclasses.replace(
        fill,
        requested_shares=math.floor(fill.requested_shares * factor),
        filled_shares=_shares(fill.filled_shares, factor, record),
        avg_price=None if fill.avg_price is None else float(fill.avg_price / factor),
    )
