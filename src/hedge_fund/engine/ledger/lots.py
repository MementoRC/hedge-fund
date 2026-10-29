"""Signed FIFO lots from the opening balance and the recorded fills.

Pure and recomputed on every settle; nothing here is stored. Inputs are not mutated.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, time
from typing import cast

from hedge_fund.domain.ledger import ClosedLot, Fill, Lot, OpeningEntry
from hedge_fund.domain.orders import OrderAction
from hedge_fund.engine.execution.window import MARKET_TZ

OPENING_SOURCE = "opening"

_BUYS = frozenset({OrderAction.BUY, OrderAction.BUY_TO_COVER})
_REDUCING = frozenset({OrderAction.SELL, OrderAction.BUY_TO_COVER})


@dataclass(frozen=True)
class _Trade:
    """A fill with shares: its price, its instant and its signed share delta."""

    fill: Fill
    price: float
    at: datetime
    delta: int


def build_lots(
    opening: Sequence[OpeningEntry], fills: Sequence[Fill]
) -> tuple[tuple[Lot, ...], tuple[ClosedLot, ...]]:
    """(open lots, closed lots) after applying ``fills`` to the ``opening`` lots.

    Opening entries become lots first (``entered_at`` midnight UTC of ``on``, ``source``
    OPENING_SOURCE). Fills with ``filled_shares > 0`` then apply sorted by ``(filled_at,
    reducing before opening, client_order_id)``, reducing = SELL, BUY_TO_COVER. A fill
    closes the opposite-side lots of its ticker oldest first (a partial close splits the
    lot) and opens any remainder as a new lot at ``avg_price``. Open lots are sorted by
    ticker, oldest first; closed lots are in closing order. Raises ValueError unless every
    opening entry's ``on`` is strictly before the New York date of the earliest such fill
    (vacuous without one), or when such a fill has no ``avg_price`` or ``filled_at``.
    """
    trades = _trades(fills)
    _check_opening(opening, trades)
    book: dict[str, list[Lot]] = {}
    for entry in opening:
        book.setdefault(entry.ticker, []).append(
            Lot(
                ticker=entry.ticker,
                shares=entry.shares,
                entry_price=entry.price,
                entered_at=datetime.combine(entry.on, time(), tzinfo=UTC),
                source=OPENING_SOURCE,
            )
        )
    closed: list[ClosedLot] = []
    for trade in trades:
        closed.extend(_apply(book.setdefault(trade.fill.ticker, []), trade))
    return tuple(lot for ticker in sorted(book) for lot in book[ticker]), tuple(closed)


def signed_quantity(fill: Fill) -> int:
    """The signed share delta of ``fill``: ``+filled_shares`` for BUY / BUY_TO_COVER,
    ``-filled_shares`` for SELL / SELL_SHORT, and 0 for a fill with ``filled_shares <= 0``.

    Raises ValueError for a fill with shares but no ``avg_price`` or ``filled_at``.
    """
    if fill.filled_shares <= 0:
        return 0
    if fill.avg_price is None or fill.filled_at is None:
        raise ValueError(
            f"fill {fill.client_order_id!r} has {fill.filled_shares} shares but no "
            "avg_price or filled_at"
        )
    return fill.filled_shares if fill.action in _BUYS else -fill.filled_shares


def _trades(fills: Sequence[Fill]) -> list[_Trade]:
    trades: list[_Trade] = []
    for fill in fills:
        delta = signed_quantity(fill)
        if delta:
            # signed_quantity refused a counted fill without avg_price or filled_at
            price, at = cast(float, fill.avg_price), cast(datetime, fill.filled_at)
            trades.append(_Trade(fill, price, at, delta))
    return sorted(
        trades,
        key=lambda trade: (
            trade.at,
            trade.fill.action not in _REDUCING,
            trade.fill.client_order_id,
        ),
    )


def _check_opening(opening: Sequence[OpeningEntry], trades: Sequence[_Trade]) -> None:
    if not trades:
        return
    first = min(trade.at.astimezone(MARKET_TZ).date() for trade in trades)
    late = sorted(entry.ticker for entry in opening if entry.on >= first)
    if late:
        raise ValueError(
            f"opening entries must be dated before {first}, the New York date of the first "
            f"fill: {', '.join(late)}"
        )


def _apply(lots: list[Lot], trade: _Trade) -> list[ClosedLot]:
    """Close opposite-side ``lots`` oldest first, then open the remainder (``lots`` is the
    builder's own list for the ticker and is updated in place).
    """
    closed: list[ClosedLot] = []
    remaining = trade.delta
    while remaining and lots and (lots[0].shares > 0) != (remaining > 0):
        lot = lots[0]
        size = min(abs(lot.shares), abs(remaining))
        portion = size if lot.shares > 0 else -size
        closed.append(
            ClosedLot(
                ticker=lot.ticker,
                shares=portion,
                entry_price=lot.entry_price,
                entered_at=lot.entered_at,
                source=lot.source,
                exit_price=trade.price,
                exited_at=trade.at,
                exit_client_order_id=trade.fill.client_order_id,
                realised_pnl=(trade.price - lot.entry_price) * portion,
            )
        )
        if portion == lot.shares:
            lots.pop(0)
        else:
            lots[0] = dataclasses.replace(lot, shares=lot.shares - portion)
        remaining += portion
    if remaining:
        lots.append(
            Lot(
                ticker=trade.fill.ticker,
                shares=remaining,
                entry_price=trade.price,
                entered_at=trade.at,
                source=trade.fill.client_order_id,
            )
        )
    return closed
