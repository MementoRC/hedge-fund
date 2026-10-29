"""The ledger on the trading-day calendar: dated trades and the signed holdings at each close.

Pure; inputs are not mutated. End-of-day holdings do not depend on intraday fill order.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime
from typing import cast

from hedge_fund.domain.ledger import Fill, OpeningEntry
from hedge_fund.engine.attribution.dating import trading_date
from hedge_fund.engine.execution.window import MARKET_TZ
from hedge_fund.engine.ledger import canonical_symbol, signed_quantity


@dataclass(frozen=True)
class Trade:
    """Signed ``shares`` of the canonical ``ticker`` at ``price`` on the trading ``day``."""

    ticker: str
    day: date
    shares: int
    price: float


def trades(
    opening: Sequence[OpeningEntry], fills: Sequence[Fill], days: Sequence[date]
) -> list[Trade]:
    """Opening entries (at ``price``, dated ``on``), then fills with shares (at ``avg_price``,
    dated by ``filled_at`` in New York), each rolled onto the sorted calendar ``days``.

    A date before the calendar rolls to its first day; one after its last day is left out.
    ``signed_quantity`` gives the sign and raises ValueError for a counted fill without
    ``avg_price`` or ``filled_at``.
    """
    out: list[Trade] = []
    for entry in opening:
        _add(out, entry.ticker, trading_date(entry.on, days), entry.shares, entry.price)
    for fill in fills:
        shares = signed_quantity(fill)
        if shares:
            # signed_quantity refused a counted fill without avg_price or filled_at
            local = cast(datetime, fill.filled_at).astimezone(MARKET_TZ).date()
            _add(out, fill.ticker, trading_date(local, days), shares, cast(float, fill.avg_price))
    return out


def _add(out: list[Trade], ticker: str, day: date | None, shares: int, price: float) -> None:
    if day is not None:
        out.append(Trade(canonical_symbol(ticker), day, shares, price))


def holdings(
    opening: Sequence[OpeningEntry], fills: Sequence[Fill], days: Sequence[date]
) -> dict[date, dict[str, int]]:
    """Signed shares per canonical ticker at each close of ``days`` (every calendar day is a
    key), from ``trades``; zero holdings are left out, tickers sorted.
    """
    moves: dict[date, dict[str, int]] = {}
    for trade in trades(opening, fills, days):
        on_day = moves.setdefault(trade.day, {})
        on_day[trade.ticker] = on_day.get(trade.ticker, 0) + trade.shares
    book: dict[str, int] = {}
    out: dict[date, dict[str, int]] = {}
    for day in days:
        for ticker, shares in moves.get(day, {}).items():
            book[ticker] = book.get(ticker, 0) + shares
        out[day] = {ticker: shares for ticker, shares in sorted(book.items()) if shares}
    return out
