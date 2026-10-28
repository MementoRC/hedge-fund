"""Lots against the broker: every ticker whose summed lot shares differ from its position."""

from __future__ import annotations

from collections.abc import Iterable

from hedge_fund.domain.ledger import Lot, ReconciliationBreak
from hedge_fund.domain.orders import Holding
from hedge_fund.engine.ledger.symbols import canonical_symbol


def reconcile(
    lots: Iterable[Lot], positions: Iterable[Holding], *, exclude: Iterable[str]
) -> tuple[ReconciliationBreak, ...]:
    """Breaks over the union of lot and broker tickers, sorted by canonical ticker.

    Lot tickers, broker symbols and ``exclude`` are compared by ``canonical_symbol``; a side
    without a ticker counts 0 shares, so lots netting to zero with no position are not a
    break. Tickers in ``exclude`` (those of pending and unsettled orders) are skipped.
    """
    ledger = _totals((lot.ticker, lot.shares) for lot in lots)
    broker = _totals((holding.ticker, holding.shares) for holding in positions)
    skipped = {canonical_symbol(ticker) for ticker in exclude}
    return tuple(
        ReconciliationBreak(
            ticker=ticker, ledger_shares=ledger.get(ticker, 0), broker_shares=broker.get(ticker, 0)
        )
        for ticker in sorted((ledger.keys() | broker.keys()) - skipped)
        if ledger.get(ticker, 0) != broker.get(ticker, 0)
    )


def _totals(pairs: Iterable[tuple[str, int]]) -> dict[str, int]:
    totals: dict[str, int] = {}
    for ticker, shares in pairs:
        key = canonical_symbol(ticker)
        totals[key] = totals.get(key, 0) + shares
    return totals
