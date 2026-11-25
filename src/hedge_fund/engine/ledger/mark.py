"""Open lots marked at their ticker's last usable close (the order-price rule), or unmarked.

Pure: ``as_of`` comes from the caller and ``closes`` is never mutated.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date

import pandas as pd

from hedge_fund.domain.ledger import Lot, MarkedLot
from hedge_fund.engine.orders.prices import order_prices


def mark(
    lots: Sequence[Lot], closes: pd.DataFrame, *, as_of: date, max_age_days: int = 7
) -> tuple[tuple[MarkedLot, ...], tuple[Lot, ...]]:
    """(marked lots, unmarked lots), each in ``lots`` order; never raises for a missing price.

    A lot is marked at its ticker's last finite, positive close dated on or before ``as_of``
    and at most ``max_age_days`` calendar days old (``order_prices``' rule); otherwise it is
    unmarked. ``unrealised_pnl = (price - entry_price) * shares``.
    """
    found: dict[str, tuple[float, date] | None] = {}
    marked: list[MarkedLot] = []
    unmarked: list[Lot] = []
    for lot in lots:
        if lot.ticker not in found:
            found[lot.ticker] = _close(closes, lot.ticker, as_of, max_age_days)
        close = found[lot.ticker]
        if close is None:
            unmarked.append(lot)
            continue
        price, day = close
        marked.append(
            MarkedLot(
                lot=lot,
                price=price,
                close_date=day,
                unrealised_pnl=(price - lot.entry_price) * lot.shares,
            )
        )
    return tuple(marked), tuple(unmarked)


def _close(
    closes: pd.DataFrame, ticker: str, as_of: date, max_age_days: int
) -> tuple[float, date] | None:
    try:
        return order_prices(closes, [ticker], as_of=as_of, max_age_days=max_age_days)[ticker]
    except ValueError:  # no usable close: missing column, none finite and positive, or stale
        return None
