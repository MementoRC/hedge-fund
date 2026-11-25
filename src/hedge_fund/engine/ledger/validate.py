"""A final broker order status, checked against the request that placed it, as a Fill.

Pure: ``recorded_at`` and ``cycle_as_of`` come from the caller.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from datetime import date, datetime

from hedge_fund.domain.execution import OrderRequest
from hedge_fund.domain.ledger import Fill, FillStatus, OrderState, OrderStatus
from hedge_fund.domain.orders import Order, OrderAction
from hedge_fund.engine.ledger.symbols import canonical_symbol

_FINAL: Mapping[OrderState, FillStatus] = {
    OrderState.FILLED: FillStatus.FILLED,
    OrderState.PARTIAL_TERMINAL: FillStatus.PARTIAL_TERMINAL,
    OrderState.CANCELED: FillStatus.CANCELED,
    OrderState.REJECTED: FillStatus.REJECTED,
    OrderState.EXPIRED: FillStatus.EXPIRED,
}
_FILLING = frozenset({OrderState.FILLED, OrderState.PARTIAL_TERMINAL})
_BUY_SIDE = frozenset({OrderAction.BUY, OrderAction.BUY_TO_COVER})


def to_fill(
    status: OrderStatus, request: OrderRequest, *, cycle_as_of: date, recorded_at: datetime
) -> Fill:
    """The Fill for a final ``status`` of ``request``; ValueError naming the first problem.

    Refused: a state that is not final; a broker symbol whose ``canonical_symbol`` differs
    from the request ticker's; a side other than ``buy`` (BUY, BUY_TO_COVER) or ``sell``
    (SELL, SELL_SHORT); filled shares that are not a whole number, negative or more than
    requested; FILLED / PARTIAL_TERMINAL with no shares or a zero-fill state with some; a
    non-zero fill whose ``avg_price`` is not finite and positive or that has no
    ``filled_at``. A zero fill carries ``avg_price=None`` and ``filled_at=None``. The Fill
    carries the request ticker.
    """
    if status.state not in _FINAL:
        raise ValueError(f"{status.state.value} is not a final state")
    order = request.order
    _check_symbol(status, order)
    _check_side(status, order)
    shares = _filled_shares(status, order)
    price: float | None = None
    filled_at: datetime | None = None
    if shares:
        price, filled_at = _priced(status)
    return Fill(
        client_order_id=request.client_order_id,
        broker_order_id=status.broker_order_id,
        cycle_as_of=cycle_as_of,
        ticker=order.ticker,
        action=order.action,
        requested_shares=order.shares,
        filled_shares=shares,
        avg_price=price,
        filled_at=filled_at,
        status=_FINAL[status.state],
        recorded_at=recorded_at,
    )


def _check_symbol(status: OrderStatus, order: Order) -> None:
    if status.ticker is None or canonical_symbol(status.ticker) != canonical_symbol(order.ticker):
        raise ValueError(
            f"broker symbol {status.ticker!r} does not match the request ticker {order.ticker!r}"
        )


def _check_side(status: OrderStatus, order: Order) -> None:
    expected = "buy" if order.action in _BUY_SIDE else "sell"
    if status.side != expected:
        raise ValueError(f"side {status.side!r} does not match {order.action.value}")


def _filled_shares(status: OrderStatus, order: Order) -> int:
    filled = status.filled_shares
    if not filled.is_finite() or filled != filled.to_integral_value():
        raise ValueError(f"filled_shares {filled} is not a whole number")
    shares = int(filled)
    if shares > order.shares:
        raise ValueError(f"filled_shares {shares} exceeds the {order.shares} requested")
    if shares < 0 or (shares > 0) != (status.state in _FILLING):
        raise ValueError(f"{status.state.value} with {shares} filled shares")
    return shares


def _priced(status: OrderStatus) -> tuple[float, datetime]:
    price = status.avg_price
    if price is None or not math.isfinite(price) or price <= 0:
        raise ValueError(f"avg_price {price!r} is not a finite positive price")
    if status.filled_at is None:
        raise ValueError("a non-zero fill has no filled_at")
    return price, status.filled_at
