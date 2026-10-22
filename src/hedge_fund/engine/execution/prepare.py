"""Order plan -> broker requests: deterministic ids, reducing/opening split, deferred flips.

Pure: no I/O, imports only the domain. ``services/execution.py`` submits the requests.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from hedge_fund.domain.execution import OrderRequest
from hedge_fund.domain.orders import Order, OrderAction, OrderPlan

MAX_CLIENT_ORDER_ID = 128  # Alpaca's client_order_id limit

_REDUCING = frozenset({OrderAction.SELL, OrderAction.BUY_TO_COVER})


@dataclass(frozen=True)
class PreparedSubmission:
    """Requests split by kind, each in plan order, so ``reducing + opening`` is plan order.

    ``deferred`` holds the client_order_ids of flip open legs: opening requests on a ticker
    that also has a reducing request.
    """

    reducing: tuple[OrderRequest, ...]
    opening: tuple[OrderRequest, ...]
    deferred: frozenset[str]


def prepare_submission(plan: OrderPlan) -> PreparedSubmission:
    """Broker requests for ``plan`` with ids ``hf-<as_of>-<ticker>-<action>-<shares>``.

    Reducing orders (SELL, BUY_TO_COVER) go to ``reducing`` and the rest to ``opening``.
    Raises ValueError when a blocked plan has orders, a reducing order follows an opening
    order, two orders share a client_order_id, or an id is longer than MAX_CLIENT_ORDER_ID
    characters. None of these can happen for a plan from ``generate_orders``.
    """
    if plan.blocked is not None and plan.orders:
        raise ValueError(
            f"blocked plan ({plan.blocked}) must have no orders, got {len(plan.orders)}"
        )
    reducing: list[OrderRequest] = []
    opening: list[OrderRequest] = []
    seen: set[str] = set()
    for order in plan.orders:
        client_order_id = _client_order_id(plan.as_of, order)
        if client_order_id in seen:
            raise ValueError(f"duplicate client_order_id {client_order_id!r}")
        if len(client_order_id) > MAX_CLIENT_ORDER_ID:
            raise ValueError(
                f"client_order_id {client_order_id!r} exceeds {MAX_CLIENT_ORDER_ID} characters"
            )
        seen.add(client_order_id)
        request = OrderRequest(order=order, client_order_id=client_order_id)
        if order.action in _REDUCING:
            if opening:
                raise ValueError(f"reducing order {client_order_id!r} follows an opening order")
            reducing.append(request)
        else:
            opening.append(request)
    closing = {request.order.ticker for request in reducing}
    deferred = frozenset(
        request.client_order_id for request in opening if request.order.ticker in closing
    )
    return PreparedSubmission(reducing=tuple(reducing), opening=tuple(opening), deferred=deferred)


def _client_order_id(as_of: date, order: Order) -> str:
    return f"hf-{as_of.isoformat()}-{order.ticker}-{order.action.value}-{order.shares}"
