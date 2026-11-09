"""Simulated broker for the offline backtest: an order book and cash/share accounting over a
FrozenData store.

Orders queue PENDING and fill at the next open. The driver calls ``open(d)`` then
``close(d)`` once per trading day, in order; ``SimulationError`` marks a misdriven
simulation (a bug, not a broker outcome), so it is not a BrokerError.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, date, datetime, time
from decimal import Decimal
from typing import Literal, cast
from zoneinfo import ZoneInfo

import pandas as pd

from hedge_fund.domain.backtest import FrozenData, SimCosts, SimEvent, SimEventKind
from hedge_fund.domain.execution import Account, OrderRequest, SubmitResult, SubmitStatus
from hedge_fund.domain.ledger import OrderState, OrderStatus
from hedge_fund.domain.orders import Holding, OrderAction

_MARKET_OPEN = time(9, 30)
_MARKET_TZ = ZoneInfo("America/New_York")
_ORDER_SIDE = {
    OrderAction.BUY: "buy",
    OrderAction.BUY_TO_COVER: "buy",
    OrderAction.SELL: "sell",
    OrderAction.SELL_SHORT: "sell",
}


class SimulationError(Exception):
    """The simulation was driven wrongly (bad cash, or days out of order)."""


@dataclass
class _SimOrder:
    request: OrderRequest
    broker_order_id: str
    state: OrderState
    fill_price: float | None
    filled_on: date | None
    submitted_on: date | None


class SimulatedBroker:
    """Paper broker over frozen prices; implements the Broker port.

    State: ``cash``, signed ``shares`` per ticker, ``marks`` (raw close of the last close
    mark), ``orders`` by client_order_id in submission order, and the ``events`` log.
    ``day`` is the last opened day and ``phase`` is ``"open"`` between ``open(d)`` and
    ``close(d)``, otherwise ``"closed"``.
    """

    def __init__(self, data: FrozenData, cash: float, costs: SimCosts = SimCosts()) -> None:  # noqa: B008
        if not math.isfinite(cash) or cash <= 0:
            raise SimulationError(f"cash must be finite and > 0, got {cash}")
        self._data = data
        self._costs = costs
        self.cash = cash
        self.shares: dict[str, int] = {}
        self.marks: dict[str, float] = {}
        self.orders: dict[str, _SimOrder] = {}
        self._events: list[SimEvent] = []
        self.day: date | None = None
        self.phase: Literal["open", "closed"] = "closed"

    @property
    def events(self) -> tuple[SimEvent, ...]:
        return tuple(self._events)

    def open(self, day: date) -> None:
        """Start ``day``; it must be a trading day after the last one, which must be closed."""
        if pd.Timestamp(day) not in self._data.closes.index:
            raise SimulationError(f"open({day}): not a trading day")
        if self.day is not None and day <= self.day:
            raise SimulationError(f"open({day}): must be after the last day {self.day}")
        if self.phase != "closed":
            raise SimulationError(f"open({day}): day {self.day} is not closed")
        self.day = day
        self.phase = "open"

    def close(self, day: date) -> None:
        """End ``day``, which must be the open day."""
        if day != self.day or self.phase != "open":
            raise SimulationError(f"close({day}): that day is not open")
        self.phase = "closed"

    def account(self) -> Account:
        held = sum(n * self.marks[t] for t, n in self.shares.items())
        return Account(equity=self.cash + held)

    def positions(self) -> tuple[Holding, ...]:
        return tuple(Holding(t, n) for t, n in sorted(self.shares.items()) if n != 0)

    def submit(self, request: OrderRequest) -> SubmitResult:
        cid = request.client_order_id
        existing = self.orders.get(cid)
        if existing is not None:
            return SubmitResult(request, SubmitStatus.DUPLICATE, existing.broker_order_id, None)
        order = request.order
        reason = None
        if order.shares <= 0:
            reason = f"shares must be positive, got {order.shares}"
        elif order.ticker not in self._data.opens.columns:
            reason = f"unknown ticker {order.ticker}"
        if reason is not None:
            self._reject(request, reason)
            return SubmitResult(request, SubmitStatus.REJECTED, None, reason)
        broker_order_id = f"sim-{len(self.orders) + 1}"
        self.orders[cid] = _SimOrder(
            request, broker_order_id, OrderState.PENDING, None, None, self.day
        )
        return SubmitResult(request, SubmitStatus.ACCEPTED, broker_order_id, None)

    def order_status(self, client_order_id: str) -> OrderStatus:
        sim = self.orders.get(client_order_id)
        if sim is None:
            return OrderStatus(
                client_order_id,
                OrderState.NOT_FOUND,
                None,
                None,
                None,
                Decimal(0),
                None,
                None,
                None,
            )
        order = sim.request.order
        filled = sim.state is OrderState.FILLED
        return OrderStatus(
            client_order_id=client_order_id,
            state=sim.state,
            broker_order_id=sim.broker_order_id,
            ticker=order.ticker,
            side=_ORDER_SIDE[order.action],
            filled_shares=Decimal(order.shares) if filled else Decimal(0),
            avg_price=sim.fill_price if filled else None,
            filled_at=_fill_time(cast(date, sim.filled_on)) if filled else None,
            broker_state=sim.state.value,
        )

    def _reject(self, request: OrderRequest, reason: str) -> None:
        self._events.append(
            SimEvent(
                day=self.day if self.day is not None else self._data.start,
                kind=SimEventKind.REJECT,
                ticker=request.order.ticker,
                shares=0,
                price=0.0,
                cash=0.0,
                client_order_id=request.client_order_id,
                reason=reason,
            )
        )


def _fill_time(filled_on: date) -> datetime:
    return datetime.combine(filled_on, _MARKET_OPEN, tzinfo=_MARKET_TZ).astimezone(UTC)
