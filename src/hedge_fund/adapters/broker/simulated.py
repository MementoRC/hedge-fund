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
from fractions import Fraction
from typing import Literal, cast
from zoneinfo import ZoneInfo

import pandas as pd

from hedge_fund.domain.backtest import FrozenData, SimCosts, SimEvent, SimEventKind
from hedge_fund.domain.execution import Account, OrderRequest, SubmitResult, SubmitStatus
from hedge_fund.domain.ledger import OrderState, OrderStatus
from hedge_fund.domain.orders import Holding, OrderAction
from hedge_fund.engine.ledger.splits import split_factor

_MARKET_OPEN = time(9, 30)
_MARKET_TZ = ZoneInfo("America/New_York")
_BUY_SIDE = frozenset({OrderAction.BUY, OrderAction.BUY_TO_COVER})
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
        self._delisted: list[str] = []
        self._last_day: dict[str, date] = {
            str(t): last.date()
            for t in data.closes.columns
            if (last := data.closes[t].last_valid_index()) is not None
        }
        self.day: date | None = None
        self.phase: Literal["open", "closed"] = "closed"

    @property
    def events(self) -> tuple[SimEvent, ...]:
        return tuple(self._events)

    @property
    def delisted(self) -> tuple[str, ...]:
        """Held names closed out because they stopped trading, in delisting order."""
        return tuple(self._delisted)

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
        # Ex-date events come before fills, in this order: delistings, splits, dividends.
        self._apply_delistings(day)
        self._apply_splits(day)
        self._apply_dividends(day)
        for sim in self.orders.values():
            if sim.state is OrderState.PENDING and (
                sim.submitted_on is None or sim.submitted_on < day
            ):
                if pd.isna(self._data.opens.at[pd.Timestamp(day), sim.request.order.ticker]):
                    sim.state = OrderState.REJECTED
                    self._reject(sim.request, "no open")
                    continue
                self._fill(sim, day)

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

    def _raw_open(self, ticker: str, day: date) -> float:
        """The open actually traded on ``day``: the stored (split-adjusted) open unwound."""
        stored = float(self._data.opens.at[pd.Timestamp(day), ticker])
        return stored * float(split_factor(ticker, day, self._data.splits))

    def _gone(self, ticker: str, day: date) -> bool:
        """Whether ``ticker`` has stopped trading before ``day`` (its last close is earlier)."""
        last = self._last_day.get(ticker)
        return last is not None and day > last

    def _apply_delistings(self, day: date) -> None:
        """Close out held names that stopped trading, and reject orders on them."""
        for ticker in sorted(self.shares):
            held = self.shares[ticker]
            if held == 0 or not self._gone(ticker, day):
                continue
            last = self._last_day[ticker]
            stored = float(self._data.closes.at[pd.Timestamp(last), ticker])
            price = stored * float(split_factor(ticker, last, self._data.splits))
            cash = held * price
            self.cash += cash
            del self.shares[ticker]
            self.marks.pop(ticker, None)
            if ticker not in self._delisted:
                self._delisted.append(ticker)
            self._events.append(
                SimEvent(
                    day=day,
                    kind=SimEventKind.DELIST,
                    ticker=ticker,
                    shares=-held,
                    price=price,
                    cash=cash,
                )
            )
        for sim in self.orders.values():
            if sim.state is OrderState.PENDING and self._gone(sim.request.order.ticker, day):
                sim.state = OrderState.REJECTED
                self._reject(sim.request, "not trading")

    def _split_price(self, ticker: str, day: date, ratio: Fraction) -> float:
        """The post-split price for cash in lieu: the raw open, else the prior raw close.

        With no open on the split day, the last earlier close is restated onto the post-split
        basis (divided by ``ratio``, as marks are). With neither, the simulation fails closed.
        """
        opened = float(self._data.opens.at[pd.Timestamp(day), ticker])
        if not math.isnan(opened):
            return self._raw_open(ticker, day)
        closes = self._data.closes[ticker]
        prior = closes[closes.index < pd.Timestamp(day)].dropna()
        if prior.empty:
            raise SimulationError(f"split of {ticker} on {day}: no open and no prior close")
        prior_day = prior.index[-1].date()
        raw = float(prior.iloc[-1]) * float(split_factor(ticker, prior_day, self._data.splits))
        return raw / float(ratio)

    def _apply_splits(self, day: date) -> None:
        """Restate holdings and marks for splits ex-dated ``day``; reject orders sized pre-split."""
        for split in self._data.splits:
            if split.ex_date != day:
                continue
            ticker = split.ticker
            for sim in self.orders.values():
                if sim.state is OrderState.PENDING and sim.request.order.ticker == ticker:
                    sim.state = OrderState.REJECTED
                    self._reject(sim.request, "split on fill day")
            held = self.shares.get(ticker, 0)
            if held == 0:
                continue
            scaled = held * split.ratio
            whole = int(scaled)  # truncates toward zero, so shorts are symmetric
            raw_open = self._split_price(ticker, day, split.ratio)
            cash = float(scaled - whole) * raw_open if scaled != whole else 0.0
            self.cash += cash
            self.shares[ticker] = whole
            if ticker in self.marks:
                self.marks[ticker] /= float(split.ratio)
            self._events.append(
                SimEvent(
                    day=day,
                    kind=SimEventKind.SPLIT,
                    ticker=ticker,
                    shares=whole - held,
                    price=raw_open,
                    cash=cash,
                )
            )

    def _apply_dividends(self, day: date) -> None:
        """Pay (or charge, for a short) the raw dividend of every held name ex-dated ``day``."""
        for dividend in self._data.dividends:
            held = self.shares.get(dividend.ticker, 0)
            if dividend.ex_date != day or held == 0:
                continue
            raw = dividend.amount * float(split_factor(dividend.ticker, day, self._data.splits))
            cash = held * raw
            self.cash += cash
            self._events.append(
                SimEvent(
                    day=day,
                    kind=SimEventKind.DIVIDEND,
                    ticker=dividend.ticker,
                    shares=held,
                    price=raw,
                    cash=cash,
                )
            )

    def _fill(self, sim: _SimOrder, day: date) -> None:
        order = sim.request.order
        held = self.shares.get(order.ticker, 0)
        reason = _fill_violation(order.action, order.shares, held)
        if reason is not None:
            sim.state = OrderState.REJECTED
            self._reject(sim.request, reason)
            return
        raw_open = self._raw_open(order.ticker, day)
        slip = self._costs.slippage_bps / 10_000
        buy = order.action in _BUY_SIDE
        signed = order.shares if buy else -order.shares
        price = raw_open * (1 + slip if buy else 1 - slip)
        cash = -signed * price
        self.cash += cash
        self.shares[order.ticker] = held + signed
        self.marks.setdefault(order.ticker, raw_open)
        sim.state, sim.fill_price, sim.filled_on = OrderState.FILLED, price, day
        self._events.append(
            SimEvent(
                day=day,
                kind=SimEventKind.FILL,
                ticker=order.ticker,
                shares=signed,
                price=price,
                cash=cash,
                cost=abs(signed) * raw_open * slip,
                client_order_id=sim.request.client_order_id,
            )
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


def _fill_violation(action: OrderAction, shares: int, held: int) -> str | None:
    """Why an order must not fill against the holding ``held``, or None (fail closed)."""
    if action is OrderAction.SELL and shares > max(held, 0):
        return "would cross zero"
    if action is OrderAction.BUY_TO_COVER and shares > max(-held, 0):
        return "would cross zero"
    if (action is OrderAction.BUY and held < 0) or (action is OrderAction.SELL_SHORT and held > 0):
        return "wrong side for the holding"
    return None


def _fill_time(filled_on: date) -> datetime:
    return datetime.combine(filled_on, _MARKET_OPEN, tzinfo=_MARKET_TZ).astimezone(UTC)
