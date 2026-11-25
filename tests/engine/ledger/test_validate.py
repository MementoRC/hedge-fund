"""to_fill: a final broker OrderStatus checked against its request and turned into a Fill."""

from __future__ import annotations

import dataclasses
import math
from datetime import UTC, date, datetime
from decimal import Decimal
from typing import Any

import pytest

from hedge_fund.domain.execution import OrderRequest
from hedge_fund.domain.ledger import Fill, FillStatus, OrderState, OrderStatus
from hedge_fund.domain.orders import Order, OrderAction
from hedge_fund.engine.ledger.validate import to_fill

CID = "hf-2026-09-30-AAA-buy-10"
CYCLE_DAY = date(2026, 9, 30)
OPEN = datetime(2026, 10, 1, 13, 30, tzinfo=UTC)
RECORDED = datetime(2026, 10, 1, 14, 0, tzinfo=UTC)
ORDER = Order(ticker="AAA", action=OrderAction.BUY, shares=10, ref_price=100.0, notional=1000.0)
REQUEST = OrderRequest(order=ORDER, client_order_id=CID)
STATUS = OrderStatus(
    client_order_id=CID,
    state=OrderState.FILLED,
    broker_order_id="b-1",
    ticker="AAA",
    side="buy",
    filled_shares=Decimal("10"),
    avg_price=101.0,
    filled_at=OPEN,
    broker_state="filled",
)
FILLED = Fill(
    client_order_id=CID,
    broker_order_id="b-1",
    cycle_as_of=CYCLE_DAY,
    ticker="AAA",
    action=OrderAction.BUY,
    requested_shares=10,
    filled_shares=10,
    avg_price=101.0,
    filled_at=OPEN,
    status=FillStatus.FILLED,
    recorded_at=RECORDED,
)


def _fill(status: OrderStatus, request: OrderRequest = REQUEST) -> Fill:
    return to_fill(status, request, cycle_as_of=CYCLE_DAY, recorded_at=RECORDED)


def _request(action: OrderAction, ticker: str = "AAA") -> OrderRequest:
    order = dataclasses.replace(ORDER, action=action, ticker=ticker)
    return OrderRequest(order=order, client_order_id=CID)


def test_a_filled_order() -> None:
    assert _fill(STATUS) == FILLED


def test_a_partial_terminal_order() -> None:
    status = dataclasses.replace(
        STATUS, state=OrderState.PARTIAL_TERMINAL, filled_shares=Decimal("4")
    )
    assert _fill(status) == dataclasses.replace(
        FILLED, filled_shares=4, status=FillStatus.PARTIAL_TERMINAL
    )


@pytest.mark.parametrize(
    ("state", "fill_status"),
    [
        (OrderState.CANCELED, FillStatus.CANCELED),
        (OrderState.REJECTED, FillStatus.REJECTED),
        (OrderState.EXPIRED, FillStatus.EXPIRED),
    ],
)
def test_zero_fill_states_have_no_price_or_time(state: OrderState, fill_status: FillStatus) -> None:
    status = dataclasses.replace(
        STATUS, state=state, filled_shares=Decimal("0"), avg_price=None, filled_at=None
    )
    assert _fill(status) == dataclasses.replace(
        FILLED, filled_shares=0, avg_price=None, filled_at=None, status=fill_status
    )


def test_a_zero_fill_drops_a_stray_price_and_time() -> None:
    status = dataclasses.replace(STATUS, state=OrderState.CANCELED, filled_shares=Decimal("0"))
    fill = _fill(status)
    assert (fill.avg_price, fill.filled_at, fill.filled_shares) == (None, None, 0)


def test_a_dot_symbol_matches_the_dash_request_and_the_fill_keeps_the_request_ticker() -> None:
    fill = _fill(dataclasses.replace(STATUS, ticker="BRK.B"), _request(OrderAction.BUY, "BRK-B"))
    assert fill.ticker == "BRK-B"


def test_an_integral_decimal_with_a_fraction_part_is_whole() -> None:
    fill = _fill(dataclasses.replace(STATUS, filled_shares=Decimal("10.0")))
    assert fill.filled_shares == 10
    assert type(fill.filled_shares) is int


@pytest.mark.parametrize(
    ("action", "side"),
    [
        (OrderAction.BUY, "buy"),
        (OrderAction.BUY_TO_COVER, "buy"),
        (OrderAction.SELL, "sell"),
        (OrderAction.SELL_SHORT, "sell"),
    ],
)
def test_the_side_matches_the_action(action: OrderAction, side: str) -> None:
    fill = _fill(dataclasses.replace(STATUS, side=side), _request(action))
    assert fill.action is action


@pytest.mark.parametrize(
    ("action", "side"),
    [
        (OrderAction.BUY, "sell"),
        (OrderAction.BUY_TO_COVER, "sell"),
        (OrderAction.SELL, "buy"),
        (OrderAction.SELL_SHORT, "buy"),
    ],
)
def test_a_side_that_does_not_match_the_action_is_refused(action: OrderAction, side: str) -> None:
    with pytest.raises(ValueError, match=f"side '{side}' does not match {action.value}"):
        _fill(dataclasses.replace(STATUS, side=side), _request(action))


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        pytest.param({"ticker": "MSFT"}, "broker symbol 'MSFT' does not match", id="ticker"),
        pytest.param({"ticker": None}, "broker symbol None does not match", id="no-ticker"),
        pytest.param({"filled_shares": Decimal("11")}, "exceeds the 10 requested", id="overfill"),
        pytest.param({"filled_shares": Decimal("2.5")}, "is not a whole number", id="fractional"),
        pytest.param({"filled_shares": Decimal("NaN")}, "is not a whole number", id="nan-shares"),
        pytest.param(
            {"filled_shares": Decimal("Infinity")}, "is not a whole number", id="inf-shares"
        ),
        pytest.param({"filled_shares": Decimal("-1")}, "with -1 filled shares", id="negative"),
        pytest.param(
            {"state": OrderState.CANCELED, "filled_shares": Decimal("-1")},
            "canceled with -1 filled shares",
            id="zero-state-negative",
        ),
        pytest.param(
            {"filled_shares": Decimal("0")}, "filled with 0 filled shares", id="no-shares"
        ),
        pytest.param(
            {"state": OrderState.CANCELED, "filled_shares": Decimal("3")},
            "canceled with 3 filled shares",
            id="zero-state-with-shares",
        ),
        pytest.param({"avg_price": math.nan}, "not a finite positive price", id="nan-price"),
        pytest.param({"avg_price": math.inf}, "not a finite positive price", id="inf-price"),
        pytest.param({"avg_price": 0.0}, "not a finite positive price", id="zero-price"),
        pytest.param({"avg_price": -1.0}, "not a finite positive price", id="negative-price"),
        pytest.param({"avg_price": None}, "not a finite positive price", id="no-price"),
        pytest.param({"filled_at": None}, "has no filled_at", id="no-filled-at"),
    ],
)
def test_refusals(overrides: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _fill(dataclasses.replace(STATUS, **overrides))


@pytest.mark.parametrize(
    "state", [OrderState.PENDING, OrderState.NOT_FOUND, OrderState.UNRECOGNISED]
)
def test_a_state_that_is_not_final_is_refused(state: OrderState) -> None:
    with pytest.raises(ValueError, match=f"{state.value} is not a final state"):
        _fill(dataclasses.replace(STATUS, state=state))
