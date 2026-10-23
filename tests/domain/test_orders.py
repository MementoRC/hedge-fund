"""Order domain types."""

from __future__ import annotations

import dataclasses
from datetime import date

import pytest

from hedge_fund.domain.orders import (
    BLOCKED_HALT,
    BLOCKED_VETO,
    SKIPPED_BAND,
    SKIPPED_ROUNDS_TO_ZERO,
    Holding,
    Order,
    OrderAction,
    OrderPlan,
)


def test_order_action_values() -> None:
    assert [action.value for action in OrderAction] == [
        "buy",
        "sell",
        "sell_short",
        "buy_to_cover",
    ]
    assert OrderAction.SELL_SHORT == "sell_short"


def test_reason_constants() -> None:
    assert (SKIPPED_BAND, SKIPPED_ROUNDS_TO_ZERO, BLOCKED_HALT, BLOCKED_VETO) == (
        "band",
        "rounds-to-zero",
        "halt",
        "veto",
    )


def test_holding_is_frozen() -> None:
    holding = Holding(ticker="A", shares=-5)
    with pytest.raises(dataclasses.FrozenInstanceError):
        holding.shares = 1  # type: ignore[misc]


def test_order_is_frozen() -> None:
    order = Order(ticker="A", action=OrderAction.BUY, shares=5, ref_price=10.0, notional=50.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        order.shares = 1  # type: ignore[misc]


def test_order_plan_is_frozen() -> None:
    plan = OrderPlan(
        as_of=date(2026, 10, 21), orders=(), skipped={}, residual_weight={}, blocked=None
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        plan.blocked = BLOCKED_HALT  # type: ignore[misc]
