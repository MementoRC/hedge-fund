"""Plan fingerprint: covers as_of and each order's ticker, action and shares, nothing else."""

from __future__ import annotations

import dataclasses
import hashlib
from datetime import date

import pytest

from hedge_fund.domain.orders import BLOCKED_HALT, SKIPPED_BAND, Order, OrderAction, OrderPlan
from hedge_fund.engine.approval.render import plan_fingerprint

AS_OF = date(2026, 10, 23)
SELL_AAPL = Order(
    ticker="AAPL", action=OrderAction.SELL, shares=40, ref_price=231.1, notional=9244.0
)
BUY_MSFT = Order(
    ticker="MSFT", action=OrderAction.BUY, shares=12, ref_price=412.05, notional=4944.6
)


def _plan(*orders: Order, as_of: date = AS_OF) -> OrderPlan:
    return OrderPlan(as_of=as_of, orders=orders, skipped={}, residual_weight={}, blocked=None)


BASE = _plan(SELL_AAPL, BUY_MSFT)


def test_known_digest() -> None:
    preimage = b"2026-10-23|AAPL,sell,40;MSFT,buy,12"
    assert plan_fingerprint(BASE) == hashlib.sha256(preimage).hexdigest()
    assert plan_fingerprint(BASE) == (
        "b30d16c30ce8d1b31f3a3218b27d2472d819e2baca2c793daa7dd3548e72f673"
    )


def test_empty_plan_has_a_fingerprint() -> None:
    assert plan_fingerprint(_plan()) == hashlib.sha256(b"2026-10-23|").hexdigest()


def test_deterministic() -> None:
    assert plan_fingerprint(BASE) == plan_fingerprint(_plan(SELL_AAPL, BUY_MSFT))


@pytest.mark.parametrize(
    "other",
    [
        pytest.param(_plan(SELL_AAPL, BUY_MSFT, as_of=date(2026, 10, 24)), id="as-of"),
        pytest.param(_plan(SELL_AAPL, dataclasses.replace(BUY_MSFT, ticker="MSFU")), id="ticker"),
        pytest.param(
            _plan(SELL_AAPL, dataclasses.replace(BUY_MSFT, action=OrderAction.SELL_SHORT)),
            id="action",
        ),
        pytest.param(_plan(SELL_AAPL, dataclasses.replace(BUY_MSFT, shares=13)), id="shares"),
        pytest.param(_plan(BUY_MSFT, SELL_AAPL), id="sequence"),
        pytest.param(_plan(SELL_AAPL), id="order-dropped"),
    ],
)
def test_changes_with_the_orders_and_as_of(other: OrderPlan) -> None:
    assert plan_fingerprint(other) != plan_fingerprint(BASE)


@pytest.mark.parametrize(
    "other",
    [
        pytest.param(
            _plan(SELL_AAPL, dataclasses.replace(BUY_MSFT, ref_price=1.0, notional=12.0)),
            id="ref-price-and-notional",
        ),
        pytest.param(dataclasses.replace(BASE, skipped={"KO": SKIPPED_BAND}), id="skipped"),
        pytest.param(dataclasses.replace(BASE, residual_weight={"KO": 0.01}), id="residuals"),
        pytest.param(dataclasses.replace(BASE, blocked=BLOCKED_HALT), id="blocked"),
    ],
)
def test_ignores_prices_skips_residuals_and_blocked(other: OrderPlan) -> None:
    assert plan_fingerprint(other) == plan_fingerprint(BASE)
