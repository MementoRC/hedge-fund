"""Seeded invariants of prepare_submission over plans built by 6b's generate_orders."""

from __future__ import annotations

import pytest

from hedge_fund.domain.orders import OrderAction, OrderPlan
from hedge_fund.engine.execution import prepare_submission
from hedge_fund.engine.orders import OrderParams, generate_orders
from tests.engine.orders.test_generate_properties import NAV, VERDICT, _book

SEEDS = range(300)
REDUCING = {OrderAction.SELL, OrderAction.BUY_TO_COVER}


def _plan(seed: int) -> OrderPlan:
    target, holdings, prices, band = _book(seed)
    return generate_orders(
        target, VERDICT, holdings, prices, NAV, params=OrderParams(no_trade_band=band)
    )


@pytest.mark.parametrize("seed", SEEDS)
def test_invariants(seed: int) -> None:
    plan = _plan(seed)
    prepared = prepare_submission(plan)
    sent = prepared.reducing + prepared.opening
    assert tuple(request.order for request in sent) == plan.orders
    ids = [request.client_order_id for request in sent]
    assert len(set(ids)) == len(ids)
    assert all(len(client_order_id) <= 128 for client_order_id in ids)
    assert all(request.order.action in REDUCING for request in prepared.reducing)
    assert not any(request.order.action in REDUCING for request in prepared.opening)
    closing = {request.order.ticker for request in prepared.reducing}
    assert prepared.deferred <= {request.client_order_id for request in prepared.opening}
    for request in prepared.opening:
        is_deferred = request.client_order_id in prepared.deferred
        assert is_deferred == (request.order.ticker in closing)


def test_grid_exercises_flips_and_plain_opening_legs() -> None:
    prepared = [prepare_submission(_plan(seed)) for seed in SEEDS]
    assert any(item.deferred for item in prepared)
    assert any(len(item.opening) > len(item.deferred) for item in prepared)
