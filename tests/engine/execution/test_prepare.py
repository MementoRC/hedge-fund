"""Order plan -> broker requests: split, ids, flips and malformed plans."""

from __future__ import annotations

from datetime import date

import pytest

from hedge_fund.domain.execution import OrderRequest
from hedge_fund.domain.orders import BLOCKED_HALT, BLOCKED_VETO, Order, OrderAction, OrderPlan
from hedge_fund.engine.execution.prepare import PreparedSubmission, prepare_submission

AS_OF = date(2026, 10, 21)
PRICE = 100.0
BUY = OrderAction.BUY
SELL = OrderAction.SELL
SELL_SHORT = OrderAction.SELL_SHORT
BUY_TO_COVER = OrderAction.BUY_TO_COVER
NOTHING = PreparedSubmission(reducing=(), opening=(), deferred=frozenset())


def _order(ticker: str, action: OrderAction, shares: int) -> Order:
    return Order(
        ticker=ticker, action=action, shares=shares, ref_price=PRICE, notional=shares * PRICE
    )


def _plan(*orders: Order, blocked: str | None = None) -> OrderPlan:
    return OrderPlan(as_of=AS_OF, orders=orders, skipped={}, residual_weight={}, blocked=blocked)


def _request(order: Order) -> OrderRequest:
    client_order_id = f"hf-2026-10-21-{order.ticker}-{order.action.value}-{order.shares}"
    return OrderRequest(order=order, client_order_id=client_order_id)


# --- split and ids ----------------------------------------------------------------------


def test_empty_plan() -> None:
    assert prepare_submission(_plan()) == NOTHING


@pytest.mark.parametrize("blocked", [BLOCKED_HALT, BLOCKED_VETO])
def test_blocked_plan_without_orders(blocked: str) -> None:
    assert prepare_submission(_plan(blocked=blocked)) == NOTHING


def test_id_format() -> None:
    prepared = prepare_submission(_plan(_order("BRK.B", SELL_SHORT, 625)))
    assert [request.client_order_id for request in prepared.opening] == [
        "hf-2026-10-21-BRK.B-sell_short-625"
    ]


def test_id_includes_shares() -> None:
    small = prepare_submission(_plan(_order("A", BUY, 10))).opening[0]
    large = prepare_submission(_plan(_order("A", BUY, 11))).opening[0]
    assert small.client_order_id != large.client_order_id


def test_split_keeps_plan_order() -> None:
    reducing = (_order("C", SELL, 1250), _order("E", BUY_TO_COVER, 1250), _order("A", SELL, 625))
    opening = (_order("D", BUY, 1250), _order("B", SELL_SHORT, 625))
    prepared = prepare_submission(_plan(*reducing, *opening))
    assert prepared == PreparedSubmission(
        reducing=tuple(_request(order) for order in reducing),
        opening=tuple(_request(order) for order in opening),
        deferred=frozenset(),
    )


# --- flips ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("close", "open_"),
    [
        pytest.param(_order("A", SELL, 1250), _order("A", SELL_SHORT, 625), id="long-to-short"),
        pytest.param(_order("A", BUY_TO_COVER, 1250), _order("A", BUY, 625), id="short-to-long"),
    ],
)
def test_flip_open_leg_is_deferred(close: Order, open_: Order) -> None:
    plain_close, plain_open = _order("B", SELL, 5), _order("C", BUY, 5)
    prepared = prepare_submission(_plan(close, plain_close, open_, plain_open))
    assert prepared.reducing == (_request(close), _request(plain_close))
    assert prepared.opening == (_request(open_), _request(plain_open))
    assert prepared.deferred == frozenset({_request(open_).client_order_id})


# --- malformed plans (cannot come from generate_orders) ---------------------------------


def test_rejects_blocked_plan_with_orders() -> None:
    with pytest.raises(ValueError, match=r"blocked plan \(halt\) must have no orders, got 1"):
        prepare_submission(_plan(_order("A", SELL, 5), blocked=BLOCKED_HALT))


@pytest.mark.parametrize(
    "opening", [_order("A", BUY, 5), _order("A", SELL_SHORT, 5)], ids=["buy", "sell-short"]
)
def test_rejects_reducing_order_after_opening_order(opening: Order) -> None:
    with pytest.raises(
        ValueError, match="reducing order 'hf-2026-10-21-B-sell-5' follows an opening order"
    ):
        prepare_submission(_plan(opening, _order("B", SELL, 5)))


def test_rejects_duplicate_client_order_id() -> None:
    order = _order("A", BUY, 5)
    with pytest.raises(ValueError, match="duplicate client_order_id 'hf-2026-10-21-A-buy-5'"):
        prepare_submission(_plan(order, order))


def test_accepts_id_of_exactly_128_characters() -> None:
    prepared = prepare_submission(_plan(_order("X" * 107, SELL, 1)))  # 14 + 107 + 7 chars
    assert len(prepared.reducing[0].client_order_id) == 128


def test_rejects_id_over_128_characters() -> None:
    with pytest.raises(ValueError, match="exceeds 128 characters"):
        prepare_submission(_plan(_order("X" * 108, SELL, 1)))


def test_package_exports() -> None:
    import hedge_fund.engine.execution as execution

    assert execution.__all__ == [
        "MARKET_OPEN",
        "MARKET_TZ",
        "SESSION_END",
        "SESSION_START",
        "PreparedSubmission",
        "in_session_window",
        "prepare_submission",
        "session_opened",
    ]
