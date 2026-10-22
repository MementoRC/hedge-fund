"""Layer 6a execution service: submission policy over a scripted broker."""

from __future__ import annotations

from datetime import date

import pytest

from hedge_fund.domain.execution import (
    REASON_FLIP_DEFERRED,
    REASON_REDUCING_FAILED,
    ExecutionReport,
    OrderRequest,
    SubmitResult,
    SubmitStatus,
)
from hedge_fund.domain.orders import BLOCKED_HALT, BLOCKED_VETO, Order, OrderAction, OrderPlan
from hedge_fund.ports.broker import BrokerError
from hedge_fund.services.execution import execute_plan
from tests.fakes import SCRIPTED_REJECTION, ScriptedBroker

AS_OF = date(2026, 10, 21)
PRICE = 100.0
ACCEPTED = SubmitStatus.ACCEPTED
DUPLICATE = SubmitStatus.DUPLICATE
REJECTED = SubmitStatus.REJECTED
ERROR = SubmitStatus.ERROR
DEFERRED = SubmitStatus.DEFERRED
SKIPPED = SubmitStatus.SKIPPED


def _order(ticker: str, action: OrderAction, shares: int) -> Order:
    return Order(
        ticker=ticker, action=action, shares=shares, ref_price=PRICE, notional=shares * PRICE
    )


SELL_A = _order("A", OrderAction.SELL, 10)
COVER_B = _order("B", OrderAction.BUY_TO_COVER, 20)
SHORT_C = _order("C", OrderAction.SELL_SHORT, 30)
BUY_D = _order("D", OrderAction.BUY, 40)
SELL_F = _order("F", OrderAction.SELL, 50)  # close leg of a long -> short flip
SHORT_F = _order("F", OrderAction.SELL_SHORT, 25)  # open leg of the same flip


def _plan(*orders: Order, blocked: str | None = None) -> OrderPlan:
    return OrderPlan(as_of=AS_OF, orders=orders, skipped={}, residual_weight={}, blocked=blocked)


def _id(order: Order) -> str:
    return f"hf-2026-10-21-{order.ticker}-{order.action.value}-{order.shares}"


def _request(order: Order) -> OrderRequest:
    return OrderRequest(order=order, client_order_id=_id(order))


def _sent(order: Order, status: SubmitStatus = ACCEPTED) -> SubmitResult:
    """What ScriptedBroker returns when ``order`` is submitted with ``status`` scripted."""
    if status is REJECTED:
        return SubmitResult(
            request=_request(order), status=status, broker_order_id=None, reason=SCRIPTED_REJECTION
        )
    return SubmitResult(
        request=_request(order), status=status, broker_order_id=f"b-{_id(order)}", reason=None
    )


def _unsent(order: Order, status: SubmitStatus, reason: str) -> SubmitResult:
    return SubmitResult(request=_request(order), status=status, broker_order_id=None, reason=reason)


def _report(*results: SubmitResult, halted: bool = False) -> ExecutionReport:
    return ExecutionReport(as_of=AS_OF, blocked=None, results=results, opening_halted=halted)


# --- plan-level outcomes ----------------------------------------------------------------


@pytest.mark.parametrize("blocked", [BLOCKED_HALT, BLOCKED_VETO])
def test_blocked_plan_makes_no_broker_calls(blocked: str) -> None:
    broker = ScriptedBroker()
    assert execute_plan(_plan(blocked=blocked), broker) == ExecutionReport(
        as_of=AS_OF, blocked=blocked, results=(), opening_halted=False
    )
    assert broker.submitted == []


def test_blocked_plan_with_orders_raises_before_any_call() -> None:
    broker = ScriptedBroker()
    with pytest.raises(ValueError, match="blocked plan"):
        execute_plan(_plan(SELL_A, blocked=BLOCKED_HALT), broker)
    assert broker.submitted == []


def test_malformed_plan_raises_before_any_call() -> None:
    broker = ScriptedBroker()
    with pytest.raises(ValueError, match="follows an opening order"):
        execute_plan(_plan(BUY_D, SELL_A), broker)
    assert broker.submitted == []


def test_empty_plan() -> None:
    broker = ScriptedBroker()
    assert execute_plan(_plan(), broker) == _report()
    assert broker.submitted == []


def test_all_accepted_in_plan_order() -> None:
    broker = ScriptedBroker()
    report = execute_plan(_plan(SELL_A, COVER_B, SHORT_C, BUY_D), broker)
    assert report == _report(_sent(SELL_A), _sent(COVER_B), _sent(SHORT_C), _sent(BUY_D))
    assert broker.submitted == [_id(SELL_A), _id(COVER_B), _id(SHORT_C), _id(BUY_D)]


def test_duplicate_counts_as_success() -> None:
    broker = ScriptedBroker({_id(SELL_A): DUPLICATE, _id(BUY_D): DUPLICATE})
    report = execute_plan(_plan(SELL_A, SHORT_C, BUY_D), broker)
    assert report == _report(_sent(SELL_A, DUPLICATE), _sent(SHORT_C), _sent(BUY_D, DUPLICATE))


# --- reducing-leg failures --------------------------------------------------------------


def test_rejected_reducing_leg_skips_opening_legs() -> None:
    broker = ScriptedBroker({_id(SELL_A): REJECTED})
    report = execute_plan(_plan(SELL_A, COVER_B, SHORT_C, BUY_D), broker)
    assert report == _report(
        _sent(SELL_A, REJECTED),
        _sent(COVER_B),
        _unsent(SHORT_C, SKIPPED, REASON_REDUCING_FAILED),
        _unsent(BUY_D, SKIPPED, REASON_REDUCING_FAILED),
        halted=True,
    )
    assert broker.submitted == [_id(SELL_A), _id(COVER_B)]


def test_broker_error_on_reducing_leg_is_error_and_halts() -> None:
    broker = ScriptedBroker({_id(SELL_A): BrokerError("POST /v2/orders: HTTP 503")})
    report = execute_plan(_plan(SELL_A, COVER_B, BUY_D), broker)
    assert report == _report(
        _unsent(SELL_A, ERROR, "POST /v2/orders: HTTP 503"),
        _sent(COVER_B),
        _unsent(BUY_D, SKIPPED, REASON_REDUCING_FAILED),
        halted=True,
    )
    assert broker.submitted == [_id(SELL_A), _id(COVER_B)]


def test_failed_reducing_leg_without_opening_legs_still_halts() -> None:
    broker = ScriptedBroker({_id(SELL_A): REJECTED})
    assert execute_plan(_plan(SELL_A), broker) == _report(_sent(SELL_A, REJECTED), halted=True)


def test_flip_open_leg_is_skipped_not_deferred_after_a_halt() -> None:
    broker = ScriptedBroker({_id(SELL_A): REJECTED})
    report = execute_plan(_plan(SELL_A, SELL_F, SHORT_F), broker)
    assert report == _report(
        _sent(SELL_A, REJECTED),
        _sent(SELL_F),
        _unsent(SHORT_F, SKIPPED, REASON_REDUCING_FAILED),
        halted=True,
    )


# --- opening legs -----------------------------------------------------------------------


def test_flip_open_leg_is_deferred_without_a_broker_call() -> None:
    broker = ScriptedBroker()
    report = execute_plan(_plan(SELL_F, SHORT_C, SHORT_F, BUY_D), broker)
    assert report == _report(
        _sent(SELL_F),
        _sent(SHORT_C),
        _unsent(SHORT_F, DEFERRED, REASON_FLIP_DEFERRED),
        _sent(BUY_D),
    )
    assert broker.submitted == [_id(SELL_F), _id(SHORT_C), _id(BUY_D)]


def test_rejected_opening_leg_does_not_stop_later_opening_legs() -> None:
    broker = ScriptedBroker({_id(SHORT_C): REJECTED})
    report = execute_plan(_plan(SELL_A, SHORT_C, BUY_D), broker)
    assert report == _report(_sent(SELL_A), _sent(SHORT_C, REJECTED), _sent(BUY_D))


def test_broker_error_on_opening_leg_is_error_and_later_legs_continue() -> None:
    broker = ScriptedBroker({_id(SHORT_C): BrokerError("GET /v2/orders: timed out")})
    report = execute_plan(_plan(SELL_A, SHORT_C, BUY_D), broker)
    assert report == _report(
        _sent(SELL_A), _unsent(SHORT_C, ERROR, "GET /v2/orders: timed out"), _sent(BUY_D)
    )


def test_non_broker_exceptions_propagate() -> None:
    broker = ScriptedBroker({_id(SELL_A): RuntimeError("bug")})
    with pytest.raises(RuntimeError, match="bug"):
        execute_plan(_plan(SELL_A), broker)
