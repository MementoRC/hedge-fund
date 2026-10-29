"""submitted_orders: every id a cycle may have put live, and the planned ids with no order."""

from __future__ import annotations

import dataclasses
from datetime import UTC, date, datetime

import pytest

from hedge_fund.domain.cycle import CycleOutcome, CycleReport
from hedge_fund.domain.execution import ExecutionReport, OrderRequest, SubmitResult, SubmitStatus
from hedge_fund.domain.ledger import SubmittedOrder, UnsettledOrder
from hedge_fund.domain.orders import Order, OrderAction, OrderPlan
from hedge_fund.engine.ledger.submitted import NO_PLANNED_ORDER, submitted_orders

CYCLE_DAY = date(2026, 9, 30)
FIRST = datetime(2026, 9, 30, 22, 0, tzinfo=UTC)
LATER = datetime(2026, 9, 30, 23, 0, tzinfo=UTC)
SELL_CCC = Order(ticker="CCC", action=OrderAction.SELL, shares=20, ref_price=30.0, notional=600.0)
BUY_AAA = Order(ticker="AAA", action=OrderAction.BUY, shares=10, ref_price=100.0, notional=1000.0)
GHOST = "hf-2026-09-30-ZZZ-buy-1"


def _cid(order: Order) -> str:
    return f"hf-{CYCLE_DAY}-{order.ticker}-{order.action.value}-{order.shares}"


def _request(order: Order) -> OrderRequest:
    return OrderRequest(order=order, client_order_id=_cid(order))


def _report(
    *legs: tuple[Order, SubmitStatus],
    started_at: datetime = FIRST,
    dry_run: bool = False,
    plan: OrderPlan | None = None,
    planned: tuple[str, ...] | None = None,
) -> CycleReport:
    """A cycle report with ``legs`` as its execution, or, when ``planned`` is given, an
    interrupted one (no execution) whose planned ids are ``planned``.
    """
    orders = tuple(order for order, _ in legs)
    if plan is None and planned is None:
        plan = OrderPlan(CYCLE_DAY, orders, {}, {}, None)
    execution = None
    if planned is None:
        results = tuple(
            SubmitResult(_request(order), status, f"b-{_cid(order)}", None)
            for order, status in legs
        )
        execution = ExecutionReport(CYCLE_DAY, None, results, opening_halted=False)
    return CycleReport(
        as_of=CYCLE_DAY,
        started_at=started_at,
        dry_run=dry_run,
        outcome=CycleOutcome.SUBMITTED if planned is None else CycleOutcome.ERROR,
        failed_stage=None if planned is None else "execution",
        error=None if planned is None else "BrokerError: down",
        account=None,
        price_unresolved={},
        fundamentals_unresolved={},
        shortlist=None,
        research=None,
        research_reused=False,
        target=None,
        verdict=None,
        order_prices=None,
        plan=plan,
        decision=None,
        execution=execution,
        execution_interrupted=planned is not None,
        planned_client_order_ids=planned or (),
    )


def _submitted(order: Order, *, acknowledged: bool, started_at: datetime = FIRST) -> SubmittedOrder:
    return SubmittedOrder(
        request=_request(order),
        cycle_as_of=CYCLE_DAY,
        started_at=started_at,
        acknowledged=acknowledged,
    )


@pytest.mark.parametrize("status", [SubmitStatus.ACCEPTED, SubmitStatus.DUPLICATE])
def test_broker_acknowledged_legs_are_taken_as_acknowledged(status: SubmitStatus) -> None:
    orders, unsettled = submitted_orders([_report((BUY_AAA, status))])
    assert orders == {_cid(BUY_AAA): _submitted(BUY_AAA, acknowledged=True)}
    assert unsettled == ()


def test_error_legs_are_taken_unacknowledged() -> None:
    orders, _ = submitted_orders([_report((BUY_AAA, SubmitStatus.ERROR))])
    assert orders == {_cid(BUY_AAA): _submitted(BUY_AAA, acknowledged=False)}


@pytest.mark.parametrize(
    "status", [SubmitStatus.REJECTED, SubmitStatus.DEFERRED, SubmitStatus.SKIPPED]
)
def test_legs_that_never_went_live_are_skipped(status: SubmitStatus) -> None:
    assert submitted_orders([_report((BUY_AAA, status))]) == ({}, ())


def test_dry_run_reports_are_skipped() -> None:
    report = _report((BUY_AAA, SubmitStatus.ACCEPTED), dry_run=True)
    assert submitted_orders([report]) == ({}, ())


def test_an_interrupted_cycles_planned_ids_are_taken_unacknowledged() -> None:
    plan = OrderPlan(CYCLE_DAY, (SELL_CCC, BUY_AAA), {}, {}, None)
    report = _report(plan=plan, planned=(_cid(SELL_CCC), _cid(BUY_AAA)))
    orders, unsettled = submitted_orders([report])
    assert orders == {
        _cid(SELL_CCC): _submitted(SELL_CCC, acknowledged=False),
        _cid(BUY_AAA): _submitted(BUY_AAA, acknowledged=False),
    }
    assert unsettled == ()


@pytest.mark.parametrize(
    ("plan", "planned", "kept"),
    [
        pytest.param(
            OrderPlan(CYCLE_DAY, (BUY_AAA,), {}, {}, "veto"),  # blocked with orders: ValueError
            (_cid(BUY_AAA),),
            (),
            id="unpreparable-plan",
        ),
        pytest.param(
            OrderPlan(CYCLE_DAY, (BUY_AAA,), {}, {}, None),
            (_cid(BUY_AAA), GHOST),
            (_cid(BUY_AAA),),
            id="unmatched-id",
        ),
        pytest.param(None, (_cid(BUY_AAA),), (), id="no-plan"),
    ],
)
def test_a_planned_id_without_a_planned_order_is_unsettled(
    plan: OrderPlan | None, planned: tuple[str, ...], kept: tuple[str, ...]
) -> None:
    report = _report(plan=plan, planned=planned)
    if plan is None:
        report = dataclasses.replace(report, plan=None)
    orders, unsettled = submitted_orders([report])
    assert tuple(orders) == kept
    assert unsettled == tuple(
        UnsettledOrder(client_order_id=cid, ticker="", reason=NO_PLANNED_ORDER)
        for cid in sorted(set(planned) - set(kept))
    )
    assert NO_PLANNED_ORDER == "no planned order for id"


@pytest.mark.parametrize(
    ("first", "later"),
    [
        pytest.param(SubmitStatus.ERROR, SubmitStatus.DUPLICATE, id="acknowledged-later"),
        pytest.param(SubmitStatus.ACCEPTED, SubmitStatus.ERROR, id="acknowledged-first"),
    ],
)
def test_a_repeated_id_is_kept_once_from_the_earliest_report(
    first: SubmitStatus, later: SubmitStatus
) -> None:
    repriced = dataclasses.replace(BUY_AAA, ref_price=99.0)  # same id, other sizing price
    reports = [
        _report((repriced, later), started_at=LATER),  # given first: order comes from started_at
        _report((BUY_AAA, first), started_at=FIRST),
    ]
    orders, _ = submitted_orders(reports)
    assert orders == {_cid(BUY_AAA): _submitted(BUY_AAA, acknowledged=True)}


def test_an_id_planned_without_an_order_but_executed_elsewhere_is_not_unsettled() -> None:
    interrupted = dataclasses.replace(_report(planned=(_cid(BUY_AAA),)), plan=None)
    executed = _report((BUY_AAA, SubmitStatus.ACCEPTED), started_at=LATER)
    orders, unsettled = submitted_orders([interrupted, executed])
    assert orders == {_cid(BUY_AAA): _submitted(BUY_AAA, acknowledged=True, started_at=LATER)}
    assert unsettled == ()


def test_an_orphan_id_is_reported_once() -> None:
    reports = [
        dataclasses.replace(_report(planned=(GHOST,)), plan=None),
        dataclasses.replace(_report(planned=(GHOST,), started_at=LATER), plan=None),
    ]
    assert submitted_orders(reports) == ({}, (UnsettledOrder(GHOST, "", NO_PLANNED_ORDER),))


def test_no_reports_give_nothing() -> None:
    assert submitted_orders([]) == ({}, ())


def test_package_exports() -> None:
    import hedge_fund.engine.ledger as ledger

    assert ledger.__all__ == [
        "FILL_CUTOFF",
        "NO_PLANNED_ORDER",
        "OPENING_SOURCE",
        "build_lots",
        "canonical_symbol",
        "fill_deadline",
        "mark",
        "reconcile",
        "signed_quantity",
        "submitted_orders",
        "to_fill",
    ]
    assert ledger.submitted_orders is submitted_orders
