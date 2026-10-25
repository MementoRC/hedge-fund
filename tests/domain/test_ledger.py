"""Tests for the ledger and settlement domain types."""

from __future__ import annotations

import dataclasses
from datetime import UTC, date, datetime
from decimal import Decimal

import pytest

from hedge_fund.domain.ledger import (
    ClosedLot,
    Fill,
    FillStatus,
    Lot,
    MarkedLot,
    OpeningEntry,
    OrderState,
    OrderStatus,
    PendingOrder,
    ReconciliationBreak,
    ReportLoad,
    SettlementReport,
    SettleOutcome,
    SubmittedOrder,
    UnsettledOrder,
)
from hedge_fund.domain.orders import OrderAction

AT = datetime(2026, 10, 1, 13, 30, tzinfo=UTC)
FILL = Fill(
    client_order_id="hf-2026-09-30-AAA-buy-10",
    broker_order_id="b-1",
    cycle_as_of=date(2026, 9, 30),
    ticker="AAA",
    action=OrderAction.BUY,
    requested_shares=10,
    filled_shares=10,
    avg_price=101.0,
    filled_at=AT,
    status=FillStatus.FILLED,
    recorded_at=AT,
)
STATUS = OrderStatus(
    client_order_id="hf-2026-09-30-AAA-buy-10",
    state=OrderState.FILLED,
    broker_order_id="b-1",
    ticker="AAA",
    side="buy",
    filled_shares=Decimal("10"),
    avg_price=101.0,
    filled_at=AT,
    broker_state="filled",
)
LOT = Lot(ticker="AAA", shares=10, entry_price=101.0, entered_at=AT, source=FILL.client_order_id)
REPORT = SettlementReport(
    as_of=date(2026, 10, 1),
    started_at=AT,
    outcome=SettleOutcome.CLEAN,
    failed_stage=None,
    error=None,
    new_fills=(FILL,),
    pending=(),
    unreadable_reports={},
    unsettled=(),
    broker_positions=None,
    open_lots=(),
    unmarked=(LOT,),
    closed_lots=(),
    breaks=(),
    realised_total=0.0,
    unrealised_total=0.0,
)


def test_fill_status_values() -> None:
    assert [status.value for status in FillStatus] == [
        "filled",
        "partial-terminal",
        "canceled",
        "rejected",
        "expired",
        "not-submitted",
    ]


def test_order_state_values() -> None:
    assert [state.value for state in OrderState] == [
        "filled",
        "partial-terminal",
        "canceled",
        "rejected",
        "expired",
        "pending",
        "not-found",
        "unrecognised",
    ]


def test_settle_outcome_values() -> None:
    assert [outcome.value for outcome in SettleOutcome] == [
        "clean",
        "pending",
        "break",
        "unsettled",
        "error",
        "interrupted",
    ]


def test_fill_field_order() -> None:
    assert [field.name for field in dataclasses.fields(Fill)] == [
        "client_order_id",
        "broker_order_id",
        "cycle_as_of",
        "ticker",
        "action",
        "requested_shares",
        "filled_shares",
        "avg_price",
        "filled_at",
        "status",
        "recorded_at",
    ]


def test_report_field_order() -> None:
    assert [field.name for field in dataclasses.fields(SettlementReport)] == [
        "as_of",
        "started_at",
        "outcome",
        "failed_stage",
        "error",
        "new_fills",
        "pending",
        "unreadable_reports",
        "unsettled",
        "broker_positions",
        "open_lots",
        "unmarked",
        "closed_lots",
        "breaks",
        "realised_total",
        "unrealised_total",
    ]


def test_types_compare_by_value() -> None:
    assert dataclasses.replace(REPORT) == REPORT
    assert dataclasses.replace(REPORT, outcome=SettleOutcome.BREAK) != REPORT
    assert dataclasses.replace(STATUS, filled_shares=Decimal("10.0")) == STATUS


@pytest.mark.parametrize(
    ("instance", "field"),
    [
        pytest.param(FILL, "ticker", id="fill"),
        pytest.param(STATUS, "state", id="order-status"),
        pytest.param(LOT, "shares", id="lot"),
        pytest.param(
            ClosedLot("AAA", 10, 100.0, AT, "opening", 101.0, AT, "x", 10.0), "shares", id="closed"
        ),
        pytest.param(MarkedLot(LOT, 102.0, date(2026, 10, 1), 10.0), "price", id="marked"),
        pytest.param(OpeningEntry("AAA", 10, 100.0, date(2026, 9, 1)), "shares", id="opening"),
        pytest.param(ReconciliationBreak("AAA", 10, 12), "broker_shares", id="break"),
        pytest.param(PendingOrder("x", "AAA", "open at broker: new"), "reason", id="pending"),
        pytest.param(UnsettledOrder("x", "AAA", "why"), "reason", id="unsettled"),
        pytest.param(ReportLoad(reports=(), unreadable={}), "reports", id="report-load"),
        pytest.param(REPORT, "outcome", id="report"),
    ],
)
def test_types_are_frozen(instance: object, field: str) -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(instance, field, None)


def test_submitted_order_is_frozen() -> None:
    from hedge_fund.domain.execution import OrderRequest
    from hedge_fund.domain.orders import Order

    order = Order(ticker="AAA", action=OrderAction.BUY, shares=10, ref_price=100.0, notional=1000.0)
    submitted = SubmittedOrder(
        request=OrderRequest(order=order, client_order_id=FILL.client_order_id),
        cycle_as_of=date(2026, 9, 30),
        started_at=AT,
        acknowledged=True,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        submitted.acknowledged = False  # type: ignore[misc]
