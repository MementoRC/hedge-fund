"""Execution domain types."""

from __future__ import annotations

import dataclasses
from datetime import date

import pytest

from hedge_fund.domain.execution import (
    REASON_FLIP_DEFERRED,
    REASON_REDUCING_FAILED,
    Account,
    ExecutionReport,
    OrderRequest,
    SubmitResult,
    SubmitStatus,
)
from hedge_fund.domain.orders import Order, OrderAction

ORDER = Order(ticker="A", action=OrderAction.BUY, shares=5, ref_price=10.0, notional=50.0)
REQUEST = OrderRequest(order=ORDER, client_order_id="hf-2026-10-21-A-buy-5")


def test_submit_status_values() -> None:
    assert [status.value for status in SubmitStatus] == [
        "accepted",
        "duplicate",
        "rejected",
        "error",
        "deferred",
        "skipped",
    ]
    assert SubmitStatus.DUPLICATE == "duplicate"


def test_reason_constants() -> None:
    assert (REASON_FLIP_DEFERRED, REASON_REDUCING_FAILED) == (
        "flip-open-leg-deferred",
        "reducing-leg-failed",
    )


@pytest.mark.parametrize(
    ("instance", "field"),
    [
        pytest.param(Account(equity=1.0), "equity", id="account"),
        pytest.param(REQUEST, "client_order_id", id="request"),
        pytest.param(
            SubmitResult(
                request=REQUEST, status=SubmitStatus.ACCEPTED, broker_order_id="b", reason=None
            ),
            "status",
            id="result",
        ),
        pytest.param(
            ExecutionReport(
                as_of=date(2026, 10, 21), blocked=None, results=(), opening_halted=False
            ),
            "opening_halted",
            id="report",
        ),
    ],
)
def test_types_are_frozen(instance: object, field: str) -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(instance, field, None)
