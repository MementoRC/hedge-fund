"""Execution types: idempotent order requests in, per-order submit results and a report out."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import StrEnum

from hedge_fund.domain.orders import Order

REASON_FLIP_DEFERRED = "flip-open-leg-deferred"
REASON_REDUCING_FAILED = "reducing-leg-failed"


@dataclass(frozen=True)
class Account:
    """Broker account snapshot; ``equity`` is what 6d records as NAV."""

    equity: float


class SubmitStatus(StrEnum):
    """Outcome of one plan order.

    ACCEPTED: the broker took the order. DUPLICATE: an order with this client_order_id
    already exists; counts as success. REJECTED: the broker refused the order. ERROR: a
    BrokerError while submitting; outcome unknown, a re-run is safe. DEFERRED: the open leg
    of a side flip, left for the next cycle. SKIPPED: an opening leg not attempted because a
    reducing leg failed.
    """

    ACCEPTED = "accepted"
    DUPLICATE = "duplicate"
    REJECTED = "rejected"
    ERROR = "error"
    DEFERRED = "deferred"
    SKIPPED = "skipped"


@dataclass(frozen=True)
class OrderRequest:
    """One plan order with its deterministic ``client_order_id`` (the idempotency key)."""

    order: Order
    client_order_id: str


@dataclass(frozen=True)
class SubmitResult:
    """Outcome of one request.

    ``broker_order_id`` is set only for ACCEPTED (the new order) and DUPLICATE (the existing
    order). ``reason`` is set only for REJECTED (broker message, ``HTTP <status>`` or
    ``existing order <status>``), ERROR (``str`` of the BrokerError), DEFERRED
    (REASON_FLIP_DEFERRED) and SKIPPED (REASON_REDUCING_FAILED).
    """

    request: OrderRequest
    status: SubmitStatus
    broker_order_id: str | None
    reason: str | None


@dataclass(frozen=True)
class ExecutionReport:
    """One result per plan order, in plan order; ``blocked`` is copied from the plan.

    ``opening_halted`` is True when a reducing leg was REJECTED or ERROR, even when the plan
    has no opening legs.
    """

    as_of: date
    blocked: str | None
    results: tuple[SubmitResult, ...]
    opening_halted: bool
