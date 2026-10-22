"""Layer 6a execution: submit an approved order plan, reducing legs first.

Approval (6c), fetching holdings, NAV and prices, cancelling pending orders and fill
reconciliation (6d) are out of scope.
"""

from __future__ import annotations

from hedge_fund.domain.execution import (
    REASON_FLIP_DEFERRED,
    REASON_REDUCING_FAILED,
    ExecutionReport,
    OrderRequest,
    SubmitResult,
    SubmitStatus,
)
from hedge_fund.domain.orders import OrderPlan
from hedge_fund.engine.execution import prepare_submission
from hedge_fund.ports.broker import Broker, BrokerError

_FAILED = frozenset({SubmitStatus.REJECTED, SubmitStatus.ERROR})


def execute_plan(plan: OrderPlan, broker: Broker) -> ExecutionReport:
    """Submit every reducing leg, then the opening legs unless a reducing leg failed.

    The plan is prepared first, so a malformed plan raises ValueError before any broker
    call. A blocked plan returns an empty report with no broker calls: the risk veto stands.
    A BrokerError from ``submit`` becomes an ERROR result and the run carries on; any other
    exception propagates. A REJECTED or ERROR reducing leg sets ``opening_halted`` and every
    opening leg becomes SKIPPED; otherwise flip open legs are DEFERRED and the other opening
    legs are submitted. Results are one per plan order, in plan order.
    """
    prepared = prepare_submission(plan)
    if plan.blocked is not None:
        return ExecutionReport(
            as_of=plan.as_of, blocked=plan.blocked, results=(), opening_halted=False
        )
    reducing = [_submit(broker, request) for request in prepared.reducing]
    halted = any(result.status in _FAILED for result in reducing)
    opening = [
        _open(broker, request, prepared.deferred, halted=halted) for request in prepared.opening
    ]
    return ExecutionReport(
        as_of=plan.as_of,
        blocked=None,
        results=tuple(reducing + opening),
        opening_halted=halted,
    )


def _open(
    broker: Broker, request: OrderRequest, deferred: frozenset[str], *, halted: bool
) -> SubmitResult:
    """Halt outranks deferral: after a failed reducing leg a flip open leg is SKIPPED."""
    if halted:
        return _unsent(request, SubmitStatus.SKIPPED, REASON_REDUCING_FAILED)
    if request.client_order_id in deferred:
        return _unsent(request, SubmitStatus.DEFERRED, REASON_FLIP_DEFERRED)
    return _submit(broker, request)


def _submit(broker: Broker, request: OrderRequest) -> SubmitResult:
    """One broker call; a BrokerError becomes ERROR so the run carries on."""
    try:
        return broker.submit(request)
    except BrokerError as exc:
        return _unsent(request, SubmitStatus.ERROR, str(exc))


def _unsent(request: OrderRequest, status: SubmitStatus, reason: str) -> SubmitResult:
    return SubmitResult(request=request, status=status, broker_order_id=None, reason=reason)
