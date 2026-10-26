"""Which client_order_ids settle must look up: every id a cycle may have put live.

Pure: reads cycle reports, prepares interrupted plans again, touches nothing else.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable

from hedge_fund.domain.cycle import CycleReport
from hedge_fund.domain.execution import OrderRequest, SubmitStatus
from hedge_fund.domain.ledger import SubmittedOrder, UnsettledOrder
from hedge_fund.domain.orders import OrderPlan
from hedge_fund.engine.execution.prepare import prepare_submission

NO_PLANNED_ORDER = "no planned order for id"

_ACKNOWLEDGED = frozenset({SubmitStatus.ACCEPTED, SubmitStatus.DUPLICATE})


def submitted_orders(
    reports: Iterable[CycleReport],
) -> tuple[dict[str, SubmittedOrder], tuple[UnsettledOrder, ...]]:
    """(id -> SubmittedOrder, unsettled ids that have no planned order), from non-dry-run reports.

    ACCEPTED and DUPLICATE legs are acknowledged; ERROR legs are not (outcome unknown);
    REJECTED, DEFERRED and SKIPPED legs are skipped. For an interrupted execution the
    planned ids are taken unacknowledged, with their requests from ``prepare_submission``;
    a planned id with no request (the plan is missing or cannot be prepared, or has no
    such order) becomes ``UnsettledOrder(id, "", NO_PLANNED_ORDER)`` unless another report
    supplies it. Reports apply by ``started_at``: a repeated id keeps the earliest report's
    request, ``as_of`` and ``started_at``, and is acknowledged if any report acknowledged it.
    Unsettled ids are sorted.
    """
    orders: dict[str, SubmittedOrder] = {}
    orphans: set[str] = set()
    live = [report for report in reports if not report.dry_run]
    for report in sorted(live, key=lambda report: report.started_at):
        found, missing = _report_orders(report)
        for request, acknowledged in found:
            _merge(orders, request, report, acknowledged=acknowledged)
        orphans.update(missing)
    unsettled = tuple(
        UnsettledOrder(client_order_id=cid, ticker="", reason=NO_PLANNED_ORDER)
        for cid in sorted(orphans - orders.keys())
    )
    return orders, unsettled


def _report_orders(report: CycleReport) -> tuple[list[tuple[OrderRequest, bool]], list[str]]:
    """((request, acknowledged) of each id the report may have put live, ids with no request)."""
    found: list[tuple[OrderRequest, bool]] = []
    if report.execution is not None:
        for result in report.execution.results:
            if result.status in _ACKNOWLEDGED:
                found.append((result.request, True))
            elif result.status is SubmitStatus.ERROR:
                found.append((result.request, False))
    if not report.execution_interrupted:
        return found, []
    planned = _planned_requests(report.plan)
    missing: list[str] = []
    for cid in report.planned_client_order_ids:
        if cid in planned:
            found.append((planned[cid], False))
        else:
            missing.append(cid)
    return found, missing


def _planned_requests(plan: OrderPlan | None) -> dict[str, OrderRequest]:
    if plan is None:
        return {}
    try:
        prepared = prepare_submission(plan)
    except ValueError:
        return {}
    return {request.client_order_id: request for request in (*prepared.reducing, *prepared.opening)}


def _merge(
    orders: dict[str, SubmittedOrder],
    request: OrderRequest,
    report: CycleReport,
    *,
    acknowledged: bool,
) -> None:
    existing = orders.get(request.client_order_id)
    if existing is None:
        orders[request.client_order_id] = SubmittedOrder(
            request=request,
            cycle_as_of=report.as_of,
            started_at=report.started_at,
            acknowledged=acknowledged,
        )
    elif acknowledged and not existing.acknowledged:
        orders[request.client_order_id] = dataclasses.replace(existing, acknowledged=True)
