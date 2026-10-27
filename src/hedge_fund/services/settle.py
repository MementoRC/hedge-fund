"""Layer 7a settlement: record final fills, rebuild FIFO lots, reconcile and mark them.

Run after the open on the day after a cycle submits; settle never writes to the broker.
Every stage runs over injected ports; the CLI (``hedge_fund.cli``) is the only composition
root. Whatever happens, one SettlementReport is saved: a run that raises saves a partial
ERROR / INTERRUPTED report naming the failed stage, then re-raises.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

from hedge_fund.domain.ledger import (
    ClosedLot,
    Fill,
    Lot,
    MarkedLot,
    PendingOrder,
    ReconciliationBreak,
    SettlementReport,
    SettleOutcome,
    SubmittedOrder,
    UnsettledOrder,
)
from hedge_fund.domain.orders import Holding
from hedge_fund.engine.dates import previous_weekday
from hedge_fund.engine.ledger import (
    build_lots,
    canonical_symbol,
    mark,
    reconcile,
    submitted_orders,
    to_fill,
)
from hedge_fund.ports.broker import Broker
from hedge_fund.ports.cycle import CycleReportStore
from hedge_fund.ports.data import DataProvider
from hedge_fund.ports.ledger import FillStore, OpeningBalance, SettlementReportStore

MARK_DAYS = 14  # calendar days of closes fetched to mark the open lots


class SettlementSaveError(Exception):
    """The settle finished but its report could not be saved; carries ``report``.

    ``__cause__`` is the store's exception.
    """

    def __init__(self, report: SettlementReport) -> None:
        super().__init__(f"the {report.outcome.value} settlement report could not be saved")
        self.report = report


@dataclass(frozen=True)
class SettleDeps:
    """The ports one settle runs over."""

    broker: Broker
    prices: DataProvider
    cycle_reports: CycleReportStore
    fills: FillStore
    opening: OpeningBalance
    report_store: SettlementReportStore


@dataclass
class _Settle:
    """What the run has reached so far; ``stage`` names the stage now running."""

    as_of: date
    started_at: datetime
    stage: str = "inputs"
    new_fills: list[Fill] = field(default_factory=list)
    pending: list[PendingOrder] = field(default_factory=list)
    unreadable: Mapping[str, str] = field(default_factory=dict)
    unsettled: list[UnsettledOrder] = field(default_factory=list)
    positions: tuple[Holding, ...] | None = None
    open_lots: tuple[MarkedLot, ...] = ()
    unmarked: tuple[Lot, ...] = ()
    closed_lots: tuple[ClosedLot, ...] = ()
    breaks: tuple[ReconciliationBreak, ...] = ()
    realised_total: float = 0.0
    unrealised_total: float = 0.0

    def outcome(self) -> SettleOutcome:
        """UNSETTLED > BREAK > PENDING (a pending order or an unreadable run file) > CLEAN."""
        if self.unsettled:
            return SettleOutcome.UNSETTLED
        if self.breaks:
            return SettleOutcome.BREAK
        if self.pending or self.unreadable:
            return SettleOutcome.PENDING
        return SettleOutcome.CLEAN

    def report(self, outcome: SettleOutcome, error: BaseException | None) -> SettlementReport:
        """The report so far; ``failed_stage`` and ``error`` only when ``error`` is set."""
        return SettlementReport(
            as_of=self.as_of,
            started_at=self.started_at,
            outcome=outcome,
            failed_stage=None if error is None else self.stage,
            error=None if error is None else f"{type(error).__name__}: {error}",
            new_fills=tuple(self.new_fills),
            pending=tuple(self.pending),
            unreadable_reports=self.unreadable,
            unsettled=tuple(self.unsettled),
            broker_positions=self.positions,
            open_lots=self.open_lots,
            unmarked=self.unmarked,
            closed_lots=self.closed_lots,
            breaks=self.breaks,
            realised_total=self.realised_total,
            unrealised_total=self.unrealised_total,
        )


def run_settle(
    deps: SettleDeps, *, as_of: date, today: date, now: Callable[[], datetime]
) -> SettlementReport:
    """Settle every order the stored cycles may have put live, save the report, return it.

    ``as_of`` dates the marks and must satisfy ``previous_weekday(today) <= as_of <= today``
    (the cycle's rule). ``now`` is read for ``started_at`` and for each recorded fill. A stage
    raising an ``Exception`` (ERROR) or a ``KeyboardInterrupt`` (INTERRUPTED) saves a partial
    report with ``failed_stage`` and ``error`` and re-raises the original exception; if that
    save fails too, the failure is added to it with ``add_note``. A completed run whose
    report cannot be saved raises SettlementSaveError carrying the report.
    """
    run = _Settle(as_of=as_of, started_at=now())
    try:
        _settle(deps, run, today=today, now=now)
    except BaseException as error:
        failed = SettleOutcome.ERROR if isinstance(error, Exception) else SettleOutcome.INTERRUPTED
        try:
            deps.report_store.save(run.report(failed, error))
        except Exception as save_error:
            error.add_note(
                "the settlement report could not be saved: "
                f"{type(save_error).__name__}: {save_error}"
            )
        raise
    report = run.report(run.outcome(), None)
    try:
        deps.report_store.save(report)
    except Exception as save_error:
        raise SettlementSaveError(report) from save_error
    return report


def _settle(deps: SettleDeps, run: _Settle, *, today: date, now: Callable[[], datetime]) -> None:
    """Steps 0-8; the caller saves the report (step 9)."""
    _check_live(run.as_of, today)
    run.stage = "reports"
    load = deps.cycle_reports.load_all()
    run.unreadable = load.unreadable
    orders, no_plan = submitted_orders(load.reports)
    run.unsettled.extend(no_plan)
    run.stage = "fills"
    recorded = deps.fills.load_all()
    run.stage = "lookup"
    done = {fill.client_order_id for fill in recorded}
    for client_order_id in sorted(orders.keys() - done):
        _look_up(deps, run, orders[client_order_id], now)
    run.stage = "opening"
    opening = deps.opening.load()
    run.stage = "lots"
    lots, run.closed_lots = build_lots(opening, (*recorded, *run.new_fills))
    run.stage = "positions"
    run.positions = deps.broker.positions()
    excluded = _order_tickers(run)
    run.breaks = reconcile(lots, run.positions, exclude=excluded)
    run.stage = "marks"
    if lots:
        tickers = sorted({lot.ticker for lot in lots})
        closes = deps.prices.closes(tickers, run.as_of - timedelta(days=MARK_DAYS), run.as_of)
        run.open_lots, run.unmarked = mark(lots, closes, as_of=run.as_of)
    _totals(run, excluded)


def _look_up(
    deps: SettleDeps, run: _Settle, order: SubmittedOrder, now: Callable[[], datetime]
) -> None:
    """Step 3 for one id: a final status becomes a fill, saved at once."""
    request = order.request
    status = deps.broker.order_status(request.client_order_id)
    fill = to_fill(status, request, cycle_as_of=order.cycle_as_of, recorded_at=now())
    _record(deps, run, fill)


def _record(deps: SettleDeps, run: _Settle, fill: Fill) -> None:
    """Save ``fill`` before going on, so a later failure keeps it."""
    deps.fills.save(fill)
    run.new_fills.append(fill)


def _order_tickers(run: _Settle) -> set[str]:
    """Canonical tickers of the pending and unsettled orders ("" never excludes anything)."""
    orders: list[PendingOrder | UnsettledOrder] = [*run.pending, *run.unsettled]
    return {canonical_symbol(order.ticker) for order in orders if order.ticker}


def _totals(run: _Settle, excluded: set[str]) -> None:
    """Both totals leave out tickers with a break, a pending or unsettled order, or an
    unmarked lot.
    """
    skipped = (
        excluded
        | {brk.ticker for brk in run.breaks}
        | {canonical_symbol(lot.ticker) for lot in run.unmarked}
    )
    run.realised_total = math.fsum(
        closed.realised_pnl
        for closed in run.closed_lots
        if canonical_symbol(closed.ticker) not in skipped
    )
    run.unrealised_total = math.fsum(
        marked.unrealised_pnl
        for marked in run.open_lots
        if canonical_symbol(marked.lot.ticker) not in skipped
    )


def _check_live(as_of: date, today: date) -> None:
    """The cycle's rule, checked before any read: ``as_of`` only dates the marks."""
    earliest = previous_weekday(today)
    if not earliest <= as_of <= today:
        raise ValueError(f"as_of {as_of} is not live: must be between {earliest} and {today}")
