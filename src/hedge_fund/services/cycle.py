"""Layer 6d pipeline cycle: one paper-trading run from the account read to execution.

Every stage runs over injected ports; the CLI (``hedge_fund.cli``) is the only composition
root. Whatever happens, one CycleReport is saved: a cycle that raises saves a partial
ERROR / INTERRUPTED report naming the failed stage, then re-raises.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import date, datetime

from hedge_fund.domain.account import AccountSnapshot
from hedge_fund.domain.approval import ApprovalDecision
from hedge_fund.domain.cycle import CycleOptions, CycleOutcome, CycleReport
from hedge_fund.domain.execution import ExecutionReport
from hedge_fund.domain.orders import OrderPlan
from hedge_fund.domain.portfolio import TargetPortfolio
from hedge_fund.domain.research import ResearchResult
from hedge_fund.domain.risk import RiskVerdict
from hedge_fund.domain.shortlist import Shortlist
from hedge_fund.engine.account import nav_history
from hedge_fund.engine.orders import DEFAULT_ORDERS, OrderParams
from hedge_fund.engine.portfolio.alpha import DEFAULT_ALPHA, AlphaParams
from hedge_fund.engine.portfolio.correlation import DEFAULT_CORRELATION, CorrelationParams
from hedge_fund.engine.portfolio.optimize import DEFAULT_OPTIMIZE, OptimizeParams
from hedge_fund.engine.portfolio.risk_model import DEFAULT_RISK, RiskParams
from hedge_fund.engine.research.conviction import DEFAULT_CONVICTION, ConvictionParams
from hedge_fund.engine.risk.breakers import DEFAULT_BREAKERS, BreakerParams, check_breakers
from hedge_fund.engine.risk.limits import DEFAULT_LIMITS, RiskLimits
from hedge_fund.engine.shortlist.select import DEFAULT_SHORTLIST, ShortlistParams
from hedge_fund.ports.account import SnapshotStore
from hedge_fund.ports.approval import ApprovalGate
from hedge_fund.ports.broker import Broker
from hedge_fund.ports.cycle import CycleReportStore
from hedge_fund.ports.data import DataProvider
from hedge_fund.ports.fundamentals import FundamentalsProvider
from hedge_fund.ports.research import Researcher, ResearchStore
from hedge_fund.services.account import read_snapshot
from hedge_fund.services.research import _previous_weekday
from hedge_fund.universe import Universe


class ReportSaveError(Exception):
    """The cycle finished but its report could not be saved; carries ``report``.

    ``__cause__`` is the store's exception.
    """

    def __init__(self, report: CycleReport) -> None:
        super().__init__(f"the {report.outcome.value} cycle report could not be saved")
        self.report = report


class SubmissionWindowError(RuntimeError):
    """Outside dry-run, ``now()`` is inside the weekday session window (decision 7)."""


@dataclass(frozen=True)
class CycleParams:
    """Every engine param set the cycle passes on; defaults are the engine's own."""

    shortlist: ShortlistParams = DEFAULT_SHORTLIST
    conviction: ConvictionParams = DEFAULT_CONVICTION
    alpha: AlphaParams = DEFAULT_ALPHA
    risk: RiskParams = DEFAULT_RISK
    correlation: CorrelationParams = DEFAULT_CORRELATION
    optimize: OptimizeParams = DEFAULT_OPTIMIZE
    limits: RiskLimits = DEFAULT_LIMITS
    breakers: BreakerParams = DEFAULT_BREAKERS
    orders: OrderParams = DEFAULT_ORDERS


@dataclass(frozen=True)
class CycleDeps:
    """The ports one cycle runs over, and its params."""

    universe: Universe
    prices: DataProvider
    fundamentals: FundamentalsProvider
    researcher: Researcher
    research_store: ResearchStore
    broker: Broker
    snapshot_store: SnapshotStore
    gate: ApprovalGate
    report_store: CycleReportStore
    params: CycleParams = CycleParams()


@dataclass
class _Run:
    """What the cycle has reached so far; ``stage`` names the stage now running."""

    as_of: date
    started_at: datetime
    dry_run: bool
    stage: str = "inputs"
    account: AccountSnapshot | None = None
    price_unresolved: Mapping[str, str] = field(default_factory=dict)
    fundamentals_unresolved: Mapping[str, str] = field(default_factory=dict)
    shortlist: Shortlist | None = None
    research: ResearchResult | None = None
    research_reused: bool = False
    target: TargetPortfolio | None = None
    verdict: RiskVerdict | None = None
    order_prices: Mapping[str, tuple[float, date]] | None = None
    plan: OrderPlan | None = None
    decision: ApprovalDecision | None = None
    execution: ExecutionReport | None = None
    execution_interrupted: bool = False
    planned_client_order_ids: tuple[str, ...] = ()

    def report(self, outcome: CycleOutcome, error: BaseException | None) -> CycleReport:
        """The report so far; ``failed_stage`` and ``error`` only when ``error`` is set."""
        return CycleReport(
            as_of=self.as_of,
            started_at=self.started_at,
            dry_run=self.dry_run,
            outcome=outcome,
            failed_stage=None if error is None else self.stage,
            error=None if error is None else f"{type(error).__name__}: {error}",
            account=self.account,
            price_unresolved=self.price_unresolved,
            fundamentals_unresolved=self.fundamentals_unresolved,
            shortlist=self.shortlist,
            research=self.research,
            research_reused=self.research_reused,
            target=self.target,
            verdict=self.verdict,
            order_prices=self.order_prices,
            plan=self.plan,
            decision=self.decision,
            execution=self.execution,
            execution_interrupted=self.execution_interrupted,
            planned_client_order_ids=self.planned_client_order_ids,
        )


def run_cycle(
    deps: CycleDeps,
    *,
    as_of: date,
    today: date,
    options: CycleOptions,
    now: Callable[[], datetime],
) -> CycleReport:
    """Run one cycle for ``as_of``, save its report and return it.

    ``today`` is the date ``research_shortlist`` checks ``as_of`` against; ``now`` is read
    for ``started_at``, the account read and each submission-window check. A stage raising
    an ``Exception`` (ERROR) or a ``KeyboardInterrupt`` (INTERRUPTED) saves a partial
    report with ``failed_stage`` and ``error`` and re-raises the original exception; if
    that save fails too, the failure is added to it with ``add_note``. A completed cycle
    whose report cannot be saved raises ReportSaveError carrying the report.
    """
    run = _Run(as_of=as_of, started_at=now(), dry_run=options.dry_run)
    try:
        outcome = _cycle(deps, run, today=today, options=options, now=now)
    except BaseException as error:
        failed = CycleOutcome.ERROR if isinstance(error, Exception) else CycleOutcome.INTERRUPTED
        try:
            deps.report_store.save(run.report(failed, error))
        except Exception as save_error:
            error.add_note(
                f"the cycle report could not be saved: {type(save_error).__name__}: {save_error}"
            )
        raise
    report = run.report(outcome, None)
    try:
        deps.report_store.save(report)
    except Exception as save_error:
        raise ReportSaveError(report) from save_error
    return report


def _cycle(
    deps: CycleDeps,
    run: _Run,
    *,
    today: date,
    options: CycleOptions,
    now: Callable[[], datetime],
) -> CycleOutcome:
    """Steps 0-3; the data, research, portfolio, risk and order stages follow (Task 8)."""
    _check_live(run.as_of, today)
    run.stage = "account"
    current = read_snapshot(deps.broker, as_of=run.as_of, now=now)
    run.account = current
    if not options.dry_run:
        deps.snapshot_store.save(current)
    run.stage = "nav"
    nav = nav_history((*deps.snapshot_store.snapshots(), current), as_of=run.as_of)
    run.stage = "preflight"
    state = check_breakers(nav, params=deps.params.breakers)
    if state.daily_tripped or state.drawdown_tripped:
        return CycleOutcome.HALTED_PREFLIGHT
    raise NotImplementedError("stages 4-13 come in Tasks 8-10")


def _check_live(as_of: date, today: date) -> None:
    """``research_shortlist``'s rule, checked before any broker read."""
    earliest = _previous_weekday(today)
    if not earliest <= as_of <= today:
        raise ValueError(f"as_of {as_of} is not live: must be between {earliest} and {today}")
