"""Layer 6d pipeline cycle: one paper-trading run from the account read to execution.

Every stage runs over injected ports; the CLI (``hedge_fund.cli``) is the only composition
root. Whatever happens, one CycleReport is saved: a cycle that raises saves a partial
ERROR / INTERRUPTED report naming the failed stage, then re-raises.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

import pandas as pd

from hedge_fund.domain.account import AccountSnapshot
from hedge_fund.domain.approval import ApprovalDecision, ApprovalStatus
from hedge_fund.domain.cost import BudgetPlan, PriceTable, Spend
from hedge_fund.domain.cycle import CycleOptions, CycleOutcome, CycleReport
from hedge_fund.domain.execution import ExecutionReport
from hedge_fund.domain.fundamentals import FundamentalsSnapshot
from hedge_fund.domain.ledger import SubmittedOrder
from hedge_fund.domain.orders import OrderPlan
from hedge_fund.domain.portfolio import TargetPortfolio
from hedge_fund.domain.research import ResearchResult
from hedge_fund.domain.risk import NavHistory, RiskVerdict
from hedge_fund.domain.shortlist import Shortlist
from hedge_fund.domain.snapshot import PriceSnapshot
from hedge_fund.engine.account import nav_history
from hedge_fund.engine.cost import check_budget, estimate, history_costs, spend
from hedge_fund.engine.cost import plan as plan_budget
from hedge_fund.engine.dates import previous_weekday
from hedge_fund.engine.execution import (
    MARKET_OPEN,
    MARKET_TZ,
    SESSION_END,
    SESSION_START,
    in_session_window,
    prepare_submission,
    session_opened,
)
from hedge_fund.engine.factors.score import score
from hedge_fund.engine.ledger.deadline import fill_deadline
from hedge_fund.engine.ledger.open_order import is_open
from hedge_fund.engine.ledger.submitted import submitted_orders
from hedge_fund.engine.orders import DEFAULT_ORDERS, OrderParams, generate_orders, order_prices
from hedge_fund.engine.portfolio.alpha import DEFAULT_ALPHA, AlphaParams
from hedge_fund.engine.portfolio.correlation import DEFAULT_CORRELATION, CorrelationParams
from hedge_fund.engine.portfolio.optimize import DEFAULT_OPTIMIZE, OptimizeParams
from hedge_fund.engine.portfolio.risk_model import DEFAULT_RISK, RiskParams
from hedge_fund.engine.research.conviction import DEFAULT_CONVICTION, ConvictionParams
from hedge_fund.engine.risk.breakers import DEFAULT_BREAKERS, BreakerParams, check_breakers
from hedge_fund.engine.risk.limits import DEFAULT_LIMITS, RiskLimits
from hedge_fund.engine.shortlist.select import DEFAULT_SHORTLIST, ShortlistParams, shortlist
from hedge_fund.ports.account import SnapshotStore
from hedge_fund.ports.approval import ApprovalGate
from hedge_fund.ports.broker import Broker, OrderDataError
from hedge_fund.ports.cycle import CycleReportStore
from hedge_fund.ports.data import DataProvider
from hedge_fund.ports.fundamentals import FundamentalsProvider
from hedge_fund.ports.research import Researcher, ResearchStore
from hedge_fund.services.account import read_snapshot
from hedge_fund.services.approval import request_approval
from hedge_fund.services.execution import execute_plan
from hedge_fund.services.fundamentals import build_fundamentals_snapshot
from hedge_fund.services.portfolio import build_portfolio, fetch_benchmark
from hedge_fund.services.research import research_shortlist
from hedge_fund.services.risk import assess_portfolio
from hedge_fund.services.snapshots import build_snapshot
from hedge_fund.universe import Universe

HELD_PRICE_DAYS = 14  # calendar days of closes fetched for held names outside the snapshot
HISTORY_RUNS = 30  # stored run costs the budget estimate is the median of (newest first)
HISTORY_DATES = 60  # stored research dates read for them, at most (newest first)

_NOT_APPROVED: Mapping[ApprovalStatus, CycleOutcome] = {
    ApprovalStatus.BLOCKED: CycleOutcome.BLOCKED,
    ApprovalStatus.NOTHING_TO_DO: CycleOutcome.NOTHING_TO_DO,
    ApprovalStatus.REJECTED: CycleOutcome.REJECTED,
}


class ReportSaveError(Exception):
    """The cycle finished but its report could not be saved; carries ``report``.

    ``__cause__`` is the store's exception.
    """

    def __init__(self, report: CycleReport) -> None:
        super().__init__(f"the {report.outcome.value} cycle report could not be saved")
        self.report = report


class SubmissionWindowError(RuntimeError):
    """Outside dry-run, ``now()`` is inside the weekday session window (decision 7)."""


class PendingOrdersError(RuntimeError):
    """Outside dry-run, orders an earlier cycle submitted are still open at the broker."""


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
    """The ports one cycle runs over, its params, and the LLM price table (7c)."""

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
    price_table: PriceTable = field(default_factory=PriceTable)


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
    budget: BudgetPlan | None = None
    spend: Spend | None = None

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
            budget=self.budget,
            spend=self.spend,
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

    ``as_of`` is ``today``, or the previous weekday while ``started_at`` is before the 09:30
    New York open on ``today``: after it the account's equity is no longer that day's close.
    ``today`` is the date ``research_shortlist`` checks ``as_of`` against; ``now`` is read
    for ``started_at``, the account read and each submission-window check (and, once, for
    the open-orders check when an earlier non-dry-run report exists). Outside dry-run, a
    cycle whose plan would submit raises PendingOrdersError, before the approval, while an
    earlier cycle's orders (other than this plan's own ids) are still open at the broker;
    re-running the same ``as_of`` after an interrupted execution is still allowed. A stage
    raising an ``Exception`` (ERROR) or a ``KeyboardInterrupt`` (INTERRUPTED) saves a
    partial report with ``failed_stage`` and ``error`` and re-raises the original exception;
    if that save fails too, the failure is added to it with ``add_note``. A completed cycle
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
    """Steps 0-12; the caller saves the report (step 13).

    A budget that fits no name (7c) ends the cycle NOTHING_TO_DO after the shortlist: no
    portfolio, no orders, no approval, no broker write. A dry run ends before the approval
    and the open-orders check (no ``order_status`` lookup).
    """
    _check_live(run.as_of, today)
    _check_not_a_moving_read(run.as_of, today, run.started_at)
    if options.budget is not None:
        check_budget(options.budget)
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
    planned = _plan(deps, run, current, nav, today=today, options=options)
    if planned is None:
        return CycleOutcome.NOTHING_TO_DO
    verdict, plan = planned
    if options.dry_run:
        return CycleOutcome.DRY_RUN
    return _approve_and_execute(deps, run, verdict, plan, options=options, now=now)


def _plan(
    deps: CycleDeps,
    run: _Run,
    current: AccountSnapshot,
    nav: NavHistory,
    *,
    today: date,
    options: CycleOptions,
) -> tuple[RiskVerdict, OrderPlan] | None:
    """Steps 4-10: data, shortlist, research, portfolio, risk, prices and orders.

    None when a budget skip (7c) stops the cycle after the shortlist.
    """
    params = deps.params
    snapshot, fundamentals = _data(deps, run)
    run.stage = "shortlist"
    candidates = shortlist(score(snapshot, fundamentals), params=params.shortlist)
    run.shortlist = candidates
    run.stage = "research"
    research = _research(deps, run, candidates, today=today, options=options)
    if research is None:
        return None
    run.research = research
    run.stage = "portfolio"
    benchmark = fetch_benchmark(deps.prices, run.as_of, lookback_days=params.risk.lookback)
    target = build_portfolio(
        research,
        snapshot,
        benchmark,
        alpha=params.alpha,
        risk=params.risk,
        correlation=params.correlation,
        optimize=params.optimize,
    )
    run.target = target
    run.stage = "risk"
    verdict = assess_portfolio(
        target,
        snapshot,
        benchmark,
        nav=nav,
        limits=params.limits,
        breakers=params.breakers,
        risk=params.risk,
    )
    run.verdict = verdict
    run.stage = "prices"
    prices = _order_prices(deps.prices, snapshot, target, current, as_of=run.as_of)
    run.order_prices = prices
    run.stage = "orders"
    sizing = {ticker: price for ticker, (price, _) in prices.items()}
    plan = generate_orders(
        target, verdict, current.holdings, sizing, current.equity, params=params.orders
    )
    run.plan = plan
    return verdict, plan


def _approve_and_execute(
    deps: CycleDeps,
    run: _Run,
    verdict: RiskVerdict,
    plan: OrderPlan,
    *,
    options: CycleOptions,
    now: Callable[[], datetime],
) -> CycleOutcome:
    """Steps 11-12: window check, open-orders check, approval, window re-check on APPROVED,
    then execution.

    The open-orders check runs only for a plan that would submit (not blocked, not empty).
    """
    _check_window(run, options, now)
    if plan.blocked is None and plan.orders:
        _check_pending(deps, run, plan, now)
    run.stage = "approval"
    decision = request_approval(verdict, plan, deps.gate)
    run.decision = decision
    if decision.status is not ApprovalStatus.APPROVED:
        return _NOT_APPROVED[decision.status]
    _check_window(run, options, now)  # a human may have answered after 09:25 ET
    run.stage = "execution"
    try:
        run.execution = execute_plan(plan, deps.broker)
    except BaseException:
        run.execution_interrupted = True
        run.planned_client_order_ids = _planned_ids(plan)
        raise
    return CycleOutcome.SUBMITTED


def _check_window(run: _Run, options: CycleOptions, now: Callable[[], datetime]) -> None:
    """Decision 7: refuse, with a fresh ``now()``, inside the weekday session window."""
    if options.any_time:
        return
    run.stage = "submission_window"
    at = now()
    if in_session_window(at):
        raise SubmissionWindowError(
            f"{at.astimezone(MARKET_TZ):%a %Y-%m-%d %H:%M} ET is inside the weekday session "
            f"window {SESSION_START:%H:%M}-{SESSION_END:%H:%M} ET: opg orders must be "
            "submitted before 09:25 ET or after the close"
        )


def _check_pending(
    deps: CycleDeps, run: _Run, plan: OrderPlan, now: Callable[[], datetime]
) -> None:
    """Refuse while an earlier cycle's orders may still fill: opg orders queued for another
    ``as_of`` are invisible to the account read, so this cycle would size against flat
    holdings and double the book past what the risk veto approved.

    Looks up, at the broker, the ids of earlier non-dry-run reports whose fill deadline is
    still ahead, leaving out this plan's own ids (a re-run of the same ``as_of`` keeps the
    broker's DUPLICATE path). ``now`` is read only when such reports exist.
    """
    run.stage = "pending_orders"
    reports = [r for r in deps.report_store.load_all().reports if not r.dry_run]
    if not reports:
        return
    at = now()
    live = [report for report in reports if fill_deadline(report.started_at) > at]
    orders, _ = submitted_orders(live)
    own = set(_planned_ids(plan))
    still_open = [
        orders[cid] for cid in sorted(orders.keys() - own) if _is_open(deps.broker, orders[cid], at)
    ]
    if still_open:
        listed = ", ".join(
            f"{order.request.client_order_id} (as_of {order.cycle_as_of})" for order in still_open
        )
        raise PendingOrdersError(
            f"{len(still_open)} order(s) from an earlier cycle are still open at the broker: "
            f"{listed}; settle or wait for the open before running another cycle"
        )


def _is_open(broker: Broker, order: SubmittedOrder, now: datetime) -> bool:
    """Settle's reading of the broker's answer; a malformed one counts as open."""
    try:
        status = broker.order_status(order.request.client_order_id)
    except OrderDataError:
        return True
    return is_open(
        status.state,
        acknowledged=order.acknowledged,
        deadline=fill_deadline(order.started_at),
        now=now,
    )


def _planned_ids(plan: OrderPlan) -> tuple[str, ...]:
    """The plan's client_order_ids in submission order; ``()`` if it cannot be prepared."""
    try:
        prepared = prepare_submission(plan)
    except ValueError:
        return ()
    return tuple(request.client_order_id for request in (*prepared.reducing, *prepared.opening))


def _data(deps: CycleDeps, run: _Run) -> tuple[PriceSnapshot, FundamentalsSnapshot]:
    """Step 4: prices for the universe, then fundamentals for the names that have prices."""
    run.stage = "data"
    prices = build_snapshot(deps.prices, deps.universe.tickers, run.as_of)
    run.price_unresolved = prices.unresolved
    priced = [ticker for ticker in prices.snapshot.sectors if ticker not in prices.unresolved]
    fundamentals = build_fundamentals_snapshot(
        deps.fundamentals, priced, run.as_of, ciks=deps.universe.ciks
    )
    run.fundamentals_unresolved = fundamentals.unresolved
    return prices.snapshot, fundamentals.snapshot


def _research(
    deps: CycleDeps, run: _Run, candidates: Shortlist, *, today: date, options: CycleOptions
) -> ResearchResult | None:
    """Step 6: reuse today's stored research if it matches (decision 2), else research the
    shortlist and save the result (a cache, so dry runs save too).

    With ``options.budget`` the ``budget`` stage first cuts K, then N (7c). A plan that
    leaves no name returns None (the cycle stops) and saves nothing, so a later cycle still
    researches. ``run.spend`` covers the runs made here (none when reused or skipped).
    """
    table = deps.price_table
    min_k = None if options.budget is None else min(deps.params.conviction.min_valid, options.k)
    stored = None if options.fresh_research else deps.research_store.load(run.as_of)
    if stored is not None and _reusable(stored, candidates, k=options.k, min_k=min_k):
        run.research_reused = True
        run.spend = spend((), table, options.budget)
        return stored
    k, planned = options.k, candidates
    if options.budget is not None:
        budget = _budget(deps, run, candidates, budget=options.budget, k=options.k, today=today)
        if budget.skipped:
            run.spend = spend((), table, options.budget)
            return None
        k = budget.k
        planned = Shortlist(
            as_of=candidates.as_of,
            candidates=candidates.candidates[: budget.n],
            trimmed=candidates.trimmed,
        )
    result = research_shortlist(
        deps.researcher,
        planned,
        k=k,
        max_workers=options.max_workers,
        params=deps.params.conviction,
        today=today,
    )
    run.spend = spend(
        (record for name in result.names for record in name.runs), table, options.budget
    )
    deps.research_store.save(result)
    return result


def _reusable(stored: ResearchResult, candidates: Shortlist, *, k: int, min_k: int | None) -> bool:
    """No error run, and: without a budget (``min_k`` None) the shortlist's tickers in its
    order with ``k`` runs per name; with one, a non-empty prefix of them with at least
    ``min_k`` runs per name (what a budgeted cycle stores).
    """
    names = stored.names
    tickers = [name.candidate.ticker for name in names]
    wanted = [c.ticker for c in candidates.candidates]
    if min_k is None:
        shaped = tickers == wanted and all(len(name.runs) == k for name in names)
    else:
        shaped = (
            bool(tickers)
            and tickers == wanted[: len(tickers)]
            and all(len(name.runs) >= min_k for name in names)
        )
    return shaped and all(record.error is None for name in names for record in name.runs)


def _budget(
    deps: CycleDeps,
    run: _Run,
    candidates: Shortlist,
    *,
    budget: float,
    k: int,
    today: date,
) -> BudgetPlan:
    """The ``budget`` stage, before any LLM call: ``research_shortlist``'s live-date check,
    the per-run estimate (median of the newest ``HISTORY_RUNS`` known run costs from the
    stored research, else the table's fallback; BudgetError with neither) and the plan.
    The stage is reset to ``research`` afterwards.
    """
    run.stage = "budget"
    _check_live(candidates.as_of, today)
    table = deps.price_table
    history = history_costs(_history(deps.research_store, run.as_of), table, window=HISTORY_RUNS)
    planned = plan_budget(
        budget,
        estimate(history, table.fallback_run_cost),
        n=len(candidates.candidates),
        k=k,
        min_k=deps.params.conviction.min_valid,
    )
    run.budget = planned
    run.stage = "research"
    return planned


def _history(store: ResearchStore, as_of: date) -> Iterator[ResearchResult]:
    """Stored results newest first, at most ``HISTORY_DATES`` dates, ``as_of`` itself skipped.

    Lazy, so ``history_costs`` stops the loads once it has its costs. Best effort: a date
    whose stored file is malformed (``load`` raises ValueError) is skipped.
    """
    days = [day for day in reversed(store.dates()) if day != as_of][:HISTORY_DATES]
    for day in days:
        result = _load_or_none(store, day)
        if result is not None:
            yield result


def _load_or_none(store: ResearchStore, day: date) -> ResearchResult | None:
    try:
        return store.load(day)
    except ValueError:
        return None


def _order_prices(
    provider: DataProvider,
    snapshot: PriceSnapshot,
    target: TargetPortfolio,
    current: AccountSnapshot,
    *,
    as_of: date,
) -> dict[str, tuple[float, date]]:
    """Step 9: a price for every target and held name; held names outside the snapshot
    are fetched over the last ``HELD_PRICE_DAYS`` calendar days.
    """
    needed = sorted(
        {position.ticker for position in target.positions}
        | {holding.ticker for holding in current.holdings}
    )
    closes = snapshot.closes
    missing = [ticker for ticker in needed if ticker not in closes.columns]
    if missing:
        fetched = provider.closes(missing, as_of - timedelta(days=HELD_PRICE_DAYS), as_of)
        closes = pd.concat([closes, fetched], axis=1)
    return order_prices(closes, needed, as_of=as_of)


def _check_not_a_moving_read(as_of: date, today: date, started_at: datetime) -> None:
    """The cycle dates the broker's equity with ``as_of``: after the open it is not a close."""
    if as_of != today and session_opened(today, started_at):
        raise ValueError(
            f"as_of {as_of} is not live: it is the previous weekday, but today's session "
            f"opened at {MARKET_OPEN:%H:%M} New York time; use as_of {today}"
        )


def _check_live(as_of: date, today: date) -> None:
    """``research_shortlist``'s rule, checked before any broker read."""
    earliest = previous_weekday(today)
    if not earliest <= as_of <= today:
        raise ValueError(f"as_of {as_of} is not live: must be between {earliest} and {today}")
