"""Layer 8 backtest: replay the unchanged ``run_cycle`` over a frozen point-in-time store.

Every substitute (simulated broker, frozen providers, neutral researcher, always-yes gate)
arrives through ``BacktestDeps``; the CLI's ``build_backtest_deps`` is the composition root.
The replay touches no network and no LLM, and runs on a simulated clock.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import date

import pandas as pd

from hedge_fund.domain.backtest import (
    MEMBERSHIP_FLOOR,
    BacktestReport,
    FrozenData,
    SimCosts,
)
from hedge_fund.domain.cycle import CycleOptions, CycleOutcome, CycleReport
from hedge_fund.domain.risk import NavHistory, NavPoint
from hedge_fund.domain.shortlist import Shortlist
from hedge_fund.engine.backtest.calendar import in_window, rebalance_days, trading_days
from hedge_fund.engine.backtest.clock import SimClock
from hedge_fund.engine.backtest.membership import PointInTimeUniverse
from hedge_fund.engine.backtest.metrics import backtest_metrics
from hedge_fund.engine.backtest.pit_prices import closes_as_of
from hedge_fund.ports.account import SnapshotStore
from hedge_fund.ports.approval import ApprovalGate
from hedge_fund.ports.cycle import CycleReportStore
from hedge_fund.ports.data import DataProvider
from hedge_fund.ports.fundamentals import FundamentalsProvider
from hedge_fund.ports.research import Researcher, ResearchStore
from hedge_fund.ports.sim_broker import SimBroker
from hedge_fund.services.account import load_nav_history, record_snapshot
from hedge_fund.services.cycle import CycleDeps, CycleParams, plan_shortlist, run_cycle

MIN_TRADING_DAYS = 3  # the metrics need a curve of at least three points
BENCHMARK = "SPY"

ACCEPTED_OUTCOMES = frozenset(
    {
        CycleOutcome.SUBMITTED,
        CycleOutcome.NOTHING_TO_DO,
        CycleOutcome.BLOCKED,
        CycleOutcome.HALTED_PREFLIGHT,
    }
)

CycleRunner = Callable[..., CycleReport]


class BacktestError(Exception):
    """A replay that must abort (non-zero exit, clear message)."""


@dataclass(frozen=True, eq=False)
class BacktestDeps:
    """The frozen store, the run's inputs and the substitutes the replay injects.

    ``prices(day)`` builds the data provider of one cycle date and ``researcher(shortlist)``
    the research stand-in for one shortlist.
    """

    data: FrozenData
    store_digest: str
    cash: float
    costs: SimCosts
    broker: SimBroker
    snapshot_store: SnapshotStore
    report_store: CycleReportStore
    research_store: ResearchStore
    gate: ApprovalGate
    prices: Callable[[date], DataProvider]
    fundamentals: FundamentalsProvider
    researcher: Callable[[Shortlist], Researcher]
    params: CycleParams = field(default_factory=CycleParams)


def backtest_run_id(deps: BacktestDeps, start: date, end: date) -> str:
    """``<start>_<end>_<sha12>``: the hash covers the digest, window, cash, costs and params."""
    key = repr((deps.store_digest, start, end, deps.cash, deps.costs, deps.params))
    return f"{start}_{end}_{hashlib.sha256(key.encode()).hexdigest()[:12]}"


def run_backtest(
    deps: BacktestDeps,
    *,
    start: date,
    end: date,
    cycle: CycleRunner = run_cycle,
    clock: SimClock | None = None,
) -> BacktestReport:
    """Replay ``cycle`` on each monthly rebalance day of ``start..end`` and summarise the run.

    ``cycle`` is a test seam. Raises BacktestError when a guard refuses the run, when a cycle
    raises, ends in an outcome outside ACCEPTED_OUTCOMES or builds another shortlist than the
    one the researcher was given, and when the benchmark has no close on a window day.
    """
    _check_guards(deps, start, end)
    clock = clock or SimClock()
    data = deps.data
    calendar = trading_days(data.closes.index)
    window = in_window(calendar, start, end)
    if len(window) < MIN_TRADING_DAYS:
        raise BacktestError("the window needs at least three trading days")
    rebalance = rebalance_days(calendar) & set(window)
    universe = PointInTimeUniverse(data.membership, data.sectors, data.ciks, flagged=data.flagged)
    reports: list[CycleReport] = []
    flagged: set[str] = set()
    for day in window:
        clock.start_day(day)
        deps.broker.open(day)
        deps.broker.close(day)
        record_snapshot(deps.broker, deps.snapshot_store, as_of=day, now=clock)
        if day in rebalance:
            flagged.update(universe.excluded_on(day))
            reports.append(_rebalance(deps, universe, day, cycle, clock))
    metrics = backtest_metrics(
        load_nav_history(deps.snapshot_store, as_of=end),
        _benchmark(data, window, end),
        deps.broker.events,
        reports,
        flagged=flagged,
        params=deps.params.breakers,
    )
    return BacktestReport(
        run_id=backtest_run_id(deps, start, end),
        store_digest=deps.store_digest,
        start=start,
        end=end,
        cash=deps.cash,
        costs=deps.costs,
        trading_days=len(window),
        rebalances=len(reports),
        metrics=metrics,
    )


def _check_guards(deps: BacktestDeps, start: date, end: date) -> None:
    if not isinstance(deps.broker, SimBroker):
        raise BacktestError("the backtest only runs against SimulatedBroker")
    runs = deps.report_store.load_all()
    if runs.unreadable:
        raise BacktestError("--root already holds unreadable run files")
    if deps.snapshot_store.snapshots() or runs.reports:
        raise BacktestError("--root already holds snapshots or runs")
    if start < MEMBERSHIP_FLOOR:
        raise BacktestError(
            f"--start {start} is before {MEMBERSHIP_FLOOR}: membership is not reliable earlier"
        )
    if start < deps.data.start or end > deps.data.end or start > end:
        raise BacktestError(
            f"window outside the store's coverage: {start}..{end} "
            f"against {deps.data.start}..{deps.data.end}"
        )


def _rebalance(
    deps: BacktestDeps,
    universe: PointInTimeUniverse,
    day: date,
    cycle: CycleRunner,
    clock: SimClock,
) -> CycleReport:
    """One cycle on ``day``, fed the shortlist it will build, checked against what it did."""
    base = CycleDeps(
        universe=universe.members(day),
        prices=deps.prices(day),
        fundamentals=deps.fundamentals,
        researcher=deps.researcher(Shortlist(as_of=day, candidates=(), trimmed={})),
        research_store=deps.research_store,
        broker=deps.broker,
        snapshot_store=deps.snapshot_store,
        gate=deps.gate,
        report_store=deps.report_store,
        params=deps.params,
    )
    try:
        planned = plan_shortlist(base, day)
    except Exception as exc:
        raise BacktestError(f"shortlist planning failed on {day}: {exc}") from exc
    options = CycleOptions(any_time=True, fresh_research=True, budget=None)
    try:
        report = cycle(
            replace(base, researcher=deps.researcher(planned)),
            as_of=day,
            today=day,
            options=options,
            now=clock,
        )
    except Exception as exc:
        raise BacktestError(f"run_cycle failed on {day}: {exc}") from exc
    if report.outcome not in ACCEPTED_OUTCOMES:
        raise BacktestError(f"run_cycle on {day} ended {report.outcome.value}: {report.error}")
    if report.shortlist is not None and _sides(report.shortlist) != _sides(planned):
        raise BacktestError(f"shortlist mismatch on {day}: the cycle built another shortlist")
    return report


def _sides(shortlist: Shortlist) -> dict[str, tuple[str, str]]:
    # ticker, side and sector are all that decisions read (researcher: ticker+side; sizing,
    # sector caps and risk: sector); depth, composite and trimmed are never read downstream.
    return {c.ticker: (c.side.value, c.sector) for c in shortlist.candidates}


def _benchmark(data: FrozenData, window: tuple[date, ...], end: date) -> NavHistory:
    """SPY's vendor-adjusted closes on the window days, as a curve."""
    if BENCHMARK not in data.closes.columns:
        raise BacktestError(f"the store has no {BENCHMARK} closes")
    series = closes_as_of(data.closes[BENCHMARK], BENCHMARK, data.splits, data.dividends, as_of=end)
    points = []
    for day in window:
        value = series.get(pd.Timestamp(day))
        if value is None or pd.isna(value):
            raise BacktestError(f"{BENCHMARK} has no close on {day}")
        points.append(NavPoint(date=day, nav=float(value)))
    return tuple(points)
