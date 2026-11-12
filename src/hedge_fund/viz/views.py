"""One frozen view per dashboard page, built from domain objects. Pure.

Reuses the engine (``nav_history``, ``performance``) and the figures the services already
stored; the only thing computed here is presentation (indexing a curve to 100, a running
drawdown, sector sums). Views take domain pieces, not the loader's ``Workspace``, so this
module never imports ``viz.load`` or an adapter.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any

from hedge_fund.domain.account import AccountSnapshot
from hedge_fund.domain.approval import ApprovalStatus
from hedge_fund.domain.attribution import AttributeOutcome, AttributionReport, AttributionTotals
from hedge_fund.domain.backtest import BacktestReport, Performance
from hedge_fund.domain.cycle import CycleOutcome, CycleReport
from hedge_fund.domain.portfolio import TargetPortfolio
from hedge_fund.domain.research import ResearchResult, Verdict
from hedge_fund.domain.risk import NavHistory, NavPoint
from hedge_fund.domain.shortlist import Side
from hedge_fund.domain.verification import VerificationReport, VerifyOutcome
from hedge_fund.engine.account.history import nav_history
from hedge_fund.engine.backtest.metrics import performance

Curve = tuple[tuple[date, float], ...]
MIN_CURVE_POINTS = 3  # what ``performance`` needs


@dataclass(frozen=True)
class OverviewView:
    """Headline numbers; every figure is None when there is no data (or too few points)."""

    as_of: date | None
    nav: float | None
    total_return: float | None
    benchmark_return: float | None
    max_drawdown: float | None
    last_cycle_as_of: date | None
    last_cycle_outcome: CycleOutcome | None


@dataclass(frozen=True)
class ExposurePoint:
    """The gross and net exposure a cycle's target aimed for."""

    as_of: date
    gross: float
    net: float


@dataclass(frozen=True)
class SectorExposure:
    """Signed weight by sector: ``long`` >= 0 and ``short`` <= 0."""

    sector: str
    long: float
    short: float


@dataclass(frozen=True)
class PositionRow:
    ticker: str
    sector: str
    side: Side
    weight: float


@dataclass(frozen=True)
class PortfolioView:
    """NAV and SPY indexed to 100, drawdown (<= 0), exposure per cycle, the latest book."""

    nav_index: Curve
    benchmark_index: Curve | None
    drawdown: Curve
    exposure: tuple[ExposurePoint, ...]
    sectors: tuple[SectorExposure, ...]
    positions: tuple[PositionRow, ...]
    target_as_of: date | None


@dataclass(frozen=True)
class Funnel:
    """What survived each stage of one cycle (the report stores no universe or gate counts)."""

    shortlist: int
    researched: int
    portfolio: int
    orders: int


@dataclass(frozen=True)
class CycleRow:
    as_of: date
    outcome: CycleOutcome
    dry_run: bool
    funnel: Funnel
    vetoes: tuple[str, ...]
    halted: bool
    approval: ApprovalStatus | None
    budget: float | None
    spent: float | None
    unresolved: int
    error: str | None


@dataclass(frozen=True)
class CycleView:
    rows: tuple[CycleRow, ...]


@dataclass(frozen=True)
class NameRow:
    """One researched name; ``dispersion`` is the std of its ratings, ``thesis`` the decision
    text of its first valid run ("" when none).
    """

    as_of: date
    ticker: str
    sector: str
    side: Side
    verdict: Verdict
    mean: float | None
    dispersion: float | None
    agreement: float | None
    valid: int
    thesis: str


@dataclass(frozen=True)
class HorizonRow:
    horizon: int
    scored: int
    hit_rate: float | None
    ic: float | None


@dataclass(frozen=True)
class AttributionView:
    by_name: Mapping[str, AttributionTotals]
    by_sector: Mapping[str, AttributionTotals]
    total: AttributionTotals


@dataclass(frozen=True)
class CostRow:
    as_of: date
    budget: float | None
    spent: float | None


@dataclass(frozen=True)
class ResearchView:
    """Latest research names, latest completed verification and attribution, LLM cost per
    cycle. Empty / None where the root holds none (a backtest root has no fills).
    """

    names: tuple[NameRow, ...]
    horizons: tuple[HorizonRow, ...]
    attribution: AttributionView | None
    costs: tuple[CostRow, ...]


@dataclass(frozen=True)
class BacktestRun:
    run_id: str
    start: date
    end: date
    trading_days: int
    rebalances: int
    total_return: float
    benchmark_return: float
    excess_return: float
    max_drawdown: float
    sharpe: float | None
    beta: float | None
    breaker_days: int
    biases: tuple[str, ...]


@dataclass(frozen=True)
class BacktestView:
    runs: tuple[BacktestRun, ...]
    nav_index: Curve
    benchmark_index: Curve | None


def overview_view(
    snapshots: Sequence[AccountSnapshot],
    benchmark: NavHistory | None,
    cycles: Sequence[CycleReport],
) -> OverviewView:
    nav = _nav(snapshots)
    ours = _performance(nav)
    aligned = _aligned(nav, benchmark)
    theirs = None if aligned is None else _performance(aligned)
    last = _latest(cycles, key=lambda report: report.as_of)
    return OverviewView(
        as_of=nav[-1].date if nav else None,
        nav=nav[-1].nav if nav else None,
        total_return=None if ours is None else ours.total_return,
        benchmark_return=None if theirs is None else theirs.total_return,
        max_drawdown=None if ours is None else ours.max_drawdown,
        last_cycle_as_of=None if last is None else last.as_of,
        last_cycle_outcome=None if last is None else last.outcome,
    )


def portfolio_view(
    snapshots: Sequence[AccountSnapshot],
    benchmark: NavHistory | None,
    cycles: Sequence[CycleReport],
) -> PortfolioView:
    nav = _nav(snapshots)
    targets = sorted(
        (report.target for report in cycles if report.target is not None),
        key=lambda target: target.as_of,
    )
    latest = targets[-1] if targets else None
    aligned = _aligned(nav, benchmark)
    return PortfolioView(
        nav_index=_index(nav),
        benchmark_index=None if aligned is None else _index(aligned),
        drawdown=_drawdown(nav),
        exposure=tuple(ExposurePoint(t.as_of, t.gross, t.net) for t in targets),
        sectors=() if latest is None else _sectors(latest),
        positions=() if latest is None else _positions(latest),
        target_as_of=None if latest is None else latest.as_of,
    )


def cycle_view(cycles: Sequence[CycleReport]) -> CycleView:
    return CycleView(rows=tuple(_cycle_row(report) for report in cycles))


def research_view(
    research: Sequence[ResearchResult],
    cycles: Sequence[CycleReport],
    verifications: Sequence[VerificationReport],
    attributions: Sequence[AttributionReport],
) -> ResearchView:
    verified = _latest(
        [r for r in verifications if r.outcome is not VerifyOutcome.FAILED],
        key=lambda r: r.started_at,
    )
    attributed = _latest(
        [r for r in attributions if r.outcome is not AttributeOutcome.FAILED],
        key=lambda r: r.started_at,
    )
    latest = _latest(research, key=lambda result: result.as_of)
    return ResearchView(
        names=() if latest is None else _names(latest),
        horizons=()
        if verified is None
        else tuple(
            HorizonRow(s.horizon, s.scored, s.overall.hit_rate, s.ic) for s in verified.summaries
        ),
        attribution=None
        if attributed is None
        else AttributionView(attributed.by_ticker, attributed.by_sector, attributed.total),
        costs=tuple(
            CostRow(
                report.as_of,
                None if report.budget is None else report.budget.budget,
                None if report.spend is None else report.spend.total,
            )
            for report in cycles
        ),
    )


def backtest_view(
    reports: Sequence[BacktestReport],
    snapshots: Sequence[AccountSnapshot],
    benchmark: NavHistory | None,
) -> BacktestView:
    nav = _nav(snapshots)
    aligned = _aligned(nav, benchmark)
    return BacktestView(
        runs=tuple(_run(report) for report in reports),
        nav_index=_index(nav),
        benchmark_index=None if aligned is None else _index(aligned),
    )


def _latest[T](items: Sequence[T], key: Callable[[T], Any]) -> T | None:
    """The item with the greatest key; the last one in input order when keys tie."""
    return max(reversed(items), key=key, default=None)


def _nav(snapshots: Sequence[AccountSnapshot]) -> NavHistory:
    if not snapshots:
        return ()
    return nav_history(snapshots, as_of=max(snapshot.as_of for snapshot in snapshots))


def _performance(nav: NavHistory) -> Performance | None:
    return performance(nav) if len(nav) >= MIN_CURVE_POINTS else None


def _aligned(nav: NavHistory, benchmark: NavHistory | None) -> NavHistory | None:
    """The benchmark on exactly the NAV dates, in NAV order; None unless it covers every one
    of them and starts above zero (it is indexed from its first value).
    """
    if benchmark is None or not nav:
        return None
    closes = {point.date: point.nav for point in benchmark}
    if any(point.date not in closes for point in nav) or closes[nav[0].date] <= 0:
        return None
    return tuple(NavPoint(point.date, closes[point.date]) for point in nav)


def _index(nav: NavHistory) -> Curve:
    return tuple((point.date, 100.0 * point.nav / nav[0].nav) for point in nav)


def _drawdown(nav: NavHistory) -> Curve:
    curve = []
    peak = 0.0
    for point in nav:
        peak = max(peak, point.nav)
        curve.append((point.date, point.nav / peak - 1.0))
    return tuple(curve)


def _sectors(target: TargetPortfolio) -> tuple[SectorExposure, ...]:
    longs: dict[str, float] = {}
    shorts: dict[str, float] = {}
    for position in target.positions:
        if position.weight == 0 or math.isnan(position.weight):
            continue
        book = longs if position.weight > 0 else shorts
        book[position.sector] = book.get(position.sector, 0.0) + position.weight
    return tuple(
        SectorExposure(sector, longs.get(sector, 0.0), shorts.get(sector, 0.0))
        for sector in sorted(longs.keys() | shorts.keys())
    )


def _positions(target: TargetPortfolio) -> tuple[PositionRow, ...]:
    return tuple(PositionRow(p.ticker, p.sector, p.side, p.weight) for p in target.positions)


def _funnel(report: CycleReport) -> Funnel:
    return Funnel(
        shortlist=len(report.shortlist.candidates) if report.shortlist else 0,
        researched=len(report.research.names) if report.research else 0,
        portfolio=len(report.target.positions) if report.target else 0,
        orders=len(report.plan.orders) if report.plan else 0,
    )


def _cycle_row(report: CycleReport) -> CycleRow:
    verdict = report.verdict
    return CycleRow(
        as_of=report.as_of,
        outcome=report.outcome,
        dry_run=report.dry_run,
        funnel=_funnel(report),
        vetoes=()
        if verdict is None
        else tuple(f"{v.check}: {v.detail}" for v in verdict.violations),
        halted=False if verdict is None else verdict.halt,
        approval=None if report.decision is None else report.decision.status,
        budget=None if report.budget is None else report.budget.budget,
        spent=None if report.spend is None else report.spend.total,
        unresolved=len(report.price_unresolved) + len(report.fundamentals_unresolved),
        error=report.error,
    )


def _names(result: ResearchResult) -> tuple[NameRow, ...]:
    return tuple(
        NameRow(
            as_of=result.as_of,
            ticker=name.candidate.ticker,
            sector=name.candidate.sector,
            side=name.candidate.side,
            verdict=name.verdict,
            mean=name.mean,
            dispersion=name.std,
            agreement=name.agreement,
            valid=name.valid,
            thesis=next((run.decision for run in name.runs if run.valid), ""),
        )
        for name in result.names
    )


def _run(report: BacktestReport) -> BacktestRun:
    metrics = report.metrics
    return BacktestRun(
        run_id=report.run_id,
        start=report.start,
        end=report.end,
        trading_days=report.trading_days,
        rebalances=report.rebalances,
        total_return=metrics.strategy.total_return,
        benchmark_return=metrics.benchmark.total_return,
        excess_return=metrics.strategy.total_return - metrics.benchmark.total_return,
        max_drawdown=metrics.strategy.max_drawdown,
        sharpe=metrics.strategy.sharpe,
        beta=metrics.beta,
        breaker_days=metrics.honesty.daily_breaker_days + metrics.honesty.drawdown_breaker_days,
        biases=report.biases,
    )
