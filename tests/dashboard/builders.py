"""Hand-built roots, states and AppTest helpers for the dashboard tests."""

from __future__ import annotations

import dataclasses
import json
import shutil
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

from streamlit.testing.v1 import AppTest

from hedge_fund.adapters.account import JsonSnapshotStore
from hedge_fund.adapters.backtest.report import BACKTEST_DIR
from hedge_fund.adapters.cycle import JsonCycleReportStore
from hedge_fund.adapters.demo_marker import DEMO_MARKER
from hedge_fund.dashboard.data import DEFAULT_ROOT, RootState
from hedge_fund.domain.approval import ApprovalDecision, ApprovalStatus
from hedge_fund.domain.attribution import AttributeOutcome, AttributionReport, AttributionTotals
from hedge_fund.domain.backtest import BacktestReport, SimCosts
from hedge_fund.domain.cost import BudgetPlan, Estimate, EstimateSource, Spend
from hedge_fund.domain.cycle import CycleOutcome, CycleReport
from hedge_fund.domain.orders import Order, OrderAction, OrderPlan
from hedge_fund.domain.research import NameResearch, Rating, ResearchResult, RunRecord, Verdict
from hedge_fund.domain.risk import BreakerState, RiskBreakdown, RiskVerdict, Violation
from hedge_fund.domain.shortlist import Candidate, Shortlist, Side
from hedge_fund.domain.verification import Bucket, HorizonSummary, VerificationReport, VerifyOutcome
from hedge_fund.engine.backtest.metrics import backtest_metrics
from hedge_fund.viz.load import Workspace
from hedge_fund.viz.views import BacktestRun
from tests.viz.builders import (
    DAY0,
    make_cycle,
    make_nav,
    make_position,
    make_snapshots,
    make_target,
)

SYNTHETIC = Path("synthetic")
NAV_VALUES = (100.0, 120.0, 90.0, 115.0)  # 2024-01-02 .. 2024-01-05


def make_workspace(**changes: Any) -> Workspace:
    base = Workspace(
        root=SYNTHETIC,
        demo=None,
        snapshots=(),
        cycles=(),
        research=(),
        settlements=(),
        attributions=(),
        verifications=(),
        backtests=(),
        benchmark=None,
        unreadable=(),
    )
    return dataclasses.replace(base, **changes)


def make_state(**changes: Any) -> RootState:
    """An existing, unclassified root holding what ``changes`` give the workspace."""
    return RootState(
        root=SYNTHETIC, workspace=make_workspace(**changes), kind=None, kind_error=None
    )


def missing_state() -> RootState:
    return RootState(root=Path(DEFAULT_ROOT), workspace=None, kind=None, kind_error=None)


def benchmark_state(*closes: float, cycles: tuple[CycleReport, ...] = ()) -> RootState:
    """NAV 100, 120, 90, 115 from 2024-01-02, and SPY ``closes`` from the second NAV date
    (no benchmark when there are none).
    """
    benchmark = make_nav(*closes, start=DAY0 + timedelta(days=1)) if closes else None
    return make_state(snapshots=make_snapshots(*NAV_VALUES), benchmark=benchmark, cycles=cycles)


def targeted_state(*closes: float) -> RootState:
    """``benchmark_state`` plus two cycles with a target each (AAA long, BBB short)."""
    first = make_target(
        DAY0, make_position("AAA", "Tech", 0.30), make_position("BBB", "Energy", -0.10)
    )
    later = DAY0 + timedelta(days=2)
    second = make_target(
        later, make_position("AAA", "Tech", 0.20), make_position("BBB", "Energy", -0.10)
    )
    cycles = (make_cycle(DAY0, target=first), make_cycle(later, target=second))
    return benchmark_state(*closes, cycles=cycles)


def flat_state() -> RootState:
    """A cycle whose target holds no position: exposure points, no sectors, no positions."""
    return benchmark_state(cycles=(make_cycle(DAY0, target=make_target(DAY0)),))


def make_run(**changes: Any) -> BacktestRun:
    base = BacktestRun(
        run_id="run-1",
        start=date(2024, 1, 10),
        end=date(2024, 3, 8),
        trading_days=40,
        rebalances=3,
        total_return=0.10,
        benchmark_return=0.05,
        excess_return=0.05,
        max_drawdown=0.20,
        sharpe=1.5,
        beta=0.25,
        breaker_days=2,
        biases=(),
    )
    return dataclasses.replace(base, **changes)


def copy_root(demo_root: Path, tmp_path: Path, name: str) -> Path:
    root = tmp_path / name
    shutil.copytree(demo_root, root)
    return root


def backtest_root(demo_root: Path, tmp_path: Path) -> Path:
    """The demo run without its marker: a plain backtest root."""
    root = copy_root(demo_root, tmp_path, "backtest")
    (root / DEMO_MARKER).unlink()
    return root


def mismatched_root(demo_root: Path, tmp_path: Path) -> Path:
    """A root whose valid-format marker names another run: ``Workspace.demo`` is set, yet
    ``classify_root`` refuses it.
    """
    root = copy_root(demo_root, tmp_path, "mismatched")
    marker = json.loads((root / DEMO_MARKER).read_text(encoding="utf-8"))
    marker["run_id"] = "other-run"
    (root / DEMO_MARKER).write_text(json.dumps(marker), encoding="utf-8")
    return root


def open_page(script: Callable[..., None], root: Path | None = None, *args: Any) -> AppTest:
    """Run ``script`` (a function from ``tests/dashboard/scripts.py``) once, then point the
    sidebar root picker at ``root`` and run again.
    """
    at = AppTest.from_function(script, default_timeout=60, args=args or None).run()
    if root is not None:
        at.sidebar.text_input(key="root").set_value(str(root)).run()
    return at


LATER = DAY0 + timedelta(days=2)  # the second cycle of ``rich_state``
STARTED = datetime(2024, 3, 1, 21, 30, tzinfo=UTC)
RUN_ID = "2024-01-02_2024-01-05_0123456789ab"
LIVE_REFUSAL = "synthetic: expected exactly one readable backtest summary, found 0 and 0 unreadable"


def with_workspace(state: RootState, **changes: Any) -> RootState:
    assert state.workspace is not None
    return dataclasses.replace(state, workspace=dataclasses.replace(state.workspace, **changes))


def make_budget(budget: float = 1.0) -> BudgetPlan:
    return BudgetPlan(budget, Estimate(0.1, EstimateSource.FALLBACK, 0), 2, 3, 2, 3, 0.6, False, ())


def make_spend(total: float = 0.12) -> Spend:
    return Spend(total, 3, 0, (), None)


def make_name(
    ticker: str = "AAA", *, valid: bool = True, mean: float = 2.0, thesis: str = "Buy: wide moat"
) -> NameResearch:
    """A long Tech name: one valid BUY run with ``thesis`` (mean ``mean``, std 0, agreement 1),
    or with ``valid=False`` one failed run and no statistics.
    """
    candidate = Candidate(ticker, "Tech", Side.LONG, 1, 0.5)
    if not valid:
        failed = RunRecord(ticker, DAY0, 0, None, "", {}, 1.0, "boom")
        return NameResearch(candidate, (failed,), 0, None, None, None, Verdict.INSUFFICIENT)
    run = RunRecord(ticker, DAY0, 0, Rating.BUY, thesis, {}, 1.0, None)
    return NameResearch(candidate, (run,), 1, mean, 0.0, 1.0, Verdict.CONFIRM)


def _verdict(*, halt: bool, violations: tuple[Violation, ...]) -> RiskVerdict:
    return RiskVerdict(
        as_of=DAY0,
        approved=False,
        halt=halt,
        violations=violations,
        breakdown=RiskBreakdown(0.0, 0.0, 0.0, 0.0, {}, {}, ()),
        breakers=BreakerState(0.0, 0.0, halt, False),
    )


def full_cycle(as_of: date = DAY0) -> CycleReport:
    """A blocked cycle that reached every stage: 2 shortlisted, 1 researched, 1 target
    position, 1 order, the veto "gross: gross 1.70 > 1.60", approval blocked, LLM budget
    $1.00 and spend $0.12.
    """
    candidates = (
        Candidate("AAA", "Tech", Side.LONG, 1, 0.5),
        Candidate("BBB", "Tech", Side.SHORT, 1, 0.4),
    )
    return make_cycle(
        as_of,
        outcome=CycleOutcome.BLOCKED,
        shortlist=Shortlist(as_of, candidates, {}),
        research=ResearchResult(as_of, (make_name(),)),
        target=make_target(as_of, make_position("AAA", "Tech", 0.1)),
        verdict=_verdict(
            halt=False, violations=(Violation("gross", 1.7, 1.6, "gross 1.70 > 1.60"),)
        ),
        plan=OrderPlan(as_of, (Order("AAA", OrderAction.BUY, 10, 100.0, 1000.0),), {}, {}, None),
        decision=ApprovalDecision(as_of, ApprovalStatus.BLOCKED, "fp", "summary"),
        budget=make_budget(),
        spend=make_spend(),
    )


def make_verification(hit_rate: float | None = 0.5, ic: float | None = 0.2) -> VerificationReport:
    bucket = Bucket("all", 5, 3, hit_rate, 0.01, 0.01)
    summary = HorizonSummary(5, 5, 0, 0, bucket, {}, {}, {}, bucket, ic, None)
    return VerificationReport(None, None, STARTED, VerifyOutcome.CLEAN, None, None, (), (summary,))


def make_attribution(market: float = 7.0) -> AttributionReport:
    """AAA / Tech: market ``market``, sector 1.00, residual -0.50, trading -0.25."""
    totals = AttributionTotals(market, 1.0, -0.5, -0.25, market + 0.25)
    return AttributionReport(
        None, None, STARTED, AttributeOutcome.CLEAN, None, None, (), {"AAA": totals},
        {"Tech": totals}, totals, (), {}, (), {},
    )  # fmt: skip


def make_backtest_report(**changes: Any) -> BacktestReport:
    curve = make_nav(*NAV_VALUES)
    base = BacktestReport(
        run_id=RUN_ID,
        store_digest="d" * 64,
        start=curve[0].date,
        end=curve[-1].date,
        cash=100_000.0,
        costs=SimCosts(),
        trading_days=len(curve),
        rebalances=1,
        metrics=backtest_metrics(curve, make_nav(50.0, 51.0, 52.0, 53.0), (), (), flagged=()),
    )
    return dataclasses.replace(base, **changes)


def rich_state(*closes: float) -> RootState:
    """Every D2 input: ``benchmark_state(*closes)`` plus ``full_cycle`` on DAY0 and a bare dry
    run on LATER, research on DAY0 (AAA valid, BBB invalid), one verification, one
    attribution and one backtest summary.
    """
    cycles = (full_cycle(DAY0), make_cycle(LATER, dry_run=True, outcome=CycleOutcome.DRY_RUN))
    return with_workspace(
        benchmark_state(*closes, cycles=cycles),
        research=(ResearchResult(DAY0, (make_name("AAA"), make_name("BBB", valid=False))),),
        verifications=(make_verification(),),
        attributions=(make_attribution(),),
        backtests=(make_backtest_report(),),
    )


def trouble_state() -> RootState:
    """One ERROR cycle: halted by a breaker, an error text, two unresolved tickers."""
    cycle = make_cycle(
        DAY0,
        outcome=CycleOutcome.ERROR,
        verdict=_verdict(halt=True, violations=()),
        error="RuntimeError: boom",
        price_unresolved={"ZZZ": "no close"},
        fundamentals_unresolved={"YYY": "no filing"},
    )
    return make_state(cycles=(cycle,))


def runs_only_state() -> RootState:
    """A backtest summary but no account snapshots."""
    return make_state(backtests=(make_backtest_report(),))


def two_runs_state() -> RootState:
    """``rich_state`` with a second summary, ``run-b``, that lists no residual bias."""
    return with_workspace(
        rich_state(),
        backtests=(make_backtest_report(), make_backtest_report(run_id="run-b", biases=())),
    )


def markdown_thesis_state() -> RootState:
    """Research whose thesis holds markdown and LaTeX characters: ``$5 *bold*``."""
    return make_state(research=(ResearchResult(DAY0, (make_name(thesis="$5 *bold*"),)),))


def live_root(tmp_path: Path, name: str = "live") -> Path:
    """A real paper root on disk: four snapshots and ``full_cycle``, no backtest summary, so
    ``classify_root`` refuses it and ``inspect_root`` marks it live.
    """
    root = tmp_path / name
    snapshots = JsonSnapshotStore(root)
    for snapshot in make_snapshots(*NAV_VALUES):
        snapshots.save(snapshot)
    JsonCycleReportStore(root).save(full_cycle(DAY0))
    return root


def busy_root(tmp_path: Path) -> Path:
    """``live_root`` under "busy" plus a dry run on LATER: two cycles (``live_root`` has one)."""
    root = live_root(tmp_path, "busy")
    JsonCycleReportStore(root).save(make_cycle(LATER, dry_run=True, outcome=CycleOutcome.DRY_RUN))
    return root


def two_run_root(demo_root: Path, tmp_path: Path) -> Path:
    """A copy of the demo root with a second summary, ``run-b`` (the demo has one). It holds
    two summaries, so ``classify_root`` refuses it; the loader reads both.
    """
    root = copy_root(demo_root, tmp_path, "two-runs")
    [summary] = (root / BACKTEST_DIR).glob("*.json")
    body = json.loads(summary.read_text(encoding="utf-8"))
    body["run_id"] = "run-b"
    (root / BACKTEST_DIR / "run-b.json").write_text(json.dumps(body), encoding="utf-8")
    return root
