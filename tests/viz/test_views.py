"""The page views: values from hand-built objects, and one pass over the demo root."""

from __future__ import annotations

import dataclasses
from datetime import UTC, date, datetime, timedelta

import pytest

from hedge_fund.domain.approval import ApprovalDecision, ApprovalStatus
from hedge_fund.domain.attribution import AttributeOutcome, AttributionReport, AttributionTotals
from hedge_fund.domain.backtest import BacktestReport, SimCosts
from hedge_fund.domain.cost import BudgetPlan, Estimate, EstimateSource, Spend
from hedge_fund.domain.cycle import CycleOutcome
from hedge_fund.domain.orders import Order, OrderAction, OrderPlan
from hedge_fund.domain.research import NameResearch, Rating, ResearchResult, RunRecord, Verdict
from hedge_fund.domain.risk import BreakerState, NavHistory, RiskBreakdown, RiskVerdict, Violation
from hedge_fund.domain.shortlist import Candidate, Shortlist, Side
from hedge_fund.domain.verification import Bucket, HorizonSummary, VerificationReport, VerifyOutcome
from hedge_fund.engine.backtest.metrics import backtest_metrics, performance
from hedge_fund.viz.load import Workspace
from hedge_fund.viz.views import (
    CostRow,
    ExposurePoint,
    Funnel,
    HorizonRow,
    NameRow,
    PositionRow,
    SectorExposure,
    _drawdown,
    backtest_view,
    cycle_view,
    overview_view,
    portfolio_view,
    research_view,
)
from tests.viz.builders import (
    DAY0,
    make_cycle,
    make_nav,
    make_position,
    make_snapshots,
    make_target,
)

STARTED = datetime(2024, 3, 1, 21, 30, tzinfo=UTC)


def test_overview_summarises_nav_return_drawdown_and_the_last_cycle() -> None:
    cycles = (make_cycle(date(2024, 1, 3), outcome=CycleOutcome.BLOCKED),)

    view = overview_view(make_snapshots(100.0, 110.0, 99.0), make_nav(100.0, 101.0, 102.0), cycles)

    assert view.as_of == date(2024, 1, 4)
    assert view.nav == 99.0
    assert view.total_return == pytest.approx(-0.01)
    assert view.benchmark_return == pytest.approx(0.02)
    assert view.max_drawdown == pytest.approx(0.1)
    assert view.last_cycle_as_of == date(2024, 1, 3)
    assert view.last_cycle_outcome is CycleOutcome.BLOCKED


def test_overview_of_nothing_is_all_none() -> None:
    view = overview_view((), None, ())

    assert (view.as_of, view.nav, view.total_return, view.benchmark_return) == (None,) * 4
    assert (view.max_drawdown, view.last_cycle_as_of, view.last_cycle_outcome) == (None,) * 3


def test_overview_of_a_two_point_curve_has_a_nav_but_no_statistics() -> None:
    view = overview_view(make_snapshots(100.0, 101.0), make_nav(100.0, 101.0), ())

    assert view.nav == 101.0
    assert view.total_return is None
    assert view.benchmark_return is None
    assert view.max_drawdown is None


def test_portfolio_indexes_both_curves_to_100_and_measures_drawdown() -> None:
    nav = make_nav(100.0, 120.0, 90.0, 110.0)

    view = portfolio_view(
        make_snapshots(100.0, 120.0, 90.0, 110.0), make_nav(50.0, 51.0, 49.0, 52.0), ()
    )

    assert view.nav_index[0] == (DAY0, 100.0)
    assert [value for _, value in view.nav_index] == pytest.approx([100.0, 120.0, 90.0, 110.0])
    assert view.benchmark_index is not None
    assert [v for _, v in view.benchmark_index] == pytest.approx([100.0, 102.0, 98.0, 104.0])
    assert [v for _, v in view.drawdown] == pytest.approx([0.0, 0.0, -0.25, 110 / 120 - 1])
    assert -min(v for _, v in view.drawdown) == pytest.approx(performance(nav).max_drawdown)


def _days(curve: tuple[tuple[date, float], ...] | None) -> list[date]:
    assert curve is not None
    return [day for day, _ in curve]


def _values(curve: tuple[tuple[date, float], ...] | None) -> list[float]:
    assert curve is not None
    return [value for _, value in curve]


def _day(offset: int) -> date:
    return DAY0 + timedelta(days=offset)


def test_a_benchmark_covering_every_nav_date_is_indexed_from_the_first_nav_date() -> None:
    snapshots = make_snapshots(100.0, 120.0, 90.0, 110.0)
    benchmark = make_nav(40.0, 45.0, 50.0, 51.0, 49.0, 52.0, 60.0, 70.0, start=_day(-2))

    portfolio = portfolio_view(snapshots, benchmark, ())
    backtest = backtest_view((), snapshots, benchmark)
    overview = overview_view(snapshots, benchmark, ())

    for view in (portfolio, backtest):
        assert _days(view.nav_index) == [_day(i) for i in range(4)]
        assert _days(view.benchmark_index) == _days(view.nav_index)
        assert _values(view.nav_index) == pytest.approx([100.0, 120.0, 90.0, 110.0])
        assert _values(view.benchmark_index) == pytest.approx([100.0, 102.0, 98.0, 104.0])
    assert overview.benchmark_return == pytest.approx(0.04)
    assert overview.total_return == pytest.approx(0.1)


def test_a_benchmark_missing_an_interior_nav_date_is_absent() -> None:
    snapshots = make_snapshots(100.0, 120.0, 90.0, 110.0)
    benchmark = make_nav(50.0, 51.0, start=DAY0) + make_nav(52.0, 53.0, start=_day(3))

    portfolio = portfolio_view(snapshots, benchmark, ())
    backtest = backtest_view((), snapshots, benchmark)
    overview = overview_view(snapshots, benchmark, ())

    assert portfolio.benchmark_index is None
    assert backtest.benchmark_index is None
    assert overview.benchmark_return is None
    assert _values(portfolio.nav_index) == pytest.approx([100.0, 120.0, 90.0, 110.0])
    assert overview.total_return == pytest.approx(0.1)


@pytest.mark.parametrize(
    "benchmark",
    [
        make_nav(50.0, 51.0, 49.0, start=_day(10)),
        make_nav(50.0, 51.0, 49.0, 52.0, start=_day(1)),
        make_nav(50.0, 51.0, 49.0, 52.0, start=_day(-1))[:3],
    ],
    ids=["disjoint", "starts-late", "ends-early"],
)
def test_a_benchmark_not_spanning_the_nav_window_is_absent(benchmark: NavHistory) -> None:
    snapshots = make_snapshots(100.0, 120.0, 90.0, 110.0)

    assert portfolio_view(snapshots, benchmark, ()).benchmark_index is None
    assert backtest_view((), snapshots, benchmark).benchmark_index is None
    assert overview_view(snapshots, benchmark, ()).benchmark_return is None


def test_a_benchmark_with_interleaved_dates_is_used_when_it_covers_the_nav_dates() -> None:
    snapshots = (
        make_snapshots(100.0, start=_day(0))
        + make_snapshots(120.0, start=_day(2))
        + make_snapshots(90.0, start=_day(4))
    )
    benchmark = make_nav(50.0, 1.0, 55.0, 2.0, 45.0, 3.0, 4.0)

    portfolio = portfolio_view(snapshots, benchmark, ())
    overview = overview_view(snapshots, benchmark, ())

    assert _days(portfolio.benchmark_index) == [_day(0), _day(2), _day(4)]
    assert _values(portfolio.benchmark_index) == pytest.approx([100.0, 110.0, 90.0])
    assert overview.benchmark_return == pytest.approx(-0.1)


@pytest.mark.parametrize("first", [0.0, -5.0])
def test_a_benchmark_whose_first_close_is_not_positive_is_absent(first: float) -> None:
    snapshots = make_snapshots(100.0, 120.0, 90.0, 110.0)
    benchmark = make_nav(first, 51.0, 49.0, 52.0)

    assert portfolio_view(snapshots, benchmark, ()).benchmark_index is None
    assert backtest_view((), snapshots, benchmark).benchmark_index is None
    assert overview_view(snapshots, benchmark, ()).benchmark_return is None


def test_a_benchmark_without_any_nav_is_absent() -> None:
    benchmark = make_nav(50.0, 51.0, 49.0)

    assert portfolio_view((), benchmark, ()).benchmark_index is None
    assert backtest_view((), (), benchmark).benchmark_index is None


def test_the_nav_figures_do_not_depend_on_the_benchmark() -> None:
    snapshots = make_snapshots(100.0, 120.0, 90.0, 110.0)
    covering = make_nav(50.0, 51.0, 49.0, 52.0)
    partial = make_nav(50.0, 51.0, 49.0)

    reference = (
        portfolio_view(snapshots, None, ()),
        overview_view(snapshots, None, ()),
        backtest_view((), snapshots, None),
    )
    for benchmark in (covering, partial):
        portfolio = portfolio_view(snapshots, benchmark, ())
        overview = overview_view(snapshots, benchmark, ())
        backtest = backtest_view((), snapshots, benchmark)
        assert portfolio.nav_index == reference[0].nav_index
        assert portfolio.drawdown == reference[0].drawdown
        assert overview.total_return == reference[1].total_return
        assert overview.max_drawdown == reference[1].max_drawdown
        assert backtest.nav_index == reference[2].nav_index


def test_portfolio_exposure_follows_each_cycle_target_and_sectors_sum_the_latest_book() -> None:
    first = make_target(
        date(2024, 1, 2),
        make_position("AAA", "Tech", 0.10),
        make_position("BBB", "Tech", -0.05),
        make_position("CCC", "Energy", 0.20),
    )
    second = make_target(
        date(2024, 2, 1), make_position("AAA", "Tech", 0.30), make_position("CCC", "Energy", -0.10)
    )
    cycles = (
        make_cycle(date(2024, 1, 2), target=first),
        make_cycle(date(2024, 2, 1), target=second),
        make_cycle(date(2024, 3, 1)),
    )

    view = portfolio_view((), None, cycles)

    assert [(p.as_of, p.gross, p.net) for p in view.exposure] == [
        (date(2024, 1, 2), pytest.approx(0.35), pytest.approx(0.25)),
        (date(2024, 2, 1), pytest.approx(0.40), pytest.approx(0.20)),
    ]
    assert view.sectors == (SectorExposure("Energy", 0.0, -0.10), SectorExposure("Tech", 0.30, 0.0))
    assert view.positions == (
        PositionRow("AAA", "Tech", Side.LONG, 0.30),
        PositionRow("CCC", "Energy", Side.SHORT, -0.10),
    )
    assert view.target_as_of == date(2024, 2, 1)
    assert isinstance(view.exposure[0], ExposurePoint)


def test_portfolio_of_nothing_is_empty() -> None:
    view = portfolio_view((), None, ())

    assert view.nav_index == view.drawdown == view.exposure == view.sectors == ()
    assert view.positions == ()
    assert view.benchmark_index is None
    assert view.target_as_of is None


def test_cycle_view_rows_carry_the_funnel_vetoes_approval_and_spend() -> None:
    candidates = (
        Candidate("AAA", "Tech", Side.LONG, 1, 0.5),
        Candidate("BBB", "Tech", Side.SHORT, 1, 0.4),
    )
    verdict = RiskVerdict(
        as_of=DAY0,
        approved=False,
        halt=False,
        violations=(Violation("gross", 1.7, 1.6, "gross 1.70 > 1.60"),),
        breakdown=RiskBreakdown(0.0, 0.0, 0.0, 0.0, {}, {}, ()),
        breakers=BreakerState(0.0, 0.0, False, False),
    )
    order = Order("AAA", OrderAction.BUY, 10, 100.0, 1000.0)
    full = make_cycle(
        DAY0,
        outcome=CycleOutcome.BLOCKED,
        shortlist=Shortlist(DAY0, candidates, {}),
        research=ResearchResult(DAY0, ()),
        target=make_target(DAY0, make_position("AAA", "Tech", 0.1)),
        verdict=verdict,
        plan=OrderPlan(DAY0, (order,), {}, {}, None),
        decision=ApprovalDecision(DAY0, ApprovalStatus.BLOCKED, "fp", "summary"),
        budget=BudgetPlan(
            1.0, Estimate(0.1, EstimateSource.FALLBACK, 0), 2, 3, 2, 3, 0.6, False, ()
        ),
        spend=Spend(0.12, 3, 0, (), None),
        price_unresolved={"ZZZ": "no close"},
        fundamentals_unresolved={"YYY": "no filing"},
        error=None,
    )

    view = cycle_view((full, make_cycle(date(2024, 2, 1), dry_run=True)))

    row, bare = view.rows
    assert row.as_of == DAY0
    assert row.outcome is CycleOutcome.BLOCKED
    assert row.funnel == Funnel(shortlist=2, researched=0, portfolio=1, orders=1)
    assert row.vetoes == ("gross: gross 1.70 > 1.60",)
    assert row.halted is False
    assert row.approval is ApprovalStatus.BLOCKED
    assert (row.budget, row.spent, row.unresolved) == (1.0, 0.12, 2)
    assert bare.funnel == Funnel(0, 0, 0, 0)
    assert bare.vetoes == ()
    assert bare.halted is False
    assert (bare.approval, bare.budget, bare.spent, bare.dry_run) == (None, None, None, True)


def _verification(outcome: VerifyOutcome, hit_rate: float) -> VerificationReport:
    bucket = Bucket("all", 5, 3, hit_rate, 0.01, 0.01)
    summary = HorizonSummary(5, 5, 0, 0, bucket, {}, {}, {}, bucket, 0.2, None)
    return VerificationReport(None, None, STARTED, outcome, None, None, (), (summary,))


def _attribution(outcome: AttributeOutcome, market: float) -> AttributionReport:
    totals = AttributionTotals(market, 0.0, 0.0, 0.0, market)
    return AttributionReport(
        None, None, STARTED, outcome, None, None, (), {"AAA": totals}, {"Tech": totals},
        totals, (), {}, (), {},
    )  # fmt: skip


def test_research_view_takes_the_latest_names_report_attribution_and_costs() -> None:
    candidate = Candidate("AAA", "Tech", Side.LONG, 1, 0.5)
    failed_run = RunRecord("AAA", DAY0, 0, None, "", {}, 1.0, "boom")
    good_run = RunRecord("AAA", DAY0, 1, Rating.BUY, "Buy: wide moat", {}, 1.0, None)
    name = NameResearch(candidate, (failed_run, good_run), 1, 2.0, 0.0, 1.0, Verdict.CONFIRM)
    cycles = (
        make_cycle(
            DAY0,
            budget=BudgetPlan(
                1.0, Estimate(0.1, EstimateSource.FALLBACK, 0), 2, 3, 2, 3, 0.6, False, ()
            ),
            spend=Spend(0.12, 3, 0, (), None),
        ),
        make_cycle(date(2024, 2, 1)),
    )

    view = research_view(
        (ResearchResult(DAY0, ()), ResearchResult(date(2024, 2, 1), (name,))),
        cycles,
        (_verification(VerifyOutcome.CLEAN, 0.5), _verification(VerifyOutcome.FAILED, 0.9)),
        (_attribution(AttributeOutcome.CLEAN, 7.0), _attribution(AttributeOutcome.FAILED, 9.0)),
    )

    assert view.names == (
        NameRow(
            date(2024, 2, 1), "AAA", "Tech", Side.LONG, Verdict.CONFIRM, 2.0, 0.0, 1.0, 1,
            "Buy: wide moat",
        ),
    )  # fmt: skip
    assert view.horizons == (HorizonRow(horizon=5, scored=5, hit_rate=0.5, ic=0.2),)
    assert view.attribution is not None
    assert view.attribution.total.market == 7.0
    assert dict(view.attribution.by_name) == {"AAA": AttributionTotals(7.0, 0.0, 0.0, 0.0, 7.0)}
    assert view.costs == (CostRow(DAY0, 1.0, 0.12), CostRow(date(2024, 2, 1), None, None))


def test_research_view_of_a_backtest_root_has_explained_gaps() -> None:
    view = research_view((), (), (), ())

    assert view.names == view.horizons == view.costs == ()
    assert view.attribution is None


def test_a_name_with_no_valid_run_has_an_empty_thesis() -> None:
    candidate = Candidate("AAA", "Tech", Side.LONG, 1, 0.5)
    run = RunRecord("AAA", DAY0, 0, None, "", {}, 1.0, "boom")
    name = NameResearch(candidate, (run,), 0, None, None, None, Verdict.INSUFFICIENT)

    view = research_view((ResearchResult(DAY0, (name,)),), (), (), ())

    assert view.names[0].thesis == ""
    assert view.names[0].mean is None


def _backtest_report() -> BacktestReport:
    curve = make_nav(100.0, 120.0, 96.0, 115.2)
    return BacktestReport(
        run_id="2024-01-02_2024-01-05_0123456789ab",
        store_digest="d" * 64,
        start=curve[0].date,
        end=curve[-1].date,
        cash=100_000.0,
        costs=SimCosts(),
        trading_days=4,
        rebalances=1,
        metrics=backtest_metrics(curve, make_nav(50.0, 51.0, 52.0, 53.0), (), (), flagged=()),
    )


def test_backtest_view_lists_runs_with_excess_return_and_breaker_days() -> None:
    report = _backtest_report()

    view = backtest_view(
        (report,), make_snapshots(100.0, 120.0, 96.0, 115.2), make_nav(50.0, 51.0, 52.0, 53.0)
    )

    [run] = view.runs
    assert run.run_id == report.run_id
    assert run.total_return == pytest.approx(0.152)
    assert run.benchmark_return == pytest.approx(0.06)
    assert run.excess_return == pytest.approx(0.092)
    assert run.max_drawdown == pytest.approx(report.metrics.strategy.max_drawdown)
    assert run.breaker_days == (
        report.metrics.honesty.daily_breaker_days + report.metrics.honesty.drawdown_breaker_days
    )
    assert run.biases == report.biases
    assert view.nav_index[0] == (DAY0, 100.0)
    assert view.benchmark_index is not None


def test_backtest_view_without_reports_is_empty() -> None:
    view = backtest_view((), (), None)

    assert view.runs == ()
    assert view.nav_index == ()
    assert view.benchmark_index is None


def test_a_zero_first_nav_is_rejected_by_the_engine_not_divided_by() -> None:
    with pytest.raises(ValueError, match="nav must be a finite number > 0"):
        portfolio_view(make_snapshots(0.0, 10.0, 5.0), None, ())


def test_drawdown_of_a_non_positive_peak_is_zero_until_the_peak_turns_positive() -> None:
    curve = _drawdown(make_nav(0.0, 0.0, 10.0, 5.0))

    assert [v for _, v in curve] == pytest.approx([0.0, 0.0, 0.0, -0.5])


def test_zero_and_nan_sector_weights_are_skipped() -> None:
    target = make_target(
        DAY0,
        make_position("AAA", "Tech", 0.10),
        make_position("BBB", "Tech", 0.0),
        make_position("CCC", "Energy", float("nan")),
        make_position("DDD", "Health", -0.05),
    )

    view = portfolio_view((), None, (make_cycle(DAY0, target=target),))

    assert view.sectors == (SectorExposure("Health", 0.0, -0.05), SectorExposure("Tech", 0.10, 0.0))


def test_unsorted_cycles_pick_the_latest_target_by_date() -> None:
    older = make_target(date(2024, 1, 2), make_position("AAA", "Tech", 0.10))
    newer = make_target(date(2024, 2, 1), make_position("BBB", "Energy", 0.20))
    cycles = (
        make_cycle(date(2024, 2, 1), target=newer),
        make_cycle(date(2024, 1, 2), target=older),
    )

    view = portfolio_view((), None, cycles)

    assert view.target_as_of == date(2024, 2, 1)
    assert [p.ticker for p in view.positions] == ["BBB"]
    assert [p.as_of for p in view.exposure] == [date(2024, 1, 2), date(2024, 2, 1)]


def test_unsorted_cycles_pick_the_latest_cycle_for_the_overview() -> None:
    cycles = (
        make_cycle(date(2024, 2, 1), outcome=CycleOutcome.BLOCKED),
        make_cycle(date(2024, 1, 2), outcome=CycleOutcome.SUBMITTED),
    )

    view = overview_view((), None, cycles)

    assert view.last_cycle_as_of == date(2024, 2, 1)
    assert view.last_cycle_outcome is CycleOutcome.BLOCKED


def test_unsorted_research_picks_the_latest_names_report_and_attribution_by_date() -> None:
    candidate = Candidate("AAA", "Tech", Side.LONG, 1, 0.5)
    name = NameResearch(candidate, (), 0, None, None, None, Verdict.INSUFFICIENT)
    later = STARTED + timedelta(days=1)

    view = research_view(
        (ResearchResult(date(2024, 2, 1), (name,)), ResearchResult(DAY0, ())),
        (),
        (
            dataclasses.replace(_verification(VerifyOutcome.CLEAN, 0.9), started_at=later),
            _verification(VerifyOutcome.CLEAN, 0.5),
        ),
        (
            dataclasses.replace(_attribution(AttributeOutcome.CLEAN, 9.0), started_at=later),
            _attribution(AttributeOutcome.CLEAN, 7.0),
        ),
    )

    assert [row.as_of for row in view.names] == [date(2024, 2, 1)]
    assert view.horizons[0].hit_rate == 0.9
    assert view.attribution is not None
    assert view.attribution.total.market == 9.0


def test_the_funnel_counts_researched_names_and_the_row_carries_halt_and_error() -> None:
    candidate = Candidate("AAA", "Tech", Side.LONG, 1, 0.5)
    name = NameResearch(candidate, (), 0, None, None, None, Verdict.INSUFFICIENT)
    verdict = RiskVerdict(
        as_of=DAY0,
        approved=False,
        halt=True,
        violations=(),
        breakdown=RiskBreakdown(0.0, 0.0, 0.0, 0.0, {}, {}, ()),
        breakers=BreakerState(0.0, 0.0, True, False),
    )
    report = make_cycle(
        DAY0,
        outcome=CycleOutcome.ERROR,
        research=ResearchResult(DAY0, (name, name)),
        verdict=verdict,
        error="RuntimeError: boom",
    )

    [row] = cycle_view((report,)).rows

    assert row.funnel.researched == 2
    assert row.halted is True
    assert row.error == "RuntimeError: boom"


def test_every_view_builds_from_the_demo_workspace(workspace: Workspace) -> None:
    overview = overview_view(workspace.snapshots, workspace.benchmark, workspace.cycles)
    portfolio = portfolio_view(workspace.snapshots, workspace.benchmark, workspace.cycles)
    cycles = cycle_view(workspace.cycles)
    research = research_view(
        workspace.research, workspace.cycles, workspace.verifications, workspace.attributions
    )
    backtest = backtest_view(workspace.backtests, workspace.snapshots, workspace.benchmark)

    assert overview.nav is not None
    assert overview.total_return == pytest.approx(
        workspace.backtests[0].metrics.strategy.total_return
    )
    assert len(portfolio.exposure) == 2
    assert [row.as_of for row in cycles.rows] == [c.as_of for c in workspace.cycles]
    assert research.names
    assert research.horizons == ()
    assert research.attribution is None
    assert backtest.runs[0].run_id == workspace.backtests[0].run_id
    assert backtest.benchmark_index is not None
