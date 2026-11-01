"""Tests for the cycle's 7c budget stage, research reuse under a budget, and spend."""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable, Sequence
from datetime import date, timedelta
from typing import Any

import pytest

from hedge_fund.domain.cost import (
    BudgetError,
    Estimate,
    EstimateSource,
    ModelPrice,
    ModelUsage,
    PriceTable,
    ResearchRunError,
    Spend,
)
from hedge_fund.domain.cycle import CycleOptions, CycleOutcome, CycleReport
from hedge_fund.domain.orders import Holding
from hedge_fund.domain.research import NameResearch, Rating, ResearchResult, RunRecord, Verdict
from hedge_fund.domain.shortlist import Candidate, Side
from hedge_fund.services.cycle import run_cycle
from tests.cycle_world import AS_OF, EVENING, HELD_OUTSIDE, TODAY, Clock, World
from tests.fakes import InMemoryResearchStore, ScriptedBroker

FALLBACK = PriceTable(fallback_run_cost=1.0)  # no prices: every run cost unknown, 1.00 assumed
PRICED = PriceTable(  # 1 USD per million input tokens
    models={"m": ModelPrice(input=1.0, output=0.0, cache_write=0.0, cache_read=0.0)}
)
ONE_DOLLAR = (ModelUsage("m", 1, 1_000_000, 0, 0, 0),)  # one run's usage: 1.00 under PRICED


def _run(
    world: World, budget: float | None = None, *, dry_run: bool = True, **options: Any
) -> CycleReport:
    return run_cycle(
        world.deps,
        as_of=AS_OF,
        today=TODAY,
        options=CycleOptions(dry_run=dry_run, budget=budget, **options),
        now=Clock(EVENING),
    )


def _history(day: date, *costs: float) -> ResearchResult:
    """Stored research for ``day``: one name whose runs cost ``costs`` USD under PRICED."""
    candidate = Candidate(ticker="OLD", sector="Tech", side=Side.LONG, depth=1, composite=1.0)
    runs = tuple(
        RunRecord(
            "OLD",
            day,
            index,
            Rating.BUY,
            "",
            {},
            1.0,
            None,
            (ModelUsage("m", 1, round(cost * 1_000_000), 0, 0, 0),),
        )
        for index, cost in enumerate(costs)
    )
    name = NameResearch(candidate, runs, len(runs), 2.0, 0.0, 1.0, Verdict.CONFIRM)
    return ResearchResult(as_of=day, names=(name,))


def _tickers(report: CycleReport) -> list[str]:
    assert report.research is not None
    return [name.candidate.ticker for name in report.research.names]


def _shortlisted(report: CycleReport) -> list[str]:
    assert report.shortlist is not None
    return [candidate.ticker for candidate in report.shortlist.candidates]


class _BadDayStore(InMemoryResearchStore):
    """``load`` raises ValueError (a malformed stored file) for ``bad``."""

    def __init__(self, results: Sequence[ResearchResult], bad: date) -> None:
        super().__init__(results)
        self.bad = bad

    def load(self, as_of: date) -> ResearchResult | None:
        if as_of == self.bad:
            self.loaded.append(as_of)
            raise ValueError(f"{as_of}: malformed research file")
        return super().load(as_of)


class _FailingSaveStore(InMemoryResearchStore):
    """``save`` raises OSError: research finished but could not be cached."""

    def save(self, result: ResearchResult) -> None:
        raise OSError("disk full")


# --- the plan ----------------------------------------------------------------------------


def test_a_budget_cuts_k_first() -> None:
    world = World(price_table=FALLBACK)
    report = _run(world, budget=12.0)
    assert report.outcome is CycleOutcome.DRY_RUN
    budget = report.budget
    assert budget is not None
    assert budget.estimate == Estimate(cost=1.0, source=EstimateSource.FALLBACK, n_samples=0)
    assert (budget.requested_n, budget.requested_k) == (6, 3)
    assert (budget.n, budget.k, budget.skipped) == (6, 2, False)
    assert budget.reasons == ("k 3→2: 6×3×$1.00=$18.00 > $12.00",)
    assert len(world.researcher.calls) == 12
    assert report.research is not None
    assert all(len(name.runs) == 2 for name in report.research.names)
    assert world.research_store.saved == [report.research]


def test_then_n_from_the_bottom_of_the_shortlist() -> None:
    world = World(price_table=FALLBACK)
    report = _run(world, budget=8.0)
    assert report.budget is not None
    assert (report.budget.n, report.budget.k) == (4, 2)
    assert report.budget.reasons == (
        "k 3→2: 6×3×$1.00=$18.00 > $8.00",
        "n 6→4: 6×2×$1.00=$12.00 > $8.00",
    )
    assert len(_shortlisted(report)) == 6  # the report keeps the whole shortlist
    assert _tickers(report) == _shortlisted(report)[:4]
    assert len(world.researcher.calls) == 8


@pytest.mark.parametrize("dry_run", [pytest.param(True, id="dry"), pytest.param(False, id="live")])
def test_a_budget_too_small_for_one_name_stops_the_cycle_before_the_portfolio(
    dry_run: bool,
) -> None:
    # Held names: an empty target would close them all, so a skip must never reach orders.
    held = (Holding(ticker="T0", shares=100), Holding(ticker=HELD_OUTSIDE, shares=-50))
    world = World(price_table=FALLBACK, broker=ScriptedBroker(holdings=held))
    report = _run(world, budget=1.0, dry_run=dry_run)
    assert report.outcome is CycleOutcome.NOTHING_TO_DO
    assert (report.failed_stage, report.error) == (None, None)
    assert report.budget is not None
    assert (report.budget.skipped, report.budget.n) == (True, 0)
    assert report.budget.reasons == (
        "k 3→2: 6×3×$1.00=$18.00 > $1.00",
        "n 6→0: 6×2×$1.00=$12.00 > $1.00",
    )
    assert report.spend == Spend(total=0.0, runs=0, unknown_runs=0, usage=(), over_budget=False)
    assert report.shortlist is not None
    assert report.research is None
    assert report.target is None
    assert report.verdict is None
    assert report.order_prices is None
    assert report.plan is None
    assert report.decision is None
    assert report.execution is None
    assert world.researcher.calls == []
    assert world.research_store.saved == []  # a later unbudgeted cycle must still research
    assert world.gate.summaries == []
    assert world.broker.submitted == []
    assert world.report_store.saved == [report]


# --- the estimate from stored research ---------------------------------------------------


def test_the_estimate_is_the_median_of_stored_runs_newest_first_skipping_today() -> None:
    store = InMemoryResearchStore(
        [
            _history(date(2026, 9, 28), 0.5, 0.5, 0.5),
            _history(date(2026, 9, 29), 1.0, 2.0),
            _history(AS_OF, 100.0),  # the cycle's own date is not history
        ]
    )
    report = _run(World(price_table=PRICED, research_store=store), budget=100.0)
    assert report.budget is not None
    assert report.budget.estimate == Estimate(cost=0.5, source=EstimateSource.HISTORY, n_samples=5)
    assert (report.budget.n, report.budget.k, report.budget.reasons) == (6, 3, ())
    assert store.loaded == [AS_OF, date(2026, 9, 29), date(2026, 9, 28)]


def test_the_history_stops_at_30_costs() -> None:
    store = InMemoryResearchStore(
        [_history(date(2026, 9, 28), *[0.5] * 30), _history(date(2026, 9, 29), *[1.0] * 30)]
    )
    report = _run(World(price_table=PRICED, research_store=store), budget=100.0)
    assert report.budget is not None
    assert report.budget.estimate == Estimate(cost=1.0, source=EstimateSource.HISTORY, n_samples=30)
    assert store.loaded == [AS_OF, date(2026, 9, 29)]


def test_the_history_reads_at_most_60_dates() -> None:
    days = [date(2026, 7, 1) + timedelta(days=offset) for offset in range(61)]
    store = InMemoryResearchStore([ResearchResult(as_of=day, names=()) for day in days])
    table = dataclasses.replace(PRICED, fallback_run_cost=1.0)
    report = _run(World(price_table=table, research_store=store), budget=100.0)
    assert report.budget is not None
    assert report.budget.estimate.source is EstimateSource.FALLBACK
    assert store.loaded == [AS_OF, *reversed(days[1:])]


def test_a_malformed_history_date_is_skipped() -> None:
    bad = date(2026, 9, 29)
    store = _BadDayStore([_history(date(2026, 9, 28), 0.5), _history(bad, 9.0)], bad=bad)
    report = _run(World(price_table=PRICED, research_store=store), budget=100.0)
    assert report.budget is not None
    assert report.budget.estimate == Estimate(cost=0.5, source=EstimateSource.HISTORY, n_samples=1)
    assert store.loaded == [AS_OF, bad, date(2026, 9, 28)]


def test_no_estimate_fails_the_budget_stage_before_any_llm_call() -> None:
    world = World(price_table=PriceTable())
    with pytest.raises(BudgetError, match="no run cost history and no fallback_run_cost"):
        _run(world, budget=10.0)
    [report] = world.report_store.saved
    assert (report.outcome, report.failed_stage) == (CycleOutcome.ERROR, "budget")
    assert report.error is not None
    assert report.error.startswith("BudgetError: ")
    assert report.budget is None
    assert world.researcher.calls == []
    assert world.research_store.saved == []


@pytest.mark.parametrize("budget", [pytest.param(0.0, id="zero"), pytest.param(math.nan, id="nan")])
def test_a_bad_budget_fails_the_inputs_even_when_research_would_be_reused(budget: float) -> None:
    world = World(price_table=FALLBACK)
    stored = _stored(world)  # matching research: the plan would never run
    with pytest.raises(BudgetError, match="budget must be a finite number > 0"):
        _run(world, budget=budget)
    report = world.report_store.saved[-1]
    assert (report.outcome, report.failed_stage) == (CycleOutcome.ERROR, "inputs")
    assert report.account is None  # before any broker read
    assert report.research is None
    assert world.researcher.calls == []
    assert world.research_store.saved == [stored]


# --- reuse -------------------------------------------------------------------------------


def _stored(world: World, budget: float | None = None) -> ResearchResult:
    """Run one dry cycle so the store holds today's research; forget the researcher calls."""
    research = _run(world, budget=budget).research
    assert research is not None
    world.researcher.calls.clear()
    return research


def test_reused_research_skips_the_plan() -> None:
    world = World(price_table=FALLBACK)
    stored = _stored(world)
    report = _run(world, budget=1.0)
    assert report.research_reused is True
    assert report.research is stored
    assert report.budget is None
    assert report.spend == Spend(total=0.0, runs=0, unknown_runs=0, usage=(), over_budget=False)
    assert world.researcher.calls == []


def _reverse(result: ResearchResult) -> ResearchResult:
    return dataclasses.replace(result, names=result.names[::-1])


def _one_run_each(result: ResearchResult) -> ResearchResult:
    names = tuple(dataclasses.replace(name, runs=name.runs[:1]) for name in result.names)
    return dataclasses.replace(result, names=names)


def _no_names(result: ResearchResult) -> ResearchResult:
    return dataclasses.replace(result, names=())


@pytest.mark.parametrize(
    ("change", "budget", "calls"),
    [
        pytest.param(None, 8.0, 0, id="budgeted-prefix-is-reused"),
        pytest.param(None, None, 18, id="unbudgeted-needs-the-whole-shortlist"),
        pytest.param(_reverse, 8.0, 8, id="not-a-prefix"),
        pytest.param(_one_run_each, 8.0, 8, id="fewer-than-min-k-runs"),
        pytest.param(_no_names, 8.0, 8, id="empty"),
    ],
)
def test_budgeted_research_is_reused_only_under_a_budget(
    change: Callable[[ResearchResult], ResearchResult] | None,
    budget: float | None,
    calls: int,
) -> None:
    world = World(price_table=FALLBACK)
    stored = _stored(world, budget=8.0)
    assert len(stored.names) == 4
    if change is not None:
        world.research_store.saved = [change(stored)]
    report = _run(world, budget=budget)
    assert len(world.researcher.calls) == calls
    assert report.research_reused is (calls == 0)


# --- spend -------------------------------------------------------------------------------


def test_without_a_budget_spend_is_reported_and_errored_runs_count() -> None:
    world = World(price_table=PRICED)
    world.researcher.usage = ONE_DOLLAR
    world.researcher.script[("T0", 0)] = ResearchRunError("RuntimeError: boom", ONE_DOLLAR)
    report = _run(world)
    assert report.budget is None
    assert report.spend == Spend(
        total=18.0,
        runs=18,
        unknown_runs=0,
        usage=(ModelUsage("m", 18, 18_000_000, 0, 0, 0),),
        over_budget=None,
    )
    assert report.research is not None
    errored = [run for name in report.research.names for run in name.runs if run.error]
    assert [run.error for run in errored] == ["RuntimeError: boom"]
    assert errored[0].usage == ONE_DOLLAR


def test_spend_over_the_budget_is_reported_not_enforced() -> None:
    store = InMemoryResearchStore([_history(date(2026, 9, 29), 0.5, 0.5, 0.5)])
    world = World(price_table=PRICED, research_store=store)
    world.researcher.usage = ONE_DOLLAR  # each run costs 1.00: twice the estimate
    report = _run(world, budget=5.0)
    assert report.outcome is CycleOutcome.DRY_RUN
    assert report.budget is not None
    assert (report.budget.n, report.budget.k) == (5, 2)
    assert report.budget.reasons == (
        "k 3→2: 6×3×$0.50=$9.00 > $5.00",
        "n 6→5: 6×2×$0.50=$6.00 > $5.00",
    )
    assert report.spend == Spend(
        total=10.0,
        runs=10,
        unknown_runs=0,
        usage=(ModelUsage("m", 10, 10_000_000, 0, 0, 0),),
        over_budget=True,
    )


def test_a_failure_after_research_still_reports_the_plan_and_the_spend() -> None:
    world = World(price_table=FALLBACK, research_store=_FailingSaveStore())
    with pytest.raises(OSError, match="disk full"):
        _run(world, budget=12.0)
    [report] = world.report_store.saved
    assert (report.outcome, report.failed_stage) == (CycleOutcome.ERROR, "research")
    assert report.budget is not None
    assert report.budget.k == 2
    assert report.spend is not None
    assert report.spend.runs == 12
    assert report.research is None
