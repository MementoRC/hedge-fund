import dataclasses
import re
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import UTC, date, datetime
from typing import Any

import numpy as np
import pandas as pd
import pytest

from hedge_fund.adapters.approval.auto import AutoApprovalGate
from hedge_fund.adapters.broker.simulated import SimulatedBroker
from hedge_fund.adapters.data.frozen_providers import FrozenDataProvider, FrozenFundamentalsProvider
from hedge_fund.adapters.research.neutral import NeutralResearcher
from hedge_fund.domain.account import AccountSnapshot
from hedge_fund.domain.backtest import (
    MEMBERSHIP_FLOOR,
    BacktestReport,
    FlaggedSpan,
    FrozenData,
    MemberSpan,
    SimCosts,
)
from hedge_fund.domain.cycle import CycleOptions, CycleOutcome, CycleReport
from hedge_fund.domain.shortlist import Candidate, Shortlist, Side
from hedge_fund.engine.backtest.clock import SimClock
from hedge_fund.engine.backtest.pit_prices import closes_as_of
from hedge_fund.ports.research import Researcher
from hedge_fund.services.backtest import (
    ACCEPTED_OUTCOMES,
    BacktestDeps,
    BacktestError,
    backtest_run_id,
    run_backtest,
)
from hedge_fund.services.cycle import CycleDeps
from tests.factories import make_frozen_data
from tests.fakes import (
    InMemoryReportStore,
    InMemoryResearchStore,
    InMemorySnapshotStore,
    ScriptedBroker,
)

START = date(2024, 1, 29)
END = date(2024, 2, 2)  # Jan 29, 30, 31, Feb 1, 2: one rebalance, Feb 1
FEB_1 = date(2024, 2, 1)
EXTRA = Candidate("ZZZ", "Energy", Side.LONG, 1, 1.0)


def _planned(deps: CycleDeps, as_of: date) -> Shortlist:
    """The shortlist the tests plan: the tiny store has no fundamentals at its dates."""
    return Shortlist(as_of, (Candidate("AAA", "Energy", Side.LONG, 1, 1.0),), {})


@pytest.fixture(autouse=True)
def _plan(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("hedge_fund.services.backtest.plan_shortlist", _planned)


class RecordingResearcher:
    """The ``researcher`` factory: records each shortlist and builds a NeutralResearcher."""

    def __init__(self) -> None:
        self.shortlists: list[Shortlist] = []
        self.made: list[Researcher] = []

    def __call__(self, shortlist: Shortlist) -> Researcher:
        self.shortlists.append(shortlist)
        self.made.append(NeutralResearcher(shortlist))
        return self.made[-1]


def _deps(data: FrozenData | None = None, **changes: Any) -> BacktestDeps:
    data = data or make_frozen_data()
    base = BacktestDeps(
        data=data,
        store_digest="a" * 64,
        cash=100_000.0,
        costs=SimCosts(),
        broker=SimulatedBroker(data, 100_000.0),
        snapshot_store=InMemorySnapshotStore(),
        report_store=InMemoryReportStore(),
        research_store=InMemoryResearchStore(),
        gate=AutoApprovalGate(),
        prices=lambda day: FrozenDataProvider(data, day),
        fundamentals=FrozenFundamentalsProvider(data),
        researcher=NeutralResearcher,
    )
    return replace(base, **changes)


def _cycle_report(
    as_of: date, started_at: datetime, outcome: CycleOutcome, shortlist: Shortlist | None
) -> CycleReport:
    return CycleReport(
        as_of=as_of,
        started_at=started_at,
        dry_run=False,
        outcome=outcome,
        failed_stage=None,
        error=None,
        account=None,
        price_unresolved={},
        fundamentals_unresolved={},
        shortlist=shortlist,
        research=None,
        research_reused=False,
        target=None,
        verdict=None,
        order_prices=None,
        plan=None,
        decision=None,
        execution=None,
        execution_interrupted=False,
        planned_client_order_ids=(),
    )


@dataclass
class Call:
    deps: CycleDeps
    as_of: date
    today: date
    options: CycleOptions
    started: datetime
    snapshots_before: int


@dataclass
class StubCycle:
    """A ``cycle`` that records its calls and reports ``outcome`` with the planned shortlist."""

    store: InMemorySnapshotStore
    outcome: CycleOutcome = CycleOutcome.SUBMITTED
    shortlist: Callable[[Shortlist], Shortlist | None] = lambda planned: planned
    raises: Exception | None = None
    calls: list[Call] = field(default_factory=list)

    def __call__(
        self,
        deps: CycleDeps,
        *,
        as_of: date,
        today: date,
        options: CycleOptions,
        now: Callable[[], datetime],
    ) -> CycleReport:
        started = now()
        self.calls.append(Call(deps, as_of, today, options, started, len(self.store.saved)))
        if self.raises is not None:
            raise self.raises
        planned = _planned(deps, as_of)
        return _cycle_report(as_of, started, self.outcome, self.shortlist(planned))


def _run(
    deps: BacktestDeps | None = None, start: date = START, end: date = END, **stub: Any
) -> tuple[BacktestReport, StubCycle, BacktestDeps]:
    deps = deps or _deps()
    assert isinstance(deps.snapshot_store, InMemorySnapshotStore)
    cycle = StubCycle(deps.snapshot_store, **stub)
    report = run_backtest(deps, start=start, end=end, cycle=cycle)
    return report, cycle, deps


def _snapshot() -> AccountSnapshot:
    return AccountSnapshot(
        as_of=START, observed_at=datetime(2024, 1, 29, tzinfo=UTC), equity=1.0, holdings=()
    )


def _held_runs() -> InMemoryReportStore:
    store = InMemoryReportStore()
    store.unreadable = {"2024-01-01_000000.json": "ValueError: bad"}
    return store


def _guard_cases() -> list[tuple[str, dict[str, Any], date, date, str]]:
    return [
        ("broker", {"broker": ScriptedBroker()}, START, END, "only runs against SimulatedBroker"),
        (
            "snapshots",
            {"snapshot_store": InMemorySnapshotStore([_snapshot()])},
            START,
            END,
            "--root already holds snapshots or runs",
        ),
        (
            "unreadable runs",
            {"report_store": _held_runs()},
            START,
            END,
            "--root already holds unreadable run files",
        ),
        (
            "runs",
            {
                "report_store": InMemoryReportStore(
                    [
                        _cycle_report(
                            START, datetime(2024, 1, 29, tzinfo=UTC), CycleOutcome.ERROR, None
                        )
                    ]
                )
            },
            START,
            END,
            "--root already holds snapshots or runs",
        ),
        ("floor", {}, date(2007, 12, 31), END, str(MEMBERSHIP_FLOOR)),
        ("before the store", {}, date(2023, 12, 29), END, "window outside the store's coverage"),
        ("after the store", {}, START, date(2024, 2, 12), "window outside the store's coverage"),
        (
            "reversed",
            {},
            date(2024, 2, 1),
            date(2024, 1, 31),
            "window outside the store's coverage",
        ),
        ("one day", {}, START, START, "at least three trading days"),
        ("two days", {}, START, date(2024, 1, 30), "at least three trading days"),
        ("weekend", {}, date(2024, 1, 5), date(2024, 1, 7), "at least three trading days"),
    ]


@pytest.mark.parametrize(
    ("name", "changes", "start", "end", "message"),
    _guard_cases(),
    ids=[case[0] for case in _guard_cases()],
)
def test_guard_refuses_before_any_cycle(
    name: str, changes: dict[str, Any], start: date, end: date, message: str
) -> None:
    deps = _deps(**changes)
    cycle = StubCycle(InMemorySnapshotStore())
    with pytest.raises(BacktestError, match=re.escape(message)):
        run_backtest(deps, start=start, end=end, cycle=cycle)
    assert cycle.calls == []


def test_guards_run_in_order() -> None:
    deps = _deps(broker=ScriptedBroker(), snapshot_store=InMemorySnapshotStore([_snapshot()]))
    with pytest.raises(BacktestError, match="SimulatedBroker"):
        run_backtest(
            deps,
            start=date(2007, 1, 1),
            end=date(2007, 1, 1),
            cycle=StubCycle(InMemorySnapshotStore()),
        )


def test_run_id_format_and_determinism() -> None:
    first = backtest_run_id(_deps(), START, END)
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}_\d{4}-\d{2}-\d{2}_[0-9a-f]{12}", first)
    assert first.startswith("2024-01-29_2024-02-02_")
    assert backtest_run_id(_deps(), START, END) == first


@pytest.mark.parametrize(
    "changes",
    [
        {"cash": 50_000.0},
        {"costs": SimCosts(slippage_bps=10.0)},
        {"costs": SimCosts(borrow_rate=0.01)},
        {"store_digest": "b" * 64},
    ],
)
def test_run_id_is_sensitive_to_its_inputs(changes: dict[str, Any]) -> None:
    assert backtest_run_id(_deps(**changes), START, END) != backtest_run_id(_deps(), START, END)


def test_run_id_is_sensitive_to_the_window() -> None:
    base = backtest_run_id(_deps(), START, END)
    assert backtest_run_id(_deps(), START, date(2024, 2, 5)) != base
    assert backtest_run_id(_deps(), date(2024, 1, 30), END) != base


def test_run_id_is_sensitive_to_params() -> None:
    deps = _deps()
    other = replace(deps.params, breakers=replace(deps.params.breakers, daily_loss=0.5))
    assert backtest_run_id(replace(deps, params=other), START, END) != backtest_run_id(
        deps, START, END
    )


def test_one_snapshot_per_window_day() -> None:
    _, _, deps = _run()
    assert isinstance(deps.snapshot_store, InMemorySnapshotStore)
    assert [snap.as_of for snap in deps.snapshot_store.saved] == [
        date(2024, 1, 29),
        date(2024, 1, 30),
        date(2024, 1, 31),
        date(2024, 2, 1),
        date(2024, 2, 2),
    ]


def test_cycle_runs_on_rebalance_days_only() -> None:
    _, cycle, _ = _run(start=date(2024, 1, 1), end=date(2024, 2, 9))
    assert [call.as_of for call in cycle.calls] == [date(2024, 1, 1), FEB_1]


def test_mid_month_start_waits_for_the_next_months_first_day() -> None:
    _, cycle, _ = _run(start=date(2024, 1, 10), end=date(2024, 2, 5))
    assert [call.as_of for call in cycle.calls] == [FEB_1]


def test_cycle_arguments() -> None:
    researcher = RecordingResearcher()
    _, cycle, deps = _run(_deps(researcher=researcher))
    (call,) = cycle.calls
    assert call.as_of == call.today == FEB_1
    assert call.options == CycleOptions(any_time=True, fresh_research=True, budget=None)
    assert call.deps.universe.list_date == FEB_1
    assert call.deps.params is deps.params
    assert call.deps.broker is deps.broker
    assert call.deps.gate is deps.gate
    assert call.deps.report_store is deps.report_store
    assert call.deps.snapshot_store is deps.snapshot_store
    assert call.deps.research_store is deps.research_store
    assert call.deps.fundamentals is deps.fundamentals
    assert isinstance(call.deps.prices, FrozenDataProvider)
    assert call.deps.prices.as_of == FEB_1
    assert call.deps.researcher is researcher.made[-1]
    assert researcher.shortlists[-1] == _planned(call.deps, FEB_1)


def test_planning_failure_aborts_with_the_cause(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(deps: CycleDeps, as_of: date) -> Shortlist:
        raise ValueError("no fundamentals")

    monkeypatch.setattr("hedge_fund.services.backtest.plan_shortlist", fail)
    cycle = StubCycle(InMemorySnapshotStore())
    with pytest.raises(
        BacktestError, match=f"shortlist planning failed on {FEB_1}: no fund"
    ) as raised:
        run_backtest(_deps(), start=START, end=END, cycle=cycle)
    assert isinstance(raised.value.__cause__, ValueError)
    assert cycle.calls == []


def test_snapshot_is_taken_before_the_cycle() -> None:
    _, cycle, _ = _run()
    assert cycle.calls[0].snapshots_before == 4  # Jan 29, 30, 31 and Feb 1


def test_cycle_clock_is_after_the_days_snapshot() -> None:
    _, cycle, deps = _run()
    assert isinstance(deps.snapshot_store, InMemorySnapshotStore)
    observed = {snap.as_of: snap.observed_at for snap in deps.snapshot_store.saved}
    assert cycle.calls[0].started > observed[FEB_1]
    assert cycle.calls[0].started.date() == FEB_1


def test_custom_clock_is_used() -> None:
    clock = SimClock()
    clock.start_day(date(2024, 3, 1))
    clock()
    deps = _deps()
    with pytest.raises(ValueError, match="failed to move forward"):
        run_backtest(
            deps, start=START, end=END, cycle=StubCycle(InMemorySnapshotStore()), clock=clock
        )


@pytest.mark.parametrize("outcome", sorted(ACCEPTED_OUTCOMES))
def test_accepted_outcomes_do_not_abort(outcome: CycleOutcome) -> None:
    report, _, _ = _run(outcome=outcome)
    assert report.rebalances == 1


@pytest.mark.parametrize(
    "outcome",
    [
        CycleOutcome.ERROR,
        CycleOutcome.INTERRUPTED,
        CycleOutcome.REJECTED,
        CycleOutcome.DRY_RUN,
    ],
)
def test_other_outcomes_abort(outcome: CycleOutcome) -> None:
    with pytest.raises(BacktestError, match=f"{outcome.value}.*{FEB_1}|{FEB_1}.*{outcome.value}"):
        _run(outcome=outcome)


def test_a_cycle_exception_aborts_with_the_cause() -> None:
    boom = RuntimeError("boom")
    with pytest.raises(BacktestError, match=f"run_cycle failed on {FEB_1}: boom") as raised:
        _run(raises=boom)
    assert raised.value.__cause__ is boom


def test_a_shortlist_mismatch_aborts() -> None:
    with pytest.raises(BacktestError, match="shortlist mismatch"):
        _run(shortlist=lambda planned: replace(planned, candidates=(*planned.candidates, EXTRA)))


def test_a_flipped_side_aborts() -> None:
    def flip(planned: Shortlist) -> Shortlist:
        return replace(
            planned, candidates=tuple(replace(c, side=Side.SHORT) for c in planned.candidates)
        )

    with pytest.raises(BacktestError, match="shortlist mismatch"):
        _run(shortlist=flip)


def test_a_changed_sector_aborts() -> None:
    def move(planned: Shortlist) -> Shortlist:
        return replace(
            planned, candidates=tuple(replace(c, sector="Tech") for c in planned.candidates)
        )

    with pytest.raises(BacktestError, match="shortlist mismatch"):
        _run(shortlist=move)


def test_a_report_without_a_shortlist_does_not_abort() -> None:
    report, _, _ = _run(shortlist=lambda planned: None)
    assert report.rebalances == 1


def test_report_fields() -> None:
    deps = _deps(store_digest="c" * 64, cash=100_000.0, costs=SimCosts(slippage_bps=7.0))
    report, _, _ = _run(deps)
    assert report.run_id == backtest_run_id(deps, START, END)
    assert report.store_digest == "c" * 64
    assert (report.start, report.end) == (START, END)
    assert report.cash == 100_000.0
    assert report.costs == SimCosts(slippage_bps=7.0)
    assert report.trading_days == 5
    assert report.rebalances == 1
    assert report.metrics.strategy.start == START
    assert report.metrics.strategy.end == END


def test_benchmark_is_spy_over_the_window() -> None:
    data = make_frozen_data()
    report, _, _ = _run(_deps(data))
    spy = closes_as_of(data.closes["SPY"], "SPY", data.splits, data.dividends, as_of=END)
    window = spy.loc[pd.Timestamp(START) : pd.Timestamp(END)]
    assert report.metrics.benchmark.total_return == pytest.approx(
        window.iloc[-1] / window.iloc[0] - 1
    )
    assert (report.metrics.benchmark.start, report.metrics.benchmark.end) == (START, END)


def test_flagged_spans_are_counted_when_they_exclude_members() -> None:
    flagged = (FlaggedSpan(MemberSpan("AAA", date(2024, 1, 1), None), "another company"),)
    report, _, _ = _run(_deps(make_frozen_data(flagged=flagged)))
    assert report.metrics.honesty.flagged_excluded == 1


def test_a_missing_spy_close_aborts() -> None:
    data = make_frozen_data()
    closes = data.closes.copy()
    closes.loc[pd.Timestamp("2024-01-31"), "SPY"] = np.nan
    with pytest.raises(BacktestError, match=r"SPY.*2024-01-31"):
        _run(_deps(dataclasses.replace(data, closes=closes)))


def test_a_store_without_spy_aborts() -> None:
    data = make_frozen_data()
    stripped = dataclasses.replace(
        data, closes=data.closes.drop(columns="SPY"), opens=data.opens.drop(columns="SPY")
    )
    with pytest.raises(BacktestError, match="SPY"):
        _run(_deps(stripped))
