"""Layer 6d pipeline cycle: one paper-trading run over fakes, and its audit report."""

from __future__ import annotations

import dataclasses
import functools
import math
import re
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from typing import Any, NoReturn

import pytest

from hedge_fund.domain.account import AccountSnapshot
from hedge_fund.domain.approval import ApprovalStatus
from hedge_fund.domain.cycle import CycleOptions, CycleOutcome, CycleReport
from hedge_fund.domain.execution import SubmitStatus
from hedge_fund.domain.ledger import OrderState, OrderStatus
from hedge_fund.domain.orders import Holding, OrderAction, OrderPlan
from hedge_fund.domain.research import ResearchResult, Verdict
from hedge_fund.domain.risk import NavPoint
from hedge_fund.engine.orders import OrderParams
from hedge_fund.engine.portfolio.alpha import AlphaParams
from hedge_fund.engine.portfolio.correlation import CorrelationParams
from hedge_fund.engine.portfolio.optimize import OptimizeParams
from hedge_fund.engine.portfolio.risk_model import RiskParams
from hedge_fund.engine.research.conviction import ConvictionParams
from hedge_fund.engine.risk.breakers import BreakerParams
from hedge_fund.engine.risk.limits import DEFAULT_LIMITS, RiskLimits
from hedge_fund.engine.shortlist.select import ShortlistParams
from hedge_fund.ports.account import SnapshotStoreError
from hedge_fund.ports.broker import BrokerError, OrderDataError
from hedge_fund.services import cycle
from hedge_fund.services.cycle import (
    PendingOrdersError,
    ReportSaveError,
    SubmissionWindowError,
    run_cycle,
)
from hedge_fund.services.portfolio import build_portfolio, fetch_benchmark
from hedge_fund.services.risk import assess_portfolio
from hedge_fund.services.snapshots import build_snapshot
from tests.cycle_world import (
    AS_OF,
    EVENING,
    FRAME,
    HELD_OUTSIDE,
    LOOSE_LIMITS,
    SECTORS,
    SESSION,
    TODAY,
    WEEKEND,
    Clock,
    World,
    universe,
)
from tests.fakes import (
    FakeProvider,
    InMemoryReportStore,
    InMemorySnapshotStore,
    ScriptedApprovalGate,
    ScriptedBroker,
    ScriptedResearcher,
    not_found,
)

EQUITY = 1_000_000.0
DRY = CycleOptions(dry_run=True)


def _run(
    world: World,
    *,
    options: CycleOptions | None = None,
    as_of: date = AS_OF,
    today: date = TODAY,
    now: Clock | None = None,
) -> CycleReport:
    return run_cycle(
        world.deps,
        as_of=as_of,
        today=today,
        options=options or CycleOptions(),
        now=now or Clock(EVENING),
    )


def _read(day: date, equity: float) -> AccountSnapshot:
    """A stored account read for ``day``, taken at 22:00 UTC."""
    return AccountSnapshot(
        as_of=day,
        observed_at=datetime(day.year, day.month, day.day, 22, 0, tzinfo=UTC),
        equity=equity,
        holdings=(),
    )


def _halting_store() -> InMemorySnapshotStore:
    """Yesterday's equity 3% above today's 1,000,000: the daily-loss breaker trips."""
    return InMemorySnapshotStore([_read(AS_OF - timedelta(days=1), 1_030_000.0)])


def _current(now: datetime = EVENING, *, as_of: date = AS_OF) -> AccountSnapshot:
    return AccountSnapshot(as_of=as_of, observed_at=now, equity=EQUITY, holdings=())


def _untouched(world: World) -> None:
    """No data, research, approval or broker write happened."""
    assert world.prices.closes_calls == []
    assert world.prices.sectors_calls == []
    assert world.fundamentals.calls == []
    assert world.researcher.calls == []
    assert world.gate.summaries == []
    assert world.broker.submitted == []


# --- step 0: inputs ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "as_of",
    [
        pytest.param(date(2026, 9, 29), id="before-the-previous-weekday"),
        pytest.param(date(2026, 10, 2), id="after-today"),
    ],
)
def test_an_as_of_that_is_not_live_fails_before_any_broker_read(as_of: date) -> None:
    world = World(broker=ScriptedBroker(account_error=AssertionError("broker was read")))
    with pytest.raises(ValueError, match=f"as_of {as_of} is not live"):
        _run(world, as_of=as_of)
    [report] = world.report_store.saved
    assert (report.outcome, report.failed_stage) == (CycleOutcome.ERROR, "inputs")
    assert report.error is not None
    assert report.error.startswith(f"ValueError: as_of {as_of} is not live")
    assert report.account is None
    assert world.snapshot_store.saved == []
    _untouched(world)


def test_a_previous_weekday_as_of_after_the_open_fails_before_any_broker_read() -> None:
    # Audit G3: Wed closed at 100; a Thursday run after the open must not re-date its moving
    # equity as Wednesday's, or nav_history would let it replace the 100 point.
    stored = [
        _read(date(2026, 9, 28), 95.0),
        _read(date(2026, 9, 29), 95.0),
        dataclasses.replace(
            _read(AS_OF, 100.0), observed_at=datetime(2026, 9, 30, 12, 0, tzinfo=UTC)
        ),
    ]
    world = World(
        broker=ScriptedBroker(account_error=AssertionError("broker was read")),
        snapshot_store=InMemorySnapshotStore(stored),
    )
    after_open = datetime(2026, 10, 1, 13, 30, tzinfo=UTC)  # Thu 09:30:00 EDT
    with pytest.raises(ValueError, match=rf"as_of {AS_OF} is not live: .*opened at 09:30"):
        _run(world, now=Clock(after_open))
    assert world.snapshot_store.saved == stored
    _untouched(world)


def test_a_previous_weekday_as_of_after_the_open_fails_in_a_dry_run_and_any_time() -> None:
    after_open = datetime(2026, 10, 1, 20, 0, tzinfo=UTC)
    for options in (DRY, CycleOptions(any_time=True)):
        world = World(broker=ScriptedBroker(account_error=AssertionError("broker was read")))
        with pytest.raises(ValueError, match="use as_of 2026-10-01"):
            _run(world, options=options, now=Clock(after_open))


def test_a_previous_weekday_as_of_before_the_open_still_works() -> None:
    world = World(snapshot_store=_halting_store())
    before_open = datetime(2026, 10, 1, 12, 0, tzinfo=UTC)  # Thu 08:00 EDT
    report = _run(world, now=Clock(before_open))
    assert report.outcome is CycleOutcome.HALTED_PREFLIGHT
    assert report.as_of == AS_OF


@pytest.mark.parametrize(
    "as_of",
    [pytest.param(AS_OF, id="previous-weekday"), pytest.param(TODAY, id="today")],
)
def test_the_live_bounds_are_inclusive(as_of: date) -> None:
    world = World(snapshot_store=_halting_store())
    world.snapshot_store.saved[0] = _read(as_of - timedelta(days=1), 1_030_000.0)
    report = _run(world, as_of=as_of)
    assert report.outcome is CycleOutcome.HALTED_PREFLIGHT
    assert report.as_of == as_of


# --- steps 1-3: account, nav, preflight --------------------------------------------------


def test_a_daily_loss_halts_before_any_data_or_research() -> None:
    world = World(snapshot_store=_halting_store())
    report = _run(world)
    assert report.outcome is CycleOutcome.HALTED_PREFLIGHT
    assert (report.failed_stage, report.error) == (None, None)
    assert report.account == _current()
    assert world.snapshot_store.saved == [_halting_store().saved[0], _current()]
    assert world.report_store.saved == [report]
    assert report.shortlist is None
    assert report.plan is None
    _untouched(world)


def test_a_drawdown_halts_too() -> None:
    # Peak 1,090,000 two days ago, flat since: daily loss 0, drawdown 8.3%.
    reads = [_read(AS_OF - timedelta(days=2), 1_090_000.0), _read(AS_OF - timedelta(1), EQUITY)]
    world = World(snapshot_store=InMemorySnapshotStore(reads))
    assert _run(world).outcome is CycleOutcome.HALTED_PREFLIGHT
    _untouched(world)


def test_a_dry_run_halts_on_todays_read_without_saving_it() -> None:
    world = World(snapshot_store=_halting_store())
    report = _run(world, options=CycleOptions(dry_run=True))
    assert report.outcome is CycleOutcome.HALTED_PREFLIGHT
    assert report.dry_run is True
    assert world.snapshot_store.saved == _halting_store().saved
    _untouched(world)


def test_started_at_is_the_first_clock_read_and_the_account_the_second() -> None:
    first, second = EVENING, EVENING + timedelta(seconds=5)
    world = World(snapshot_store=_halting_store())
    report = _run(world, now=Clock(first, second))
    assert report.started_at == first
    assert report.account is not None
    assert report.account.observed_at == second


# --- failures: ERROR / INTERRUPTED reports, and the raise ---------------------------------


@pytest.mark.parametrize(
    ("world", "raised", "stage", "error"),
    [
        pytest.param(
            World(broker=ScriptedBroker(account_error=BrokerError("account down"))),
            BrokerError,
            "account",
            "BrokerError: account down",
            id="account",
        ),
        pytest.param(
            World(snapshot_store=InMemorySnapshotStore(fail_on_read=SnapshotStoreError("x.json"))),
            SnapshotStoreError,
            "nav",
            "SnapshotStoreError: x.json",
            id="nav",
        ),
        pytest.param(
            World(params=dataclasses.replace(World().params, breakers=BreakerParams(2.0))),
            ValueError,
            "preflight",
            "ValueError: daily_loss must satisfy 0 < daily_loss < 1, got 2.0",
            id="preflight",
        ),
    ],
)
def test_a_failing_stage_raises_and_saves_an_error_report(
    world: World, raised: type[Exception], stage: str, error: str
) -> None:
    with pytest.raises(raised):
        _run(world)
    [report] = world.report_store.saved
    assert report.outcome is CycleOutcome.ERROR
    assert (report.failed_stage, report.error) == (stage, error)
    assert report.as_of == AS_OF
    assert report.started_at == EVENING


def test_an_interrupt_raises_and_saves_an_interrupted_report() -> None:
    world = World(broker=ScriptedBroker(account_error=KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        _run(world)
    [report] = world.report_store.saved
    assert report.outcome is CycleOutcome.INTERRUPTED
    assert (report.failed_stage, report.error) == ("account", "KeyboardInterrupt: ")


# --- report save failures ----------------------------------------------------------------


def test_a_completed_cycle_whose_report_cannot_be_saved_raises_report_save_error() -> None:
    disk_full = OSError("disk full")
    world = World(snapshot_store=_halting_store(), report_store=InMemoryReportStore(fail=disk_full))
    with pytest.raises(ReportSaveError, match="could not be saved") as info:
        _run(world)
    assert info.value.report.outcome is CycleOutcome.HALTED_PREFLIGHT
    assert info.value.report.account == _current()
    assert info.value.__cause__ is disk_full


def test_a_submitted_cycle_whose_report_cannot_be_saved_still_carries_the_execution() -> None:
    disk_full = OSError("disk full")
    world = World(report_store=InMemoryReportStore(fail=disk_full))
    with pytest.raises(ReportSaveError, match="could not be saved") as info:
        _run(world)
    report = info.value.report
    assert report.outcome is CycleOutcome.SUBMITTED
    assert report.decision is not None
    assert report.decision.status is ApprovalStatus.APPROVED
    assert report.execution is not None
    submitted = [result.request.client_order_id for result in report.execution.results]
    assert submitted != []
    assert submitted == world.broker.submitted
    assert submitted == _ids(report.plan)
    assert info.value.__cause__ is disk_full


def test_a_failed_cycle_whose_report_cannot_be_saved_raises_the_original_error() -> None:
    world = World(
        broker=ScriptedBroker(account_error=BrokerError("account down")),
        report_store=InMemoryReportStore(fail=OSError("disk full")),
    )
    with pytest.raises(BrokerError, match="account down") as info:
        _run(world)
    assert info.value.__notes__ == ["the cycle report could not be saved: OSError: disk full"]


# --- steps 4-10: data to the order plan, in dry runs --------------------------------------


def _params(**changes: Any) -> Any:
    return dataclasses.replace(World().params, **changes)


def test_a_dry_run_builds_the_whole_plan_and_writes_no_account_state() -> None:
    world = World()
    report = _run(world, options=DRY)
    assert report.outcome is CycleOutcome.DRY_RUN
    assert (report.failed_stage, report.error) == (None, None)
    assert report.shortlist is not None
    assert {candidate.ticker for candidate in report.shortlist.candidates} == set(SECTORS)
    assert report.research is not None
    assert [name.verdict for name in report.research.names] == [Verdict.CONFIRM] * 6
    assert report.research_reused is False
    assert report.target is not None
    assert {position.ticker for position in report.target.positions} == set(SECTORS)
    assert report.verdict is not None
    assert (report.verdict.approved, report.verdict.halt) == (True, False)
    last = FRAME.iloc[-1]
    assert report.order_prices == {
        ticker: (float(last[ticker]), AS_OF) for ticker in sorted(SECTORS)
    }
    assert report.plan is not None
    assert report.plan.blocked is None
    assert {order.ticker for order in report.plan.orders} == set(SECTORS)
    assert all(order.ref_price == float(last[order.ticker]) for order in report.plan.orders)
    assert (report.decision, report.execution) == (None, None)
    assert world.gate.summaries == []
    assert world.broker.submitted == []
    assert world.snapshot_store.saved == []
    assert world.research_store.saved == [report.research]
    assert world.report_store.saved == [report]


def test_the_same_as_of_reaches_every_stage() -> None:
    world = World()
    report = _run(world, options=DRY)
    assert [call[0] for call in world.prices.closes_calls] == [tuple(SECTORS), ("SPY",)]
    assert {call[2] for call in world.prices.closes_calls} == {AS_OF}
    assert {call[1] for call in world.researcher.calls} == {AS_OF}
    for stage in ("account", "shortlist", "research", "target", "verdict", "plan"):
        assert getattr(report, stage).as_of == AS_OF, stage


def test_unresolved_names_are_reported_and_only_priced_names_get_fundamentals() -> None:
    # NOSEC has no sector, NOPRICE has a sector but no closes, T0 has no CIK.
    prices = FakeProvider(FRAME, {**SECTORS, "NOPRICE": "Tech"})
    listed = universe({**SECTORS, "NOSEC": "Tech", "NOPRICE": "Tech"})
    ciks = {ticker: cik for ticker, cik in listed.ciks.items() if ticker != "T0"}
    world = World(prices=prices, universe=dataclasses.replace(listed, ciks=ciks))
    report = _run(world, options=DRY)
    assert report.price_unresolved == {"NOPRICE": "no_prices", "NOSEC": "no_sector"}
    assert report.fundamentals_unresolved == {"T0": "no_cik"}
    assert world.fundamentals.calls == [tuple(ticker for ticker in SECTORS if ticker != "T0")]


def test_a_held_name_missing_from_the_snapshot_is_fetched_and_traded_to_zero() -> None:
    world = World(broker=ScriptedBroker(holdings=(Holding(ticker=HELD_OUTSIDE, shares=100),)))
    report = _run(world, options=DRY)
    window = AS_OF - timedelta(days=14)
    assert world.prices.closes_calls[-1] == ((HELD_OUTSIDE,), window, AS_OF)
    assert report.order_prices is not None
    assert report.order_prices[HELD_OUTSIDE] == (float(FRAME[HELD_OUTSIDE].iloc[-1]), AS_OF)
    assert list(report.order_prices) == sorted([*SECTORS, HELD_OUTSIDE])
    assert report.plan is not None
    [exit_leg] = [order for order in report.plan.orders if order.ticker == HELD_OUTSIDE]
    assert (exit_leg.action, exit_leg.shares) == (OrderAction.SELL, 100)


def _stale(column: str, last: str) -> Any:
    """FRAME with ``column``'s closes after ``last`` set to NaN."""
    frame = FRAME.copy()
    frame.loc[frame.index > last, column] = math.nan
    return frame


@pytest.mark.parametrize(
    ("frame", "holdings", "named"),
    [
        pytest.param(
            _stale(HELD_OUTSIDE, "2026-09-18"),
            (Holding(ticker="QQQ", shares=5), Holding(ticker=HELD_OUTSIDE, shares=10)),
            "QQQ (no finite close), ZZZ (last close 2026-09-18)",
            id="fetched-missing-and-stale",
        ),
        pytest.param(
            _stale("T0", "2026-09-16"),
            (Holding(ticker="T0", shares=10),),
            "T0 (last close 2026-09-16)",
            id="stale-snapshot-ticker",
        ),
    ],
)
def test_a_missing_or_stale_price_fails_closed_before_orders(
    frame: Any, holdings: tuple[Holding, ...], named: str
) -> None:
    world = World(prices=FakeProvider(frame, SECTORS), broker=ScriptedBroker(holdings=holdings))
    with pytest.raises(ValueError, match=re.escape(named)):
        _run(world, options=DRY)
    [report] = world.report_store.saved
    assert (report.outcome, report.failed_stage) == (CycleOutcome.ERROR, "prices")
    assert report.verdict is not None
    assert (report.order_prices, report.plan) == (None, None)


@pytest.mark.parametrize(
    ("changes", "options", "stage", "message"),
    [
        pytest.param(
            {"shortlist": ShortlistParams(per_sector=0)},
            DRY,
            "shortlist",
            "per_sector must be >= 1",
            id="shortlist",
        ),
        pytest.param(
            {"conviction": ConvictionParams(min_valid=0)},
            DRY,
            "research",
            "min_valid must be an int >= 1",
            id="conviction",
        ),
        pytest.param({}, CycleOptions(dry_run=True, k=0), "research", "k must be", id="k"),
        pytest.param(
            {}, CycleOptions(dry_run=True, max_workers=0), "research", "max_workers", id="workers"
        ),
        pytest.param({"alpha": AlphaParams(ic=0.0)}, DRY, "portfolio", "ic must be", id="alpha"),
        pytest.param(
            {"risk": RiskParams(min_obs=1)}, DRY, "portfolio", "min_obs must satisfy", id="risk"
        ),
        pytest.param(
            {"correlation": CorrelationParams(max_rho=2.0)},
            DRY,
            "portfolio",
            "max_rho must satisfy",
            id="correlation",
        ),
        pytest.param(
            {"optimize": OptimizeParams(risk_aversion=0.0)},
            DRY,
            "portfolio",
            "risk_aversion must be > 0",
            id="optimize",
        ),
        pytest.param(
            {"limits": RiskLimits(gross=0.0)}, DRY, "risk", "gross must be > 0", id="limits"
        ),
        pytest.param(
            {"orders": OrderParams(no_trade_band=1.0)},
            DRY,
            "orders",
            "no_trade_band must satisfy",
            id="orders",
        ),
    ],
)
def test_each_param_set_and_option_reaches_its_stage(
    changes: dict[str, Any], options: CycleOptions, stage: str, message: str
) -> None:
    world = World(params=_params(**changes))
    with pytest.raises(ValueError, match=message):
        _run(world, options=options)
    [report] = world.report_store.saved
    assert (report.outcome, report.failed_stage) == (CycleOutcome.ERROR, stage)


def test_an_empty_universe_fails_in_the_data_stage() -> None:
    world = World(universe=universe({}))
    with pytest.raises(ValueError, match="no tickers requested"):
        _run(world, options=DRY)
    [report] = world.report_store.saved
    assert (report.outcome, report.failed_stage) == (CycleOutcome.ERROR, "data")


def test_breaker_params_reach_the_preflight() -> None:
    # 3% down trips the default 2.5% limit but not a 5% one, so the cycle goes on.
    world = World(snapshot_store=_halting_store(), params=_params(breakers=BreakerParams(0.05)))
    report = _run(world, options=DRY)
    assert report.outcome is CycleOutcome.DRY_RUN
    assert report.verdict is not None
    assert report.verdict.halt is False  # assess_portfolio got the same 5% breakers


def test_risk_params_and_nav_reach_the_portfolio_and_the_risk_check() -> None:
    # 295 rows: more than the ~283 the default lookback's calendar window would fetch.
    risk = RiskParams(lookback=295, min_obs=150)
    reads = [_read(AS_OF - timedelta(days=1), 1_020_000.0)]  # -2.0%: no halt
    world = World(snapshot_store=InMemorySnapshotStore(reads), params=_params(risk=risk))
    report = _run(world, options=DRY)
    provider = FakeProvider(FRAME, SECTORS)
    snapshot = build_snapshot(provider, list(SECTORS), AS_OF).snapshot
    benchmark = fetch_benchmark(provider, AS_OF, lookback_days=295)
    assert report.research is not None
    assert report.target == build_portfolio(report.research, snapshot, benchmark, risk=risk)
    nav = (NavPoint(AS_OF - timedelta(days=1), 1_020_000.0), NavPoint(AS_OF, EQUITY))
    expected = assess_portfolio(
        report.target, snapshot, benchmark, nav=nav, limits=LOOSE_LIMITS, risk=risk
    )
    assert report.verdict == expected
    assert report.verdict.breakers.daily_loss == pytest.approx(EQUITY / 1_020_000.0 - 1)


# --- step 6: research reuse --------------------------------------------------------------


def _stored(world: World) -> ResearchResult:
    """Run one dry cycle so the store holds today's research; forget the researcher calls."""
    research = _run(world, options=DRY).research
    assert research is not None
    world.researcher.calls.clear()
    return research


def test_matching_stored_research_is_reused_without_calling_the_researcher() -> None:
    world = World()
    stored = _stored(world)
    report = _run(world, options=DRY)
    assert world.researcher.calls == []
    assert report.research_reused is True
    assert report.research is stored
    assert world.research_store.saved == [stored]
    assert world.research_store.loaded == [AS_OF, AS_OF]


def _drop_last_name(result: ResearchResult) -> ResearchResult:
    return dataclasses.replace(result, names=result.names[:-1])


def _reverse_names(result: ResearchResult) -> ResearchResult:
    return dataclasses.replace(result, names=result.names[::-1])


def _with_error_run(result: ResearchResult) -> ResearchResult:
    first = result.names[0]
    failed = dataclasses.replace(first.runs[0], rating=None, error="RuntimeError: boom")
    name = dataclasses.replace(first, runs=(failed, *first.runs[1:]))
    return dataclasses.replace(result, names=(name, *result.names[1:]))


@pytest.mark.parametrize(
    ("change", "options"),
    [
        pytest.param(None, CycleOptions(dry_run=True, fresh_research=True), id="fresh-research"),
        pytest.param(None, CycleOptions(dry_run=True, k=2), id="k-mismatch"),
        pytest.param(_drop_last_name, DRY, id="changed-shortlist"),
        pytest.param(_reverse_names, DRY, id="reordered-shortlist"),
        pytest.param(_with_error_run, DRY, id="error-run"),
    ],
)
def test_research_reruns_and_is_saved_unless_it_matches(
    change: Callable[[ResearchResult], ResearchResult] | None, options: CycleOptions
) -> None:
    world = World()
    stored = _stored(world)
    if change is not None:
        world.research_store.saved = [change(stored)]
    report = _run(world, options=options)
    assert len(world.researcher.calls) == options.k * len(SECTORS)
    assert report.research_reused is False
    assert report.research is not None
    assert world.research_store.saved[-1] is report.research
    assert len(world.research_store.saved) == 2
    # The first cycle loaded (and found nothing); a fresh one does not even look.
    loads = [AS_OF] if options.fresh_research else [AS_OF, AS_OF]
    assert world.research_store.loaded == loads


# --- steps 11-13: submission window, approval, execution ----------------------------------

HELD = (Holding(ticker=HELD_OUTSIDE, shares=100),)  # sold first: a reducing leg


def _ids(plan: OrderPlan | None) -> list[str]:
    """``hf-<as_of>-<ticker>-<action>-<shares>`` per order, in plan (= submission) order."""
    assert plan is not None
    return [f"hf-{plan.as_of}-{o.ticker}-{o.action.value}-{o.shares}" for o in plan.orders]


@functools.cache
def _held_ids() -> tuple[str, ...]:
    """The client_order_ids of the plan for a book holding HELD (from one dry run)."""
    return tuple(_ids(_run(World(broker=ScriptedBroker(holdings=HELD)), options=DRY).plan))


class _SlowGate(ScriptedApprovalGate):
    """Approves, but the human answers only after the session window has opened."""

    def __init__(self, clock: Clock) -> None:
        super().__init__([True])
        self.clock = clock

    def confirm(self, summary: str) -> bool:
        self.clock.times = [SESSION]
        return super().confirm(summary)


def test_an_approved_plan_is_submitted() -> None:
    world = World()
    report = _run(world)
    assert report.outcome is CycleOutcome.SUBMITTED
    assert (report.failed_stage, report.error) == (None, None)
    assert report.decision is not None
    assert report.decision.status is ApprovalStatus.APPROVED
    assert len(world.gate.summaries) == 1
    assert world.broker.submitted == _ids(report.plan)
    assert all(client_id.startswith("hf-2026-09-30-") for client_id in world.broker.submitted)
    assert report.execution is not None
    assert [result.status for result in report.execution.results] == [SubmitStatus.ACCEPTED] * 6
    assert (report.execution_interrupted, report.planned_client_order_ids) == (False, ())
    assert world.snapshot_store.saved == [report.account]
    assert world.report_store.saved == [report]


@pytest.mark.parametrize(
    ("world", "outcome", "status"),
    [
        pytest.param(
            World(gate=ScriptedApprovalGate([False])),
            CycleOutcome.REJECTED,
            ApprovalStatus.REJECTED,
            id="rejected",
        ),
        pytest.param(
            World(params=_params(limits=DEFAULT_LIMITS)),  # the seeded book breaks name_risk
            CycleOutcome.BLOCKED,
            ApprovalStatus.BLOCKED,
            id="blocked-by-the-veto",
        ),
        pytest.param(
            World(researcher=ScriptedResearcher()),  # every run Hold: NEUTRAL, an empty book
            CycleOutcome.NOTHING_TO_DO,
            ApprovalStatus.NOTHING_TO_DO,
            id="nothing-to-do",
        ),
    ],
)
def test_a_plan_that_is_not_approved_is_never_submitted(
    world: World, outcome: CycleOutcome, status: ApprovalStatus
) -> None:
    report = _run(world)
    assert report.outcome is outcome
    assert report.decision is not None
    assert report.decision.status is status
    assert world.broker.submitted == []
    assert report.execution is None
    assert len(world.gate.summaries) == (1 if status is ApprovalStatus.REJECTED else 0)


def test_inside_the_session_window_the_gate_is_never_called() -> None:
    world = World()
    # started_at and the account read at 18:00 EDT; the window check reads 10:00 EDT.
    with pytest.raises(SubmissionWindowError, match="opg orders must be submitted before 09:25"):
        _run(world, now=Clock(EVENING, EVENING, SESSION))
    [report] = world.report_store.saved
    assert (report.outcome, report.failed_stage) == (CycleOutcome.ERROR, "submission_window")
    assert report.error is not None
    assert report.error.startswith("SubmissionWindowError: Wed 2026-09-30 10:00 ET")
    assert report.plan is not None
    assert len(report.plan.orders) == 6
    assert report.decision is None
    assert world.gate.summaries == []
    assert world.broker.submitted == []


def test_the_window_is_checked_again_after_a_slow_approval() -> None:
    clock = Clock(EVENING)
    world = World(gate=_SlowGate(clock))
    with pytest.raises(SubmissionWindowError, match="opg"):
        _run(world, now=clock)
    [report] = world.report_store.saved
    assert (report.outcome, report.failed_stage) == (CycleOutcome.ERROR, "submission_window")
    assert report.decision is not None
    assert report.decision.status is ApprovalStatus.APPROVED
    assert len(world.gate.summaries) == 1
    assert world.broker.submitted == []
    assert (report.execution, report.execution_interrupted) == (None, False)


@pytest.mark.parametrize(
    ("now", "options", "outcome"),
    [
        pytest.param(SESSION, CycleOptions(any_time=True), CycleOutcome.SUBMITTED, id="any-time"),
        pytest.param(SESSION, DRY, CycleOutcome.DRY_RUN, id="dry-run"),
        pytest.param(WEEKEND, CycleOptions(), CycleOutcome.SUBMITTED, id="weekend"),
        pytest.param(EVENING, CycleOptions(), CycleOutcome.SUBMITTED, id="weekday-evening"),
    ],
)
def test_the_window_check_is_skipped_or_passes(
    now: datetime, options: CycleOptions, outcome: CycleOutcome
) -> None:
    world = World()
    # AS_OF is only live before TODAY's open: on the weekend clock the cycle is dated the weekend.
    as_of, today = (now.date(),) * 2 if now == WEEKEND else (AS_OF, TODAY)
    report = _run(world, options=options, as_of=as_of, today=today, now=Clock(now))
    assert report.outcome is outcome


def test_failed_legs_still_end_submitted_with_the_failures_reported() -> None:
    first = _held_ids()[0]
    world = World(broker=ScriptedBroker({first: SubmitStatus.REJECTED}, holdings=HELD))
    report = _run(world)
    assert report.outcome is CycleOutcome.SUBMITTED
    assert report.execution is not None
    assert report.execution.results[0].status is SubmitStatus.REJECTED
    assert report.execution.opening_halted is True
    assert world.broker.submitted == [first]


@pytest.mark.parametrize(
    ("raised", "outcome"),
    [
        pytest.param(RuntimeError("socket closed"), CycleOutcome.ERROR, id="exception"),
        pytest.param(KeyboardInterrupt(), CycleOutcome.INTERRUPTED, id="interrupt"),
    ],
)
def test_an_execution_that_raises_records_the_planned_ids(
    raised: BaseException, outcome: CycleOutcome
) -> None:
    ids = _held_ids()
    world = World(broker=ScriptedBroker({ids[0]: raised}, holdings=HELD))
    with pytest.raises(type(raised)):
        _run(world)
    [report] = world.report_store.saved
    assert (report.outcome, report.failed_stage) == (outcome, "execution")
    assert report.decision is not None
    assert report.decision.status is ApprovalStatus.APPROVED
    assert (report.execution, report.execution_interrupted) == (None, True)
    assert report.planned_client_order_ids == ids
    assert ids[0].startswith(f"hf-2026-09-30-{HELD_OUTSIDE}-sell-100")  # reducing leg first
    assert world.broker.submitted == [ids[0]]


def test_an_unpreparable_plan_records_no_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    def malformed(plan: OrderPlan) -> NoReturn:
        raise ValueError("duplicate client_order_id")

    monkeypatch.setattr(cycle, "prepare_submission", malformed)
    world = World(broker=ScriptedBroker({_held_ids()[0]: RuntimeError("down")}, holdings=HELD))
    with pytest.raises(RuntimeError, match="down"):
        _run(world)
    [report] = world.report_store.saved
    assert (report.execution_interrupted, report.planned_client_order_ids) == (True, ())


def test_a_failing_gate_is_an_approval_error() -> None:
    world = World(gate=ScriptedApprovalGate([]))  # raises: out of answers
    with pytest.raises(AssertionError, match="out of answers"):
        _run(world)
    [report] = world.report_store.saved
    assert (report.outcome, report.failed_stage) == (CycleOutcome.ERROR, "approval")
    assert report.decision is None


# --- orders an earlier cycle left open at the broker ---------------------------------------

PRIOR_AS_OF = date(2026, 9, 29)  # Tuesday
PRIOR_AT = datetime(2026, 9, 29, 22, 0, tzinfo=UTC)  # Tue 18:00 EDT: its deadline is Wed 16:00 ET
MORNING = datetime(2026, 9, 30, 12, 0, tzinfo=UTC)  # Wed 08:00 EDT: before the open and deadline


def _status(client_order_id: str, state: OrderState) -> OrderStatus:
    return dataclasses.replace(not_found(client_order_id), state=state)


def _earlier_cycle(
    world: World, *, as_of: date = PRIOR_AS_OF, at: datetime = PRIOR_AT
) -> list[str]:
    """Submit a full cycle for ``as_of`` at ``at`` (the broker still holds nothing), give the
    world a fresh gate and return the submitted ids.
    """
    report = _run(world, as_of=as_of, today=AS_OF, now=Clock(at))
    assert report.outcome is CycleOutcome.SUBMITTED
    ids = list(world.broker.submitted)
    world.broker.submitted.clear()
    world.gate = ScriptedApprovalGate([True])
    return ids


def _open_at_broker(world: World, ids: list[str], state: OrderState = OrderState.PENDING) -> None:
    world.broker.orders = {cid: _status(cid, state) for cid in ids}


def _interrupted(world: World) -> None:
    """Turn the stored report into an interrupted execution: its planned ids, no results."""
    [report] = world.report_store.saved
    world.report_store.saved[:] = [
        dataclasses.replace(
            report,
            outcome=CycleOutcome.ERROR,
            execution=None,
            execution_interrupted=True,
            planned_client_order_ids=tuple(_ids(report.plan)),
        )
    ]


def _refused(world: World, *, now: datetime = MORNING) -> PendingOrdersError:
    with pytest.raises(PendingOrdersError) as raised:
        _run(world, now=Clock(now))
    assert world.gate.summaries == []
    assert world.broker.submitted == []
    report = world.report_store.saved[-1]
    assert (report.outcome, report.failed_stage) == (CycleOutcome.ERROR, "pending_orders")
    assert report.decision is None
    return raised.value


def test_orders_still_queued_from_the_evening_cycle_block_the_morning_one() -> None:
    world = World()
    ids = _earlier_cycle(world)  # Tuesday evening: six opg orders queued, positions still flat
    assert len(ids) == 6
    assert world.broker.holdings == ()
    _open_at_broker(world, ids)
    message = str(_refused(world))
    assert message.startswith("6 order(s) from an earlier cycle are still open at the broker: ")
    assert message.endswith("; settle or wait for the open before running another cycle")
    listed = sorted(ids)
    assert message.index(f"{listed[0]} (as_of {PRIOR_AS_OF})") < message.index(listed[-1])
    assert sorted(world.broker.looked_up) == listed


@pytest.mark.parametrize(
    "state",
    [
        OrderState.FILLED,
        OrderState.PARTIAL_TERMINAL,
        OrderState.CANCELED,
        OrderState.REJECTED,
        OrderState.EXPIRED,
        OrderState.NOT_FOUND,  # the leg was acknowledged: settle's concern, not an open order
    ],
)
def test_earlier_orders_that_are_final_do_not_block(state: OrderState) -> None:
    world = World()
    ids = _earlier_cycle(world)
    _open_at_broker(world, ids, state)
    assert _run(world, now=Clock(MORNING)).outcome is CycleOutcome.SUBMITTED
    assert sorted(world.broker.looked_up) == sorted(ids)


def test_an_earlier_report_past_its_fill_deadline_is_not_looked_up() -> None:
    world = World()
    ids = _earlier_cycle(world)
    _open_at_broker(world, ids)
    assert _run(world, now=Clock(EVENING)).outcome is CycleOutcome.SUBMITTED  # Wed 18:00 EDT
    assert world.broker.looked_up == []


def test_an_earlier_dry_run_is_not_looked_up() -> None:
    world = World()
    ids = _earlier_cycle(world)
    _open_at_broker(world, ids)
    [report] = world.report_store.saved
    world.report_store.saved[:] = [dataclasses.replace(report, dry_run=True)]
    assert _run(world, now=Clock(MORNING)).outcome is CycleOutcome.SUBMITTED
    assert world.broker.looked_up == []


@pytest.mark.parametrize("interrupted", [False, True])
def test_a_rerun_of_the_same_as_of_keeps_the_duplicate_path(interrupted: bool) -> None:
    world = World()
    ids = _earlier_cycle(world, as_of=AS_OF, at=EVENING)
    _open_at_broker(world, ids)  # pending at the broker, but they are the new plan's own ids
    if interrupted:
        _interrupted(world)
    report = _run(world, now=Clock(EVENING + timedelta(minutes=5)))
    assert report.outcome is CycleOutcome.SUBMITTED
    assert world.broker.submitted == ids
    assert world.broker.looked_up == []


def test_only_ids_outside_the_new_plan_are_looked_up_and_listed() -> None:
    world = World()
    other = _earlier_cycle(world)
    own = _earlier_cycle(world, as_of=AS_OF, at=MORNING)
    assert not set(own) & set(other)
    _open_at_broker(world, [*own, *other])
    world.broker.looked_up.clear()  # the second earlier cycle looked the first one up
    message = str(_refused(world, now=MORNING + timedelta(minutes=5)))
    assert message.startswith("6 order(s) from an earlier cycle are still open")
    assert sorted(world.broker.looked_up) == sorted(other)
    assert not any(cid in message for cid in own)


@pytest.mark.parametrize(
    "answer",
    [
        pytest.param(OrderState.UNRECOGNISED, id="unrecognised"),
        pytest.param(OrderDataError("garbled body"), id="malformed-answer"),
    ],
)
def test_an_answer_that_cannot_be_trusted_fails_closed(answer: OrderState | Exception) -> None:
    world = World()
    ids = _earlier_cycle(world)
    _open_at_broker(world, ids, OrderState.FILLED)
    first_id = sorted(ids)[0]
    world.broker.orders[first_id] = (
        answer if isinstance(answer, Exception) else _status(first_id, answer)
    )
    message = str(_refused(world))
    assert message.startswith("1 order(s) from an earlier cycle are still open")
    assert first_id in message


def test_an_unacknowledged_unknown_order_blocks_until_its_fill_deadline() -> None:
    world = World()
    _earlier_cycle(world)  # the broker answers NOT_FOUND for every id
    _interrupted(world)  # ... and the report never saw it acknowledged
    assert str(_refused(world)).startswith("6 order(s) from an earlier cycle are still open")
    assert _run(world, now=Clock(EVENING)).outcome is CycleOutcome.SUBMITTED  # Wed after 16:00 ET


def test_a_dry_run_ignores_open_orders() -> None:
    world = World()
    ids = _earlier_cycle(world)
    _open_at_broker(world, ids)
    assert _run(world, options=DRY, now=Clock(MORNING)).outcome is CycleOutcome.DRY_RUN
    assert world.broker.looked_up == []


@pytest.mark.parametrize(
    ("changes", "outcome"),
    [
        pytest.param(
            {"limits": DEFAULT_LIMITS}, CycleOutcome.BLOCKED, id="blocked-by-the-veto"
        ),  # the seeded book breaks name_risk
        pytest.param({"researcher": ScriptedResearcher()}, CycleOutcome.NOTHING_TO_DO, id="empty"),
    ],
)
def test_a_plan_that_would_submit_nothing_is_not_checked(
    changes: dict[str, Any], outcome: CycleOutcome
) -> None:
    world = World()
    ids = _earlier_cycle(world)
    _open_at_broker(world, ids)
    if "limits" in changes:
        world.params = _params(**changes)
    else:
        world.researcher = changes["researcher"]
    assert _run(world, now=Clock(MORNING)).outcome is outcome
    assert world.broker.looked_up == []
