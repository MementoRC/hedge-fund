"""Layer 6d pipeline cycle: one paper-trading run over fakes, and its audit report."""

from __future__ import annotations

import dataclasses
import math
import re
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from typing import Any

import pytest

from hedge_fund.domain.account import AccountSnapshot
from hedge_fund.domain.cycle import CycleOptions, CycleOutcome, CycleReport
from hedge_fund.domain.orders import Holding, OrderAction
from hedge_fund.domain.research import ResearchResult, Verdict
from hedge_fund.domain.risk import NavPoint
from hedge_fund.engine.orders import OrderParams
from hedge_fund.engine.portfolio.alpha import AlphaParams
from hedge_fund.engine.portfolio.correlation import CorrelationParams
from hedge_fund.engine.portfolio.optimize import OptimizeParams
from hedge_fund.engine.portfolio.risk_model import RiskParams
from hedge_fund.engine.research.conviction import ConvictionParams
from hedge_fund.engine.risk.breakers import BreakerParams
from hedge_fund.engine.risk.limits import RiskLimits
from hedge_fund.engine.shortlist.select import ShortlistParams
from hedge_fund.ports.account import SnapshotStoreError
from hedge_fund.ports.broker import BrokerError
from hedge_fund.services.cycle import ReportSaveError, run_cycle
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
    TODAY,
    Clock,
    World,
    universe,
)
from tests.fakes import FakeProvider, InMemoryReportStore, InMemorySnapshotStore, ScriptedBroker

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
