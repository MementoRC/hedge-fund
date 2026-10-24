"""Layer 6d pipeline cycle: one paper-trading run over fakes, and its audit report."""

from __future__ import annotations

import dataclasses
from datetime import UTC, date, datetime, timedelta

import pytest

from hedge_fund.domain.account import AccountSnapshot
from hedge_fund.domain.cycle import CycleOptions, CycleOutcome, CycleReport
from hedge_fund.engine.risk.breakers import BreakerParams
from hedge_fund.ports.account import SnapshotStoreError
from hedge_fund.ports.broker import BrokerError
from hedge_fund.services.cycle import ReportSaveError, run_cycle
from tests.cycle_world import AS_OF, EVENING, TODAY, Clock, World
from tests.fakes import InMemoryReportStore, InMemorySnapshotStore, ScriptedBroker

EQUITY = 1_000_000.0


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
