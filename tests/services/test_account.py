"""Layer 6d account service: record broker reads, fold them into NAV, trip the breakers."""

from __future__ import annotations

import math
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta, timezone
from pathlib import Path

import pytest

from hedge_fund.adapters.account import JsonSnapshotStore
from hedge_fund.domain.account import AccountSnapshot
from hedge_fund.domain.execution import Account
from hedge_fund.domain.orders import Holding
from hedge_fund.domain.portfolio import TargetPortfolio
from hedge_fund.domain.risk import NavPoint
from hedge_fund.ports.account import SnapshotStore, SnapshotStoreError
from hedge_fund.ports.broker import BrokerError
from hedge_fund.services.account import load_nav_history, read_snapshot, record_snapshot
from hedge_fund.services.risk import assess_portfolio
from tests.factories import make_closes, snapshot_from
from tests.fakes import InMemorySnapshotStore, ScriptedBroker

AS_OF = date(2026, 10, 23)
OBSERVED_AT = datetime(2026, 10, 23, 13, 0, tzinfo=UTC)
HOLDINGS = (Holding(ticker="AAPL", shares=40), Holding(ticker="XOM", shares=-25))


def _at(moment: datetime) -> Callable[[], datetime]:
    return lambda: moment


# --- record_snapshot --------------------------------------------------------------------


def test_records_equity_and_positions_as_one_snapshot() -> None:
    broker = ScriptedBroker(equity=250_000.0, holdings=HOLDINGS)
    store = InMemorySnapshotStore()
    snapshot = record_snapshot(broker, store, as_of=AS_OF, now=_at(OBSERVED_AT))
    assert snapshot == AccountSnapshot(
        as_of=AS_OF, observed_at=OBSERVED_AT, equity=250_000.0, holdings=HOLDINGS
    )
    assert store.saved == [snapshot]
    assert broker.submitted == []


def test_default_clock_is_utc_now() -> None:
    before = datetime.now(UTC)
    snapshot = record_snapshot(ScriptedBroker(), InMemorySnapshotStore(), as_of=AS_OF)
    assert before <= snapshot.observed_at <= datetime.now(UTC)
    assert snapshot.observed_at.utcoffset() == timedelta(0)


def test_as_of_need_not_match_the_date_of_the_read() -> None:
    # 22:00 on the 22nd at UTC-4 is 02:00 UTC on the 23rd: neither date is checked.
    evening = datetime(2026, 10, 22, 22, 0, tzinfo=timezone(timedelta(hours=-4)))
    snapshot = record_snapshot(
        ScriptedBroker(), InMemorySnapshotStore(), as_of=AS_OF, now=_at(evening)
    )
    assert snapshot.observed_at == evening


@pytest.mark.parametrize(
    ("broker", "now", "message"),
    [
        pytest.param(ScriptedBroker(equity=0.0), OBSERVED_AT, "nav must be", id="zero-equity"),
        pytest.param(ScriptedBroker(equity=math.nan), OBSERVED_AT, "nav must be", id="nan-equity"),
        pytest.param(
            ScriptedBroker(holdings=tuple(reversed(HOLDINGS))),
            OBSERVED_AT,
            "strictly ascending",
            id="unsorted-holdings",
        ),
        pytest.param(ScriptedBroker(), datetime(2026, 10, 23, 13, 0), "tz-aware", id="naive-clock"),
    ],
)
def test_an_invalid_read_is_never_saved(
    broker: ScriptedBroker, now: datetime, message: str
) -> None:
    store = InMemorySnapshotStore()
    with pytest.raises(ValueError, match=message):
        record_snapshot(broker, store, as_of=AS_OF, now=_at(now))
    assert store.saved == []


@pytest.mark.parametrize("failing", ["account_error", "positions_error"])
def test_a_broker_error_saves_nothing(failing: str) -> None:
    broker = ScriptedBroker(**{failing: BrokerError(f"{failing} boom")})
    store = InMemorySnapshotStore()
    with pytest.raises(BrokerError, match=f"{failing} boom"):
        record_snapshot(broker, store, as_of=AS_OF, now=_at(OBSERVED_AT))
    assert store.saved == []


def test_the_account_is_read_before_the_positions() -> None:
    broker = ScriptedBroker(
        account_error=BrokerError("account first"), positions_error=BrokerError("positions")
    )
    with pytest.raises(BrokerError, match="account first"):
        record_snapshot(broker, InMemorySnapshotStore(), as_of=AS_OF, now=_at(OBSERVED_AT))


def test_the_clock_is_read_after_both_broker_reads() -> None:
    events: list[str] = []

    class RecordingBroker(ScriptedBroker):
        def account(self) -> Account:
            events.append("account")
            return super().account()

        def positions(self) -> tuple[Holding, ...]:
            events.append("positions")
            return super().positions()

    def clock() -> datetime:
        events.append("now")
        return OBSERVED_AT

    record_snapshot(RecordingBroker(), InMemorySnapshotStore(), as_of=AS_OF, now=clock)
    assert events == ["account", "positions", "now"]


# --- load_nav_history -------------------------------------------------------------------


def test_load_folds_the_stored_snapshots() -> None:
    reads = [
        AccountSnapshot(AS_OF, OBSERVED_AT, 97.0, ()),
        AccountSnapshot(AS_OF - timedelta(days=1), OBSERVED_AT - timedelta(days=1), 100.0, ()),
        AccountSnapshot(AS_OF + timedelta(days=1), OBSERVED_AT + timedelta(days=1), 1.0, ()),
    ]
    assert load_nav_history(InMemorySnapshotStore(reads), as_of=AS_OF) == (
        NavPoint(date=AS_OF - timedelta(days=1), nav=100.0),
        NavPoint(date=AS_OF, nav=97.0),
    )


def test_load_propagates_a_store_failure() -> None:
    store = InMemorySnapshotStore(fail_on_read=SnapshotStoreError("x.json: digest mismatch"))
    with pytest.raises(SnapshotStoreError, match="digest mismatch"):
        load_nav_history(store, as_of=AS_OF)


@pytest.mark.parametrize(
    ("snapshot", "message"),
    [
        pytest.param(AccountSnapshot(AS_OF, OBSERVED_AT, 0.0, ()), "nav must be", id="zero-equity"),
        pytest.param(
            AccountSnapshot(AS_OF, OBSERVED_AT, 100.0, tuple(reversed(HOLDINGS))),
            "strictly ascending",
            id="unsorted-holdings",
        ),
    ],
)
def test_load_fails_closed_on_an_invalid_stored_snapshot(
    snapshot: AccountSnapshot, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        load_nav_history(InMemorySnapshotStore([snapshot]), as_of=AS_OF)


def test_load_of_an_empty_store_is_an_empty_history() -> None:
    assert load_nav_history(InMemorySnapshotStore(), as_of=AS_OF) == ()


# --- end to end with assess_portfolio ---------------------------------------------------

SECTORS = {"AAA": "Tech", "BBB": "Health"}
PRICES = snapshot_from(make_closes(list(SECTORS), seed=3), SECTORS)
SPY = make_closes(["SPY"], seed=11)["SPY"]
RISK_AS_OF = PRICES.as_of  # 2026-09-30
EMPTY_BOOK = TargetPortfolio(
    as_of=RISK_AS_OF, positions=(), excluded={}, gross=0.0, net=0.0, beta=0.0, expected_vol=0.0
)


def _record_runs(store: SnapshotStore, equities: list[float]) -> None:
    """One run per day, the last on RISK_AS_OF, the broker reporting each equity in turn."""
    broker = ScriptedBroker()
    first = RISK_AS_OF - timedelta(days=len(equities) - 1)
    for offset, equity in enumerate(equities):
        day = first + timedelta(days=offset)
        broker.equity = equity
        moment = datetime(day.year, day.month, day.day, 13, 0, tzinfo=UTC)
        record_snapshot(broker, store, as_of=day, now=_at(moment))


def test_a_three_percent_day_trips_the_daily_breaker() -> None:
    store = InMemorySnapshotStore()
    _record_runs(store, [100.0, 97.0])
    verdict = assess_portfolio(
        EMPTY_BOOK, PRICES, SPY, nav=load_nav_history(store, as_of=RISK_AS_OF)
    )
    assert verdict.breakers.daily_tripped is True
    assert verdict.breakers.drawdown_tripped is False
    assert (verdict.approved, verdict.halt) == (False, True)


def test_a_fall_of_exactly_eight_percent_from_the_peak_trips_drawdown() -> None:
    # 1 - 92 / 100 is exactly 8% below the peak; the last day is +1.1%, so only drawdown trips.
    store = InMemorySnapshotStore()
    _record_runs(store, [100.0, 91.0, 92.0])
    verdict = assess_portfolio(
        EMPTY_BOOK, PRICES, SPY, nav=load_nav_history(store, as_of=RISK_AS_OF)
    )
    assert verdict.breakers.drawdown == pytest.approx(0.08)
    assert verdict.breakers.drawdown_tripped is True
    assert verdict.breakers.daily_tripped is False
    assert verdict.halt is True


def test_breakers_trip_through_the_json_store(tmp_path: Path) -> None:
    store = JsonSnapshotStore(tmp_path)
    _record_runs(store, [100.0, 97.0])
    nav = load_nav_history(JsonSnapshotStore(tmp_path), as_of=RISK_AS_OF)
    assert nav == (NavPoint(RISK_AS_OF - timedelta(days=1), 100.0), NavPoint(RISK_AS_OF, 97.0))
    assert assess_portfolio(EMPTY_BOOK, PRICES, SPY, nav=nav).breakers.daily_tripped is True


def test_a_quiet_history_does_not_halt() -> None:
    store = InMemorySnapshotStore()
    _record_runs(store, [100.0, 99.0, 98.5])
    verdict = assess_portfolio(
        EMPTY_BOOK, PRICES, SPY, nav=load_nav_history(store, as_of=RISK_AS_OF)
    )
    assert (verdict.approved, verdict.halt) == (True, False)


# --- read_snapshot ----------------------------------------------------------------------


def test_read_returns_the_validated_snapshot() -> None:
    broker = ScriptedBroker(equity=250_000.0, holdings=HOLDINGS)
    assert read_snapshot(broker, as_of=AS_OF, now=_at(OBSERVED_AT)) == AccountSnapshot(
        as_of=AS_OF, observed_at=OBSERVED_AT, equity=250_000.0, holdings=HOLDINGS
    )
    assert broker.submitted == []


def test_read_default_clock_is_utc_now() -> None:
    before = datetime.now(UTC)
    snapshot = read_snapshot(ScriptedBroker(), as_of=AS_OF)
    assert before <= snapshot.observed_at <= datetime.now(UTC)
    assert snapshot.observed_at.utcoffset() == timedelta(0)


def test_read_takes_account_then_positions_then_the_clock() -> None:
    events: list[str] = []

    class RecordingBroker(ScriptedBroker):
        def account(self) -> Account:
            events.append("account")
            return super().account()

        def positions(self) -> tuple[Holding, ...]:
            events.append("positions")
            return super().positions()

    def clock() -> datetime:
        events.append("now")
        return OBSERVED_AT

    read_snapshot(RecordingBroker(), as_of=AS_OF, now=clock)
    assert events == ["account", "positions", "now"]


@pytest.mark.parametrize(
    ("broker", "now", "message"),
    [
        pytest.param(ScriptedBroker(equity=0.0), OBSERVED_AT, "nav must be", id="zero-equity"),
        pytest.param(
            ScriptedBroker(holdings=tuple(reversed(HOLDINGS))),
            OBSERVED_AT,
            "strictly ascending",
            id="unsorted-holdings",
        ),
        pytest.param(ScriptedBroker(), datetime(2026, 10, 23, 13, 0), "tz-aware", id="naive-clock"),
    ],
)
def test_read_rejects_an_invalid_read(broker: ScriptedBroker, now: datetime, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        read_snapshot(broker, as_of=AS_OF, now=_at(now))


@pytest.mark.parametrize("failing", ["account_error", "positions_error"])
def test_read_propagates_a_broker_error(failing: str) -> None:
    broker = ScriptedBroker(**{failing: BrokerError(f"{failing} boom")})
    with pytest.raises(BrokerError, match=f"{failing} boom"):
        read_snapshot(broker, as_of=AS_OF, now=_at(OBSERVED_AT))
