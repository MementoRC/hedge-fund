"""Layer 6d account service: record broker reads, fold them into NAV, trip the breakers."""

from __future__ import annotations

import math
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta, timezone

import pytest

from hedge_fund.domain.account import AccountSnapshot
from hedge_fund.domain.orders import Holding
from hedge_fund.ports.broker import BrokerError
from hedge_fund.services.account import record_snapshot
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
