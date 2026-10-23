"""Tests for the SnapshotStore port, exercised through the in-memory fake."""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest

from hedge_fund.domain.account import AccountSnapshot
from hedge_fund.domain.orders import Holding
from hedge_fund.ports.account import SnapshotStore, SnapshotStoreError
from tests.fakes import InMemorySnapshotStore

FIRST = AccountSnapshot(
    as_of=date(2026, 10, 22),
    observed_at=datetime(2026, 10, 22, 13, 0, tzinfo=UTC),
    equity=100.0,
    holdings=(Holding(ticker="AAPL", shares=40),),
)
SECOND = AccountSnapshot(
    as_of=date(2026, 10, 23),
    observed_at=datetime(2026, 10, 23, 13, 0, tzinfo=UTC),
    equity=97.0,
    holdings=(),
)


def test_snapshot_store_error_is_an_exception() -> None:
    assert issubclass(SnapshotStoreError, Exception)


def test_in_memory_store_satisfies_the_port() -> None:
    fake = InMemorySnapshotStore([FIRST])
    store: SnapshotStore = fake
    store.save(SECOND)
    assert store.snapshots() == (FIRST, SECOND)
    assert fake.saved == [FIRST, SECOND]


def test_in_memory_store_starts_empty() -> None:
    assert InMemorySnapshotStore().snapshots() == ()


def test_in_memory_store_raises_on_read_when_told_to() -> None:
    fake = InMemorySnapshotStore([FIRST], fail_on_read=SnapshotStoreError("x.json: corrupt"))
    with pytest.raises(SnapshotStoreError, match="x.json: corrupt"):
        fake.snapshots()
    fake.save(SECOND)
    assert fake.saved == [FIRST, SECOND]
