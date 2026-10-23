"""Tests for the account domain type."""

from __future__ import annotations

import dataclasses
from datetime import UTC, date, datetime

import pytest

from hedge_fund.domain.account import AccountSnapshot
from hedge_fund.domain.orders import Holding

SNAPSHOT = AccountSnapshot(
    as_of=date(2026, 10, 23),
    observed_at=datetime(2026, 10, 23, 13, 0, tzinfo=UTC),
    equity=1_000_000.0,
    holdings=(Holding(ticker="AAPL", shares=40), Holding(ticker="XOM", shares=-25)),
)


def test_snapshot_compares_by_value() -> None:
    copy = AccountSnapshot(
        as_of=date(2026, 10, 23),
        observed_at=datetime(2026, 10, 23, 13, 0, tzinfo=UTC),
        equity=1_000_000.0,
        holdings=(Holding(ticker="AAPL", shares=40), Holding(ticker="XOM", shares=-25)),
    )
    assert copy == SNAPSHOT
    assert dataclasses.replace(SNAPSHOT, equity=999_999.0) != SNAPSHOT


def test_snapshot_is_frozen() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        SNAPSHOT.equity = 0.0
