"""Tests for the SnapshotStore port."""

from __future__ import annotations

from hedge_fund.ports.account import SnapshotStoreError


def test_snapshot_store_error_is_an_exception() -> None:
    assert issubclass(SnapshotStoreError, Exception)
