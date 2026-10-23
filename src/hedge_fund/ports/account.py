"""Account snapshot port: keep every dated broker read and hand them all back."""

from __future__ import annotations

from typing import Protocol

from hedge_fund.domain.account import AccountSnapshot


class SnapshotStoreError(Exception):
    """A stored snapshot that cannot be read back: corrupt, tampered with or mistyped."""


class SnapshotStore(Protocol):
    """Account snapshot storage. Implementations live in adapters/account/."""

    def save(self, snapshot: AccountSnapshot) -> None:
        """Persist ``snapshot``; never overwrites or drops a stored one."""

    def snapshots(self) -> tuple[AccountSnapshot, ...]:
        """Every stored snapshot; raises SnapshotStoreError if any cannot be read back."""
