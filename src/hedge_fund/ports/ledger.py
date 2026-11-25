"""Ledger ports: write-once fill records, the opening balance and settlement reports."""

from __future__ import annotations

from typing import Protocol

from hedge_fund.domain.ledger import Fill, OpeningEntry, SettlementReport


class FillStore(Protocol):
    """Write-once fill records, one per client_order_id. Implementations live in adapters/ledger/."""

    def save(self, fill: Fill) -> None:
        """Record ``fill``; FileExistsError, keeping the first record, if its id is recorded."""

    def load_all(self) -> tuple[Fill, ...]:
        """Every recorded fill. Raises ValueError naming a malformed record."""


class OpeningBalance(Protocol):
    """Holdings before the system's first fill. Implementations live in adapters/ledger/."""

    def load(self) -> tuple[OpeningEntry, ...]:
        """The entries, ``()`` when none are recorded. Raises ValueError when malformed."""


class SettlementReportStore(Protocol):
    """Settlement report storage. Implementations live in adapters/ledger/."""

    def save(self, report: SettlementReport) -> None:
        """Persist ``report``; never overwrites a stored one. Raises (any Exception) on failure."""
