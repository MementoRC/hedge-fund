"""Verification port: where verification reports are kept."""

from __future__ import annotations

from typing import Protocol

from hedge_fund.domain.verification import VerificationReport


class VerificationReportStore(Protocol):
    """Verification report storage. Implementations live in adapters/verification/."""

    def save(self, report: VerificationReport) -> None:
        """Persist ``report``; never overwrites a stored one (FileExistsError). Raises (any
        Exception) on failure."""
