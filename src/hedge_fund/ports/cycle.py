"""Cycle report port: keep one audit report per pipeline cycle."""

from __future__ import annotations

from typing import Protocol

from hedge_fund.domain.cycle import CycleReport


class CycleReportStore(Protocol):
    """Cycle report storage. Implementations live in adapters/cycle/."""

    def save(self, report: CycleReport) -> None:
        """Persist ``report``; never overwrites a stored one. Raises (any Exception) on failure."""
