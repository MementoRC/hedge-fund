"""Cycle report port: keep one audit report per pipeline cycle and read them all back."""

from __future__ import annotations

from typing import Protocol

from hedge_fund.domain.cycle import CycleReport
from hedge_fund.domain.ledger import ReportLoad


class CycleReportStore(Protocol):
    """Cycle report storage. Implementations live in adapters/cycle/."""

    def save(self, report: CycleReport) -> None:
        """Persist ``report``; never overwrites a stored one. Raises (any Exception) on failure."""

    def load_all(self) -> ReportLoad:
        """Every stored report that decodes, sorted by ``started_at``, plus the stored files
        that do not (name -> ``"<ExceptionType>: <message>"``); one bad file never raises.
        """
