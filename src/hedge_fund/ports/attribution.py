"""Attribution port: where attribution reports are kept."""

from __future__ import annotations

from typing import Protocol

from hedge_fund.domain.attribution import AttributionReport


class AttributionReportStore(Protocol):
    """Attribution report storage. Implementations live in adapters/attribution/."""

    def save(self, report: AttributionReport) -> None:
        """Persist ``report``; never overwrites a stored one. Raises (any Exception) on failure."""
