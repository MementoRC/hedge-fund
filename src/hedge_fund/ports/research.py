"""Researcher port: one independent research run on a ticker."""

from __future__ import annotations

from datetime import date
from typing import Protocol

from hedge_fund.domain.research import RunRecord


class Researcher(Protocol):
    """Source of research ratings. Implementations live in adapters/research/."""

    def run(self, ticker: str, as_of: date, run_index: int) -> RunRecord:
        """Run one full research pass on ``ticker`` as of ``as_of``.

        ``run_index`` labels the repetition and has no other effect. May raise on any
        failure; the research service turns exceptions into error records.
        """
