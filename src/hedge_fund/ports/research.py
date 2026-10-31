"""Research ports: one independent research run on a ticker, and stored research results."""

from __future__ import annotations

from datetime import date
from typing import Protocol

from hedge_fund.domain.research import ResearchResult, RunRecord


class Researcher(Protocol):
    """Source of research ratings. Implementations live in adapters/research/."""

    def run(self, ticker: str, as_of: date, run_index: int) -> RunRecord:
        """Run one full research pass on ``ticker`` as of ``as_of``.

        ``run_index`` labels the repetition and has no other effect. May raise on any
        failure; the research service turns exceptions into error records.
        """


class ResearchStore(Protocol):
    """Research result storage. Implementations live in adapters/research/."""

    def save(self, result: ResearchResult) -> None:
        """Persist ``result``; never overwrites or drops a stored one."""

    def load(self, as_of: date) -> ResearchResult | None:
        """The newest stored result for ``as_of``, or None when there is none.

        Raises ValueError when a stored file for ``as_of`` is malformed.
        """

    def dates(self) -> tuple[date, ...]:
        """The dates of the stored results, sorted and without duplicates; ``()`` when none.

        Never raises for a malformed stored file: that surfaces in ``load``.
        """
