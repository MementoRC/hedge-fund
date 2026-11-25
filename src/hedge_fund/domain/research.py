"""Research-layer types: per-run ratings and the per-name conviction verdict."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from enum import StrEnum

from hedge_fund.domain.cost import ModelUsage
from hedge_fund.domain.shortlist import Candidate


class Rating(StrEnum):
    """A TradingAgents rating; ``REVIEW`` marks an unparseable one."""

    BUY = "Buy"
    OVERWEIGHT = "Overweight"
    HOLD = "Hold"
    UNDERWEIGHT = "Underweight"
    SELL = "Sell"
    REVIEW = "REVIEW"

    @property
    def score(self) -> int | None:
        """Signed score in [-2, +2] (Buy +2 ... Sell -2); None for ``REVIEW``."""
        return _SCORES.get(self)


_SCORES: dict[Rating, int] = {
    Rating.BUY: 2,
    Rating.OVERWEIGHT: 1,
    Rating.HOLD: 0,
    Rating.UNDERWEIGHT: -1,
    Rating.SELL: -2,
}


class Verdict(StrEnum):
    """How the K ratings line up with the candidate's side."""

    CONFIRM = "confirm"
    NEUTRAL = "neutral"
    CONTRADICT = "contradict"
    INSUFFICIENT = "insufficient"


@dataclass(frozen=True)
class RunRecord:
    """One research run; a failed run has ``rating=None`` and an ``error``.

    ``usage`` is the run's LLM usage per model, sorted by model; None when it was not
    captured (files written before 7c, fakes, a failed counter). An errored run keeps the
    usage it was billed for.
    """

    ticker: str
    as_of: date
    run_index: int
    rating: Rating | None
    decision: str
    reports: Mapping[str, str]
    seconds: float
    error: str | None
    usage: tuple[ModelUsage, ...] | None = None

    @property
    def valid(self) -> bool:
        """True when the run produced a usable rating (no error, not ``REVIEW``)."""
        return self.error is None and self.rating is not None and self.rating is not Rating.REVIEW


@dataclass(frozen=True)
class NameResearch:
    """Runs for one candidate plus side-aligned stats; stats are None when ``valid == 0``."""

    candidate: Candidate
    runs: tuple[RunRecord, ...]
    valid: int
    mean: float | None
    std: float | None
    agreement: float | None
    verdict: Verdict


@dataclass(frozen=True)
class ResearchResult:
    """Research for a whole shortlist, in shortlist candidate order."""

    as_of: date
    names: tuple[NameResearch, ...]
