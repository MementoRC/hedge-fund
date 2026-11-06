"""Plain data for the backtest's frozen store."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class Dividend:
    """A cash dividend of ``amount`` per share, ex-dated ``ex_date``.

    ``amount`` is on the frozen store's split basis (every split up to the freeze date
    applied), the basis of the stored Open and Close.
    """

    ticker: str
    ex_date: date
    amount: float


@dataclass(frozen=True)
class MemberSpan:
    """``ticker`` is an index member on every day ``d`` with ``start <= d < end``.

    ``end`` None means still a member. Each date is the first dataset snapshot on or after
    the effective date, half-open ``[start, end)``: a name is a member on ``start`` and not
    on ``end``.
    """

    ticker: str
    start: date
    end: date | None

    def __post_init__(self) -> None:
        if self.end is not None and self.end <= self.start:
            raise ValueError(f"{self.ticker}: span end {self.end} is not after start {self.start}")

    def contains(self, day: date) -> bool:
        """True when ``ticker`` is a member on ``day``."""
        return self.start <= day and (self.end is None or day < self.end)

    def overlaps(self, first: date, last: date) -> bool:
        """True when the span holds at least one day of ``first..last`` inclusive."""
        return self.start <= last and (self.end is None or self.end > first)
