"""Risk engine types: NAV history, limit violations, risk breakdown and the veto verdict."""

from __future__ import annotations

import datetime
from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class NavPoint:
    """Net asset value of the book at the close of ``date``."""

    date: datetime.date
    nav: float


NavHistory = tuple[NavPoint, ...]
"""Ascending unique dates with ``nav > 0``; checked by ``engine.risk.breakers.validate_nav``."""


@dataclass(frozen=True)
class Violation:
    """One failed check: ``observed`` is the value compared against ``limit``.

    ``check`` is the check id (``gross``, ``name:<T>``, ``unassessable:<T>``, ...). An
    ``unassessable`` violation has ``observed = limit = nan``.
    """

    check: str
    observed: float
    limit: float
    detail: str


@dataclass(frozen=True)
class RiskBreakdown:
    """Annualised ex-ante factor-model risk of the assessable names of a book.

    ``by_factor`` holds variance contributions (they sum to ``factor_vol ** 2``) and
    ``by_name`` shares of total variance (they sum to 1; a hedge can be negative), keyed by
    ticker in sorted order. ``unassessed`` lists the sorted book tickers left out. A book
    with nothing assessable is all zeros with empty mappings.
    """

    total_vol: float
    factor_vol: float
    specific_vol: float
    factor_share: float
    by_factor: Mapping[str, float]
    by_name: Mapping[str, float]
    unassessed: tuple[str, ...]


@dataclass(frozen=True)
class BreakerState:
    """Circuit breakers at the latest NAV point; ``daily_loss`` is the signed last return."""

    daily_loss: float
    drawdown: float
    daily_tripped: bool
    drawdown_tripped: bool


@dataclass(frozen=True)
class RiskVerdict:
    """Whole-book decision: ``approved = not violations and not halt``.

    ``violations`` are sorted by ``(check, detail)``; ``halt`` is set when a breaker tripped.
    """

    as_of: datetime.date
    approved: bool
    halt: bool
    violations: tuple[Violation, ...]
    breakdown: RiskBreakdown
    breakers: BreakerState
