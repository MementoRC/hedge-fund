"""Whole-book veto: merge every violation and the breaker state into one verdict."""

from __future__ import annotations

import datetime
from collections.abc import Iterable

from hedge_fund.domain.risk import BreakerState, RiskBreakdown, RiskVerdict, Violation


def assess(
    as_of: datetime.date,
    violations: Iterable[Violation],
    breakdown: RiskBreakdown,
    breaker_state: BreakerState,
) -> RiskVerdict:
    """Approve only with no violation and no tripped breaker; never repair the book.

    Violations are sorted by ``(check, detail)``; ``halt`` is set when either breaker
    tripped, and a halt always rejects.
    """
    ordered = tuple(sorted(violations, key=lambda v: (v.check, v.detail)))
    halt = breaker_state.daily_tripped or breaker_state.drawdown_tripped
    return RiskVerdict(
        as_of=as_of,
        approved=not ordered and not halt,
        halt=halt,
        violations=ordered,
        breakdown=breakdown,
        breakers=breaker_state,
    )
