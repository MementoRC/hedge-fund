"""Tests for the veto: approval, halt and violation ordering."""

from datetime import date

import pytest

from hedge_fund.domain.risk import BreakerState, RiskBreakdown, RiskVerdict, Violation
from hedge_fund.engine.risk.veto import assess

AS_OF = date(2026, 9, 30)
ZERO = RiskBreakdown(0.0, 0.0, 0.0, 0.0, {}, {}, ())
QUIET = BreakerState(daily_loss=0.0, drawdown=0.0, daily_tripped=False, drawdown_tripped=False)
DAILY = BreakerState(daily_loss=-0.03, drawdown=0.03, daily_tripped=True, drawdown_tripped=False)
DRAWDOWN = BreakerState(daily_loss=0.0, drawdown=0.09, daily_tripped=False, drawdown_tripped=True)


def _v(check: str, detail: str = "d") -> Violation:
    return Violation(check=check, observed=1.0, limit=0.5, detail=detail)


def test_clean_book_is_approved() -> None:
    assert assess(AS_OF, [], ZERO, QUIET) == RiskVerdict(
        as_of=AS_OF,
        approved=True,
        halt=False,
        violations=(),
        breakdown=ZERO,
        breakers=QUIET,
    )


def test_any_violation_rejects_without_halting() -> None:
    verdict = assess(AS_OF, [_v("net")], ZERO, QUIET)
    assert (verdict.approved, verdict.halt) == (False, False)
    assert verdict.violations == (_v("net"),)


@pytest.mark.parametrize("state", [DAILY, DRAWDOWN], ids=["daily", "drawdown"])
def test_a_tripped_breaker_halts_and_rejects(state: BreakerState) -> None:
    verdict = assess(AS_OF, [], ZERO, state)
    assert (verdict.approved, verdict.halt) == (False, True)
    assert verdict.breakers == state


def test_halt_and_violations_are_both_reported() -> None:
    verdict = assess(AS_OF, [_v("vol"), _v("gross")], ZERO, DAILY)
    assert (verdict.approved, verdict.halt) == (False, True)
    assert [v.check for v in verdict.violations] == ["gross", "vol"]


def test_violations_are_sorted_by_check_then_detail() -> None:
    given = [_v("vol"), _v("name:B"), _v("name:A", "z"), _v("name_risk:A"), _v("name:A", "a")]
    expected = (_v("name:A", "a"), _v("name:A", "z"), _v("name:B"), _v("name_risk:A"), _v("vol"))
    assert assess(AS_OF, given, ZERO, QUIET).violations == expected
    assert assess(AS_OF, reversed(given), ZERO, QUIET).violations == expected
