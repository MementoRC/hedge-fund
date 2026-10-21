"""Tests for the risk domain types."""

import dataclasses
from datetime import date

import pytest

from hedge_fund.domain.risk import (
    BreakerState,
    NavHistory,
    NavPoint,
    RiskBreakdown,
    RiskVerdict,
    Violation,
)

AS_OF = date(2026, 9, 30)


def test_nav_history_is_a_tuple_of_points() -> None:
    history: NavHistory = (NavPoint(date=date(2026, 9, 29), nav=100.0), NavPoint(AS_OF, 99.0))
    assert [point.nav for point in history] == [100.0, 99.0]


def test_risk_types_are_frozen() -> None:
    violation = Violation(check="gross", observed=2.5, limit=2.0, detail="sum of |weight|")
    breakdown = RiskBreakdown(
        total_vol=0.1,
        factor_vol=0.06,
        specific_vol=0.08,
        factor_share=0.36,
        by_factor={"market": 0.0036},
        by_name={"A": 1.0},
        unassessed=(),
    )
    state = BreakerState(daily_loss=0.0, drawdown=0.0, daily_tripped=False, drawdown_tripped=False)
    verdict = RiskVerdict(
        as_of=AS_OF,
        approved=False,
        halt=False,
        violations=(violation,),
        breakdown=breakdown,
        breakers=state,
    )
    for instance, field in [
        (NavPoint(AS_OF, 100.0), "nav"),
        (violation, "observed"),
        (breakdown, "total_vol"),
        (state, "drawdown"),
        (verdict, "approved"),
    ]:
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(instance, field, 0.0)
