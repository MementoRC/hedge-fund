"""Honesty counts of a backtest."""

from __future__ import annotations

import random
from datetime import date, timedelta

import pytest

from hedge_fund.domain.risk import NavPoint
from hedge_fund.engine.backtest.honesty import breaker_days
from hedge_fund.engine.risk.breakers import BreakerParams, check_breakers

D0 = date(2024, 1, 1)


def _curve(*values: float) -> tuple[NavPoint, ...]:
    return tuple(NavPoint(date=D0 + timedelta(days=i), nav=v) for i, v in enumerate(values))


# --- breaker_days ---


def test_breaker_days_hand_counted():
    # daily (-2.5%): 100->97, 104->100, 98->95. drawdown (8%) from 104: 95, 94
    curve = _curve(100.0, 97.0, 104.0, 100.0, 98.0, 95.0, 94.0)
    assert breaker_days(curve) == (3, 2)


def test_breaker_days_use_the_given_params():
    # daily (-0.5%): 100->99, 99->98. drawdown (1.5%): 98
    curve = _curve(100.0, 99.0, 98.0)
    assert breaker_days(curve, params=BreakerParams(daily_loss=0.005, drawdown=0.015)) == (2, 1)


def test_an_empty_curve_has_no_breaker_days():
    assert breaker_days(()) == (0, 0)


@pytest.mark.parametrize("seed", range(200))
def test_breaker_days_match_the_per_prefix_check(seed):
    rng = random.Random(seed)
    values = [
        rng.choice((rng.uniform(1.0, 200.0), 100.0, 92.0, 97.5)) for _ in range(rng.randint(1, 40))
    ]
    curve = _curve(*values)
    states = [check_breakers(curve[: end + 1]) for end in range(len(curve))]
    expected = (sum(s.daily_tripped for s in states), sum(s.drawdown_tripped for s in states))
    assert breaker_days(curve) == expected
