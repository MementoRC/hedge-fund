"""Honesty counts of a backtest."""

from __future__ import annotations

import random
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date, timedelta

import pytest

from hedge_fund.domain.backtest import SimEvent, SimEventKind
from hedge_fund.domain.cycle import CycleOutcome
from hedge_fund.domain.risk import NavPoint
from hedge_fund.engine.backtest.honesty import breaker_days, honesty
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


# --- honesty ---


@dataclass(frozen=True)
class _Facts:
    outcome: CycleOutcome
    price_unresolved: Mapping[str, str] = field(default_factory=dict)
    fundamentals_unresolved: Mapping[str, str] = field(default_factory=dict)


def _event(kind, *, ticker="AAA", reason=None):
    return SimEvent(day=D0, kind=kind, ticker=ticker, shares=0, price=0.0, cash=0.0, reason=reason)


CURVE = _curve(100.0, 97.0, 104.0)  # one daily trip (-3%), no drawdown trip


def test_honesty_counts_distinct_tickers_and_events():
    reports = (
        _Facts(
            CycleOutcome.SUBMITTED,
            price_unresolved={"AAA": "no_prices", "BBB": "no_sector"},
            fundamentals_unresolved={"CCC": "no facts"},
        ),
        _Facts(
            CycleOutcome.BLOCKED,
            price_unresolved={"AAA": "no_prices", "DDD": "no_prices"},
            fundamentals_unresolved={"CCC": "no facts", "EEE": "no cik"},
        ),
        _Facts(CycleOutcome.HALTED_PREFLIGHT),
    )
    events = (
        _event(SimEventKind.DELIST, ticker="XXX"),
        _event(SimEventKind.DELIST, ticker="XXX"),
        _event(SimEventKind.REJECT, reason="no open"),
        _event(SimEventKind.REJECT, reason="no open"),
        _event(SimEventKind.REJECT, reason="not trading"),
        _event(SimEventKind.FILL),
    )
    result = honesty(reports, events, CURVE, flagged=("Q", "Q", "DOW"))
    assert result.price_unresolved == {"no_prices": 2, "no_sector": 1}
    assert result.fundamentals_unresolved == 2
    assert result.flagged_excluded == 2
    assert result.stopped_trading == 1
    assert result.rejects == {"no open": 2, "not trading": 1}
    assert (result.daily_breaker_days, result.drawdown_breaker_days) == (1, 0)
    assert result.outcomes == {"blocked": 1, "halted-preflight": 1, "submitted": 1}


def test_honesty_of_a_quiet_run_is_all_zero():
    result = honesty((), (), _curve(100.0, 100.0, 100.0), flagged=())
    assert (result.daily_breaker_days, result.drawdown_breaker_days) == (0, 0)
    assert (result.price_unresolved, result.rejects, result.outcomes) == ({}, {}, {})
    assert (result.fundamentals_unresolved, result.flagged_excluded, result.stopped_trading) == (
        0,
        0,
        0,
    )


def test_a_reject_without_a_reason_raises():
    with pytest.raises(ValueError, match="without a reason"):
        honesty((), (_event(SimEventKind.REJECT),), CURVE, flagged=())
