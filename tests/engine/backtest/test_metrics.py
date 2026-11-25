"""Backtest metrics against hand-computed values."""

from __future__ import annotations

import math
from datetime import date, timedelta

import pytest

from hedge_fund.domain.backtest import SimEvent, SimEventKind
from hedge_fund.domain.risk import NavPoint
from hedge_fund.engine.backtest.honesty import honesty
from hedge_fund.engine.backtest.metrics import (
    backtest_metrics,
    beta,
    daily_returns,
    performance,
    trading_costs,
)

D0 = date(2024, 1, 1)


def _curve(*values: float) -> tuple[NavPoint, ...]:
    return tuple(NavPoint(date=D0 + timedelta(days=i), nav=v) for i, v in enumerate(values))


def _day(i: int) -> date:
    return D0 + timedelta(days=i)


# --- daily_returns ---


def test_daily_returns_are_simple_returns():
    assert daily_returns(_curve(100.0, 125.0, 100.0, 150.0)) == (0.25, -0.2, 0.5)


@pytest.mark.parametrize("values", [(), (100.0,), (100.0, 110.0)])
def test_a_curve_needs_three_points(values):
    with pytest.raises(ValueError, match="at least three points"):
        daily_returns(_curve(*values))


def test_a_malformed_curve_raises():
    with pytest.raises(ValueError, match="finite number > 0"):
        daily_returns(_curve(100.0, 0.0, 100.0))


# --- performance ---


def test_performance_hand_computed():
    result = performance(_curve(100.0, 125.0, 100.0, 150.0))
    mean = 0.55 / 3
    stdev = math.sqrt(((0.25 - mean) ** 2 + (-0.2 - mean) ** 2 + (0.5 - mean) ** 2) / 2)
    assert result.start == _day(0)
    assert result.end == _day(3)
    assert result.total_return == pytest.approx(0.5)
    assert result.cagr == pytest.approx(1.5 ** (252 / 3) - 1)
    assert result.volatility == pytest.approx(stdev * math.sqrt(252))
    assert result.sharpe == pytest.approx(mean * 252 / (stdev * math.sqrt(252)))
    assert result.max_drawdown == pytest.approx(0.2)
    assert (result.peak, result.trough) == (_day(1), _day(2))


def test_cagr_of_a_doubling_over_one_year_is_one():
    curve = _curve(*(100.0 * 2 ** (i / 252) for i in range(253)))
    assert performance(curve).cagr == pytest.approx(1.0)


def test_sharpe_is_none_without_volatility():
    result = performance(_curve(1.0, 2.0, 4.0, 8.0))
    assert result.volatility == 0.0
    assert result.sharpe is None


def test_max_drawdown_keeps_the_worst_peak_to_trough():
    result = performance(_curve(100.0, 120.0, 90.0, 130.0, 65.0, 70.0))
    assert result.max_drawdown == pytest.approx(0.5)
    assert (result.peak, result.trough) == (_day(3), _day(4))


def test_no_drawdown_dates_both_at_the_start():
    result = performance(_curve(1.0, 2.0, 4.0))
    assert result.max_drawdown == 0.0
    assert (result.peak, result.trough) == (_day(0), _day(0))


# --- beta ---


def test_beta_of_a_doubled_benchmark_is_two():
    benchmark = _curve(100.0, 110.0, 99.0, 108.9)  # +10%, -10%, +10%
    strategy = _curve(100.0, 120.0, 96.0, 115.2)  # +20%, -20%, +20%
    assert beta(strategy, benchmark) == pytest.approx(2.0)


def test_beta_is_none_for_a_benchmark_without_variance():
    assert beta(_curve(100.0, 125.0, 100.0, 150.0), _curve(1.0, 2.0, 4.0, 8.0)) is None


def test_beta_needs_the_same_dates():
    curve = _curve(1.0, 2.0, 3.0)
    shifted = tuple(NavPoint(date=p.date + timedelta(days=1), nav=p.nav) for p in curve)
    with pytest.raises(ValueError, match="same dates"):
        beta(curve, shifted)


# --- trading_costs ---


def _event(kind, *, day=D0, ticker="AAA", shares=0, price=0.0, cost=0.0, reason=None):
    return SimEvent(
        day=day,
        kind=kind,
        ticker=ticker,
        shares=shares,
        price=price,
        cash=0.0,
        cost=cost,
        reason=reason,
    )


YEAR_FLAT = _curve(*([1000.0] * 253))  # 252 returns = one year, mean NAV 1000


def test_trading_costs_hand_computed():
    events = (
        _event(SimEventKind.FILL, shares=10, price=50.0, cost=0.25),
        _event(SimEventKind.FILL, shares=-4, price=25.0, cost=0.0625),
        _event(SimEventKind.BORROW, cost=0.125),
        _event(SimEventKind.DIVIDEND, shares=10, price=1.0),
    )
    costs = trading_costs(events, YEAR_FLAT)
    assert costs.traded_notional == 600.0
    assert costs.turnover == pytest.approx(0.6)
    assert costs.slippage == 0.3125
    assert costs.borrow == 0.125


def test_no_events_cost_nothing():
    costs = trading_costs((), YEAR_FLAT)
    assert (costs.traded_notional, costs.turnover, costs.slippage, costs.borrow) == (0, 0, 0, 0)


@pytest.mark.parametrize("offset", [-1, 253])
def test_an_event_outside_the_curve_raises(offset):
    with pytest.raises(ValueError, match="outside the curve"):
        trading_costs((_event(SimEventKind.FILL, day=_day(offset)),), YEAR_FLAT)


# --- backtest_metrics ---


def test_backtest_metrics_assembles_the_parts():
    strategy = _curve(100.0, 120.0, 96.0, 115.2)
    benchmark = _curve(100.0, 110.0, 99.0, 108.9)
    events = (_event(SimEventKind.FILL, day=_day(1), shares=2, price=10.0, cost=0.01),)
    result = backtest_metrics(strategy, benchmark, events, (), flagged=("Q",))
    assert result.strategy == performance(strategy)
    assert result.benchmark == performance(benchmark)
    assert result.beta == beta(strategy, benchmark)
    assert result.costs == trading_costs(events, strategy)
    assert result.honesty == honesty((), events, strategy, flagged=("Q",))
