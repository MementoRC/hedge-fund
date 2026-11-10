"""Backtest metrics against hand-computed values."""

from __future__ import annotations

import math
from datetime import date, timedelta

import pytest

from hedge_fund.domain.risk import NavPoint
from hedge_fund.engine.backtest.metrics import beta, daily_returns, performance

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
