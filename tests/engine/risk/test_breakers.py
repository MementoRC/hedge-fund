"""Tests for the circuit breakers: daily loss and drawdown at the latest NAV point."""

import dataclasses
from datetime import date, timedelta

import pytest

from hedge_fund.domain.risk import BreakerState, NavPoint
from hedge_fund.engine.risk.breakers import (
    DEFAULT_BREAKERS,
    BreakerParams,
    check_breakers,
    validate_breakers,
)

NAN = float("nan")
INF = float("inf")
START = date(2026, 9, 1)
# Dyadic thresholds make "exactly at the threshold" exact in binary floating point:
# 31 / 32 - 1 == -1 / 32 and 1 - 7 / 8 == 1 / 8 with no rounding.
DYADIC = BreakerParams(daily_loss=1 / 32, drawdown=1 / 8)
QUIET = BreakerState(daily_loss=0.0, drawdown=0.0, daily_tripped=False, drawdown_tripped=False)


def _nav(*values: float) -> tuple[NavPoint, ...]:
    return tuple(NavPoint(date=START + timedelta(days=i), nav=v) for i, v in enumerate(values))


# --- params ----------------------------------------------------------------


def test_default_params() -> None:
    assert BreakerParams(daily_loss=0.025, drawdown=0.08) == DEFAULT_BREAKERS


@pytest.mark.parametrize("field", ["daily_loss", "drawdown"])
@pytest.mark.parametrize("value", [NAN, INF, True, "x"])
def test_rejects_non_finite_or_non_numeric_thresholds(field: str, value: object) -> None:
    params = dataclasses.replace(DEFAULT_BREAKERS, **{field: value})
    with pytest.raises(ValueError, match=f"{field} must be a finite number"):
        check_breakers((), params=params)


@pytest.mark.parametrize("field", ["daily_loss", "drawdown"])
@pytest.mark.parametrize("value", [0, 0.0, -0.1, 1, 1.0, 1.5])
def test_rejects_thresholds_outside_zero_one(field: str, value: float) -> None:
    params = dataclasses.replace(DEFAULT_BREAKERS, **{field: value})
    with pytest.raises(ValueError, match=f"0 < {field} < 1"):
        check_breakers((), params=params)


def test_accepts_thresholds_just_inside_zero_one() -> None:
    validate_breakers(BreakerParams(daily_loss=1e-9, drawdown=0.999))


# --- breakers --------------------------------------------------------------


def test_empty_history_trips_nothing() -> None:
    assert check_breakers(()) == QUIET


def test_single_point_has_no_daily_loss_and_no_drawdown() -> None:
    assert check_breakers(_nav(100.0)) == QUIET


@pytest.mark.parametrize(
    ("last", "tripped"),
    [(31.0, True), (31.0 - 1e-9, True), (31.0 + 1e-9, False)],
    ids=["at", "past", "short"],
)
def test_daily_loss_threshold_is_inclusive(last: float, tripped: bool) -> None:
    # 32 -> 31 is a -1/32 day; the drawdown (also 1/32) stays below 1/8.
    state = check_breakers(_nav(32.0, last), params=DYADIC)
    assert state.daily_tripped is tripped
    assert state.drawdown_tripped is False
    assert state.daily_loss == pytest.approx(last / 32 - 1)


def test_daily_loss_at_the_threshold_is_exact() -> None:
    assert check_breakers(_nav(32.0, 31.0), params=DYADIC).daily_loss == -1 / 32


@pytest.mark.parametrize(
    ("last", "tripped"),
    [(7.0, True), (7.0 - 1e-9, True), (7.0 + 1e-9, False)],
    ids=["at", "past", "short"],
)
def test_drawdown_threshold_is_inclusive(last: float, tripped: bool) -> None:
    # Peak 8, then 6, then ~7: the last day is a gain, so only the drawdown can trip.
    state = check_breakers(_nav(8.0, 6.0, last), params=DYADIC)
    assert state.drawdown_tripped is tripped
    assert state.daily_tripped is False
    assert state.drawdown == pytest.approx(1 - last / 8)


def test_default_thresholds() -> None:
    assert check_breakers(_nav(100.0, 97.0)).daily_tripped is True  # -3%
    assert check_breakers(_nav(100.0, 98.0)).daily_tripped is False  # -2%
    assert check_breakers(_nav(100.0, 90.0, 91.0)).drawdown_tripped is True  # 9%
    assert check_breakers(_nav(100.0, 90.0, 93.0)).drawdown_tripped is False  # 7%


def test_new_high_resets_the_drawdown() -> None:
    # The 10% drawdown at 90 is history: the latest point is a new high.
    state = check_breakers(_nav(100.0, 90.0, 101.0))
    assert state.drawdown == 0.0
    assert state.drawdown_tripped is False


def test_drawdown_is_measured_from_the_running_peak() -> None:
    # 110 is above the first point but 8.3% below the 120 peak; the last day is -1.8%.
    state = check_breakers(_nav(100.0, 120.0, 112.0, 110.0))
    assert state.drawdown == pytest.approx(1 - 110 / 120)
    assert state.drawdown_tripped is True
    assert state.daily_tripped is False


@pytest.mark.parametrize("value", [0.0, -1.0, NAN, INF, True])
def test_rejects_non_positive_or_non_finite_nav(value: float) -> None:
    with pytest.raises(ValueError, match="nav must be a finite number > 0"):
        check_breakers(_nav(100.0, value))


@pytest.mark.parametrize(
    "history",
    [
        (NavPoint(START, 100.0), NavPoint(START, 101.0)),
        (NavPoint(START + timedelta(days=1), 100.0), NavPoint(START, 101.0)),
    ],
    ids=["duplicate", "descending"],
)
def test_rejects_dates_not_strictly_ascending(history: tuple[NavPoint, ...]) -> None:
    with pytest.raises(ValueError, match="strictly ascending"):
        check_breakers(history)
