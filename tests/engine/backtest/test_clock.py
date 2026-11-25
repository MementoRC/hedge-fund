"""The replay's simulated clock."""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest

from hedge_fund.engine.backtest.clock import SimClock


def test_each_call_adds_one_second_from_21_30_utc():
    clock = SimClock()
    clock.start_day(date(2024, 1, 2))
    assert clock() == datetime(2024, 1, 2, 21, 30, 0, tzinfo=UTC)
    assert clock() == datetime(2024, 1, 2, 21, 30, 1, tzinfo=UTC)


def test_the_next_day_starts_at_its_own_base():
    clock = SimClock()
    clock.start_day(date(2024, 1, 2))
    clock()
    clock.start_day(date(2024, 1, 3))
    assert clock() == datetime(2024, 1, 3, 21, 30, 0, tzinfo=UTC)


def test_a_day_that_does_not_move_forward_raises():
    clock = SimClock()
    clock.start_day(date(2024, 1, 3))
    clock()
    with pytest.raises(ValueError, match="failed to move forward"):
        clock.start_day(date(2024, 1, 3))
    with pytest.raises(ValueError, match="failed to move forward"):
        clock.start_day(date(2024, 1, 2))


def test_reading_before_a_day_starts_raises():
    with pytest.raises(ValueError, match="no day started"):
        SimClock()()
