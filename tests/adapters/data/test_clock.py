"""market_today: the New York calendar date, whatever the host timezone."""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest

from hedge_fund.adapters.data._clock import market_today


@pytest.mark.parametrize(
    ("utc_now", "expected"),
    [
        (datetime(2026, 11, 3, 1, 0, tzinfo=UTC), date(2026, 11, 2)),
        (datetime(2026, 11, 2, 15, 0, tzinfo=UTC), date(2026, 11, 2)),
        (datetime(2026, 3, 8, 4, 30, tzinfo=UTC), date(2026, 3, 7)),
        (datetime(2026, 3, 9, 4, 30, tzinfo=UTC), date(2026, 3, 9)),
    ],
)
def test_market_today_is_the_new_york_date(utc_now: datetime, expected: date) -> None:
    assert market_today(lambda: utc_now) == expected


def test_default_now_is_timezone_aware() -> None:
    assert isinstance(market_today(), date)
