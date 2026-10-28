"""previous_weekday: the previous Friday for Saturday, Sunday and Monday, else the day before."""

from __future__ import annotations

from datetime import date

import pytest

from hedge_fund.engine.dates import previous_weekday


@pytest.mark.parametrize(
    ("today", "expected"),
    [
        (date(2026, 10, 6), date(2026, 10, 5)),  # Tuesday -> Monday
        (date(2026, 10, 2), date(2026, 10, 1)),  # Friday -> Thursday
        (date(2026, 10, 3), date(2026, 10, 2)),  # Saturday -> Friday
        (date(2026, 10, 4), date(2026, 10, 2)),  # Sunday -> Friday
        (date(2026, 10, 5), date(2026, 10, 2)),  # Monday -> Friday
    ],
)
def test_previous_weekday(today: date, expected: date) -> None:
    assert previous_weekday(today) == expected
