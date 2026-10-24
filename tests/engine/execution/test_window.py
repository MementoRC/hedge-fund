"""in_session_window: weekday 09:25 <= New York wall time < 16:15, DST-aware, no holidays."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from hedge_fund.engine.execution.window import (
    MARKET_TZ,
    SESSION_END,
    SESSION_START,
    in_session_window,
)

NEW_YORK = ZoneInfo("America/New_York")


def _ny(day: int, hour: int, minute: int, second: int = 0, *, month: int = 9) -> datetime:
    """A New York wall time in 2026 (2026-09-30 is a Wednesday, 2026-10-03 a Saturday)."""
    return datetime(2026, month, day, hour, minute, second, tzinfo=NEW_YORK)


def test_constants() -> None:
    assert SESSION_START.isoformat() == "09:25:00"
    assert SESSION_END.isoformat() == "16:15:00"
    assert MARKET_TZ.key == "America/New_York"


@pytest.mark.parametrize(
    ("at", "inside"),
    [
        pytest.param(_ny(30, 9, 24, 59), False, id="09:24:59"),
        pytest.param(_ny(30, 9, 25), True, id="09:25:00"),
        pytest.param(_ny(30, 12, 0), True, id="noon"),
        pytest.param(_ny(30, 16, 14, 59), True, id="16:14:59"),
        pytest.param(_ny(30, 16, 15), False, id="16:15:00"),
        pytest.param(_ny(30, 0, 0), False, id="midnight"),
        pytest.param(_ny(30, 23, 59, 59), False, id="23:59:59"),
        pytest.param(_ny(28, 10, 0), True, id="monday-10:00"),
        pytest.param(_ny(2, 10, 0, month=10), True, id="friday-10:00"),
        pytest.param(_ny(3, 10, 0, month=10), False, id="saturday-10:00"),
        pytest.param(_ny(4, 10, 0, month=10), False, id="sunday-10:00"),
    ],
)
def test_weekday_session_bounds(at: datetime, inside: bool) -> None:
    assert in_session_window(at) is inside


@pytest.mark.parametrize(
    ("at", "inside"),
    [
        pytest.param(datetime(2026, 9, 30, 13, 30, tzinfo=UTC), True, id="13:30-utc-is-09:30-edt"),
        pytest.param(datetime(2026, 9, 30, 13, 24, tzinfo=UTC), False, id="13:24-utc-is-09:24-edt"),
        pytest.param(datetime(2026, 12, 2, 14, 30, tzinfo=UTC), True, id="14:30-utc-is-09:30-est"),
        pytest.param(datetime(2026, 12, 2, 13, 30, tzinfo=UTC), False, id="13:30-utc-is-08:30-est"),
        pytest.param(datetime(2026, 12, 2, 21, 14, tzinfo=UTC), True, id="21:14-utc-is-16:14-est"),
        pytest.param(datetime(2026, 9, 30, 20, 15, tzinfo=UTC), False, id="20:15-utc-is-16:15-edt"),
        pytest.param(
            datetime(2026, 10, 3, 1, 0, tzinfo=timezone(timedelta(hours=10))),
            True,
            id="saturday-offset-is-friday-11:00-edt",
        ),
        pytest.param(
            datetime(2026, 10, 3, 0, 30, tzinfo=UTC), False, id="saturday-utc-is-friday-20:30-edt"
        ),
    ],
)
def test_other_zones_are_converted_to_new_york(at: datetime, inside: bool) -> None:
    assert in_session_window(at) is inside


def test_a_naive_datetime_raises() -> None:
    with pytest.raises(ValueError, match="must be timezone-aware"):
        in_session_window(datetime(2026, 9, 30, 10, 0))


def test_package_exports() -> None:
    import hedge_fund.engine.execution as execution

    assert execution.in_session_window is in_session_window
    assert execution.MARKET_TZ is MARKET_TZ
