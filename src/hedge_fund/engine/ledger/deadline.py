"""When an order the broker never acknowledged may be recorded as NOT_SUBMITTED.

Pure: the instant is passed in. New York wall time via ``zoneinfo`` (DST-aware); no holiday
calendar, so a holiday only delays the deadline by keeping the id pending one more day.
"""

from __future__ import annotations

from datetime import UTC, datetime, time, timedelta

from hedge_fund.engine.execution.window import MARKET_TZ

FILL_CUTOFF = time(16, 0)  # New York close: an opg order is final by then

_SATURDAY = 5


def fill_deadline(started_at: datetime) -> datetime:
    """16:00 New York on the first weekday strictly after ``started_at``'s New York date, in UTC.

    Raises ValueError for a naive ``started_at``.
    """
    if started_at.utcoffset() is None:
        raise ValueError(f"started_at must be timezone-aware, got {started_at!r}")
    day = started_at.astimezone(MARKET_TZ).date() + timedelta(days=1)
    while day.weekday() >= _SATURDAY:
        day += timedelta(days=1)
    return datetime.combine(day, FILL_CUTOFF, tzinfo=MARKET_TZ).astimezone(UTC)
