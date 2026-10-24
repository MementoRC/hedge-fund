"""Weekday submission window: when ``opg`` (market-on-open) orders must not be sent.

Pure: the instant is passed in, no clock is read. New York wall time via ``zoneinfo``, so
the window follows DST; there is no holiday calendar.
"""

from __future__ import annotations

from datetime import datetime, time
from zoneinfo import ZoneInfo

SESSION_START = time(9, 25)
SESSION_END = time(16, 15)
MARKET_TZ = ZoneInfo("America/New_York")

_SATURDAY = 5


def in_session_window(at: datetime) -> bool:
    """True when ``at`` in New York is Monday to Friday, ``SESSION_START <= time < SESSION_END``.

    Raises ValueError for a naive ``at``.
    """
    if at.utcoffset() is None:
        raise ValueError(f"at must be timezone-aware, got {at!r}")
    local = at.astimezone(MARKET_TZ)
    return local.weekday() < _SATURDAY and SESSION_START <= local.time() < SESSION_END
