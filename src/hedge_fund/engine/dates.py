"""Calendar helpers: weekdays only, no holiday calendar. Pure."""

from __future__ import annotations

from datetime import date, timedelta

_WEEKEND_DAYS_BACK = {0: 3, 5: 1, 6: 2}  # Monday, Saturday, Sunday -> previous Friday


def previous_weekday(day: date) -> date:
    """Previous Friday for Saturday, Sunday and Monday, otherwise the day before."""
    return day - timedelta(days=_WEEKEND_DAYS_BACK.get(day.weekday(), 1))
