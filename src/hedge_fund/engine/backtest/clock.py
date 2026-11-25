"""A simulated UTC clock that strictly increases across the replay. Pure."""

from __future__ import annotations

from datetime import UTC, date, datetime, time, timedelta

_AFTER_CLOSE = time(21, 30)  # after the New York close in both EST and EDT
_TICK = timedelta(seconds=1)


class SimClock:
    """``now()`` for the replay: day ``d`` starts at ``d`` 21:30 UTC, and every call adds 1 s.

    Cycle reports are named ``<as_of>_<HHMMSS>`` in UTC, so the clock must strictly
    increase; ``start_day`` raises if it would not.
    """

    def __init__(self) -> None:
        self._next: datetime | None = None
        self._last: datetime | None = None

    def start_day(self, day: date) -> None:
        base = datetime.combine(day, _AFTER_CLOSE, tzinfo=UTC)
        if self._last is not None and base <= self._last:
            raise ValueError(f"clock failed to move forward: {base} after {self._last}")
        self._next = base

    def __call__(self) -> datetime:
        if self._next is None:
            raise ValueError("no day started on the simulated clock")
        now = self._next
        self._next = now + _TICK
        self._last = now
        return now
