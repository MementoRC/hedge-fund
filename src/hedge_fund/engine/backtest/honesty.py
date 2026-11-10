"""Honesty counts of a backtest: what was dropped, rejected, closed out or would trip. Pure."""

from __future__ import annotations

from hedge_fund.domain.risk import NavHistory, NavPoint
from hedge_fund.engine.risk.breakers import (
    DEFAULT_BREAKERS,
    BreakerParams,
    check_breakers,
    validate_nav,
)


def breaker_days(nav: NavHistory, *, params: BreakerParams = DEFAULT_BREAKERS) -> tuple[int, int]:
    """Days on which the daily-loss and the drawdown breakers would trip, in that order.

    Equal to running ``check_breakers`` on every prefix of ``nav``, in one pass: each day
    is checked on (running peak, yesterday, today), whose max is the prefix max.
    """
    validate_nav(nav)
    daily = drawdown = 0
    peak: NavPoint | None = None
    for i, point in enumerate(nav):
        window: tuple[NavPoint, ...] = (point,)
        if i:
            previous = nav[i - 1]
            window = (previous, point)
            if peak is not None and peak.date < previous.date:
                window = (peak, *window)
        state = check_breakers(window, params=params)
        daily += state.daily_tripped
        drawdown += state.drawdown_tripped
        if peak is None or point.nav > peak.nav:
            peak = point
    return daily, drawdown
