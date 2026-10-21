"""Circuit breakers over a NAV history: daily loss and drawdown at the latest point."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

from hedge_fund.domain.risk import BreakerState, NavHistory


@dataclass(frozen=True)
class BreakerParams:
    """Trip on a last daily return <= -daily_loss or a drawdown >= drawdown (fractions)."""

    daily_loss: float = 0.025
    drawdown: float = 0.08


DEFAULT_BREAKERS = BreakerParams()


def check_breakers(nav: NavHistory, *, params: BreakerParams = DEFAULT_BREAKERS) -> BreakerState:
    """Measure both breakers at the latest NAV point; thresholds are inclusive.

    ``daily_loss`` is ``nav_t / nav_{t-1} - 1`` (0 with fewer than two points) and
    ``drawdown`` is ``1 - nav_t / max(nav)`` (0 for an empty history). Raises ValueError on
    invalid params or a malformed history.
    """
    validate_breakers(params)
    validate_nav(nav)
    if not nav:
        return BreakerState(
            daily_loss=0.0, drawdown=0.0, daily_tripped=False, drawdown_tripped=False
        )
    last = nav[-1].nav
    daily = last / nav[-2].nav - 1 if len(nav) > 1 else 0.0
    drawdown = 1 - last / max(point.nav for point in nav)
    return BreakerState(
        daily_loss=daily,
        drawdown=drawdown,
        daily_tripped=daily <= -params.daily_loss,
        drawdown_tripped=drawdown >= params.drawdown,
    )


def validate_breakers(params: BreakerParams) -> None:
    """Type check first: bool and NaN would slip past plain range comparisons."""
    names = ("daily_loss", "drawdown")
    for name in names:
        value = getattr(params, name)
        if (
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(value)
        ):
            raise ValueError(f"{name} must be a finite number, got {value!r}")
    for name in names:
        value = getattr(params, name)
        if not 0 < value < 1:
            raise ValueError(f"{name} must satisfy 0 < {name} < 1, got {value!r}")


def validate_nav(nav: NavHistory) -> None:
    """Every nav a finite number > 0 and dates strictly ascending (hence unique)."""
    for point in nav:
        value = point.nav
        if (
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(value)
            or not value > 0
        ):
            raise ValueError(f"nav must be a finite number > 0, got {value!r} on {point.date}")
    for before, after in itertools.pairwise(nav):
        if not before.date < after.date:
            raise ValueError(
                f"NAV dates must be strictly ascending, got {after.date} after {before.date}"
            )
