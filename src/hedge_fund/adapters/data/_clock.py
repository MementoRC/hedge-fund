"""The exchange-calendar date the price providers treat as today."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, date, datetime

from hedge_fund.engine.execution.window import MARKET_TZ


def market_today(now: Callable[[], datetime] = lambda: datetime.now(UTC)) -> date:
    """Today's date in New York, the exchange calendar the closes are dated in."""
    return now().astimezone(MARKET_TZ).date()
