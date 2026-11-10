"""Honesty counts of a backtest: what was dropped, rejected, closed out or would trip. Pure."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Protocol

from hedge_fund.domain.backtest import Honesty, SimEvent, SimEventKind
from hedge_fund.domain.cycle import CycleOutcome
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


class CycleFacts(Protocol):
    """The parts of a ``CycleReport`` the counts read."""

    @property
    def outcome(self) -> CycleOutcome:
        """How the cycle ended."""

    @property
    def price_unresolved(self) -> Mapping[str, str]:
        """Ticker -> reason for every ticker left unpriced."""

    @property
    def fundamentals_unresolved(self) -> Mapping[str, str]:
        """Ticker -> reason for every ticker left without fundamentals."""


def honesty(
    reports: Iterable[CycleFacts],
    events: Iterable[SimEvent],
    nav: NavHistory,
    *,
    flagged: Iterable[str],
    params: BreakerParams = DEFAULT_BREAKERS,
) -> Honesty:
    """Count drops, rejects, close-outs, would-trip days and cycle outcomes."""
    reports, events = tuple(reports), tuple(events)
    unpriced: dict[str, set[str]] = {}
    for report in reports:
        for ticker, reason in report.price_unresolved.items():
            unpriced.setdefault(reason, set()).add(ticker)
    rejects: Counter[str] = Counter()
    for event in events:
        if event.kind is SimEventKind.REJECT:
            if event.reason is None:
                raise ValueError(f"reject of {event.ticker} on {event.day} without a reason")
            rejects[event.reason] += 1
    daily, drawdown = breaker_days(nav, params=params)
    return Honesty(
        price_unresolved={reason: len(unpriced[reason]) for reason in sorted(unpriced)},
        fundamentals_unresolved=len(
            {ticker for report in reports for ticker in report.fundamentals_unresolved}
        ),
        flagged_excluded=len(set(flagged)),
        stopped_trading=len({e.ticker for e in events if e.kind is SimEventKind.DELIST}),
        rejects=dict(sorted(rejects.items())),
        daily_breaker_days=daily,
        drawdown_breaker_days=drawdown,
        outcomes=dict(sorted(Counter(report.outcome.value for report in reports).items())),
    )
