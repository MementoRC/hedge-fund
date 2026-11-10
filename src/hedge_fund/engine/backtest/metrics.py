"""Backtest metrics: an equity curve against its benchmark, and trading costs. Pure."""

from __future__ import annotations

import itertools
import math
import statistics
from collections.abc import Iterable
from datetime import date

from hedge_fund.domain.backtest import Performance, SimEvent, SimEventKind, TradingCosts
from hedge_fund.domain.risk import NavHistory
from hedge_fund.engine.factors.definitions import TRADING_DAYS
from hedge_fund.engine.risk.breakers import validate_nav


def daily_returns(nav: NavHistory) -> tuple[float, ...]:
    """Simple daily returns of a validated curve of at least three points."""
    _validate_curve(nav)
    return tuple((after.nav - before.nav) / before.nav for before, after in itertools.pairwise(nav))


def performance(nav: NavHistory) -> Performance:
    """CAGR, volatility (ddof=1), Sharpe (risk-free 0) and max drawdown of ``nav``."""
    returns = daily_returns(nav)
    years = len(returns) / TRADING_DAYS
    total = nav[-1].nav / nav[0].nav - 1
    volatility = statistics.stdev(returns) * math.sqrt(TRADING_DAYS)
    sharpe = statistics.fmean(returns) * TRADING_DAYS / volatility if volatility > 0 else None
    drawdown, peak, trough = _max_drawdown(nav)
    return Performance(
        start=nav[0].date,
        end=nav[-1].date,
        total_return=total,
        cagr=(1 + total) ** (1 / years) - 1,
        volatility=volatility,
        sharpe=sharpe,
        max_drawdown=drawdown,
        peak=peak,
        trough=trough,
    )


def beta(nav: NavHistory, benchmark: NavHistory) -> float | None:
    """``cov(r, b) / var(b)`` of daily returns; None when the benchmark has no variance."""
    if [point.date for point in nav] != [point.date for point in benchmark]:
        raise ValueError("strategy and benchmark curves must have the same dates")
    ours, theirs = daily_returns(nav), daily_returns(benchmark)
    variance = statistics.variance(theirs)
    if variance == 0:
        return None
    return statistics.covariance(ours, theirs) / variance


def trading_costs(events: Iterable[SimEvent], nav: NavHistory) -> TradingCosts:
    """Traded notional, turnover, slippage and borrow over the curve's window."""
    _validate_curve(nav)
    window = _in_window(events, nav)
    fills = [event for event in window if event.kind is SimEventKind.FILL]
    notional = math.fsum(abs(event.shares) * event.price for event in fills)
    years = (len(nav) - 1) / TRADING_DAYS
    return TradingCosts(
        traded_notional=notional,
        turnover=notional / statistics.fmean(point.nav for point in nav) / years,
        slippage=math.fsum(event.cost for event in fills),
        borrow=math.fsum(event.cost for event in window if event.kind is SimEventKind.BORROW),
    )


def _in_window(events: Iterable[SimEvent], nav: NavHistory) -> tuple[SimEvent, ...]:
    collected = tuple(events)
    for event in collected:
        if not nav[0].date <= event.day <= nav[-1].date:
            raise ValueError(
                f"{event.kind} event on {event.day} is outside the curve "
                f"{nav[0].date}..{nav[-1].date}"
            )
    return collected


def _max_drawdown(nav: NavHistory) -> tuple[float, date, date]:
    peak = nav[0]
    worst, worst_peak, worst_trough = 0.0, nav[0].date, nav[0].date
    for point in nav:
        if point.nav > peak.nav:
            peak = point
        drawdown = 1 - point.nav / peak.nav
        if drawdown > worst:
            worst, worst_peak, worst_trough = drawdown, peak.date, point.date
    return worst, worst_peak, worst_trough


def _validate_curve(nav: NavHistory) -> None:
    validate_nav(nav)
    if len(nav) < 3:
        raise ValueError(f"an equity curve needs at least three points, got {len(nav)}")
