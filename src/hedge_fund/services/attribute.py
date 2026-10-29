"""Layer 7b realised attribution: the book's daily P&L per ticker split into market,
sector, residual and trading, from the 7a ledger and the risk model, ex-post and
point-in-time.

Read-only: the broker is only asked for its positions. Run after ``settle``: a fill settle
has not recorded yet shows as a break. Every stage runs over injected ports; the CLI
(``hedge_fund.cli``) is the only composition root.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import cast

import pandas as pd

from hedge_fund.domain.attribution import (
    NO_PNL,
    AttributeOutcome,
    AttributionReport,
    AttributionTotals,
    DayAttribution,
)
from hedge_fund.domain.ledger import Fill, OpeningEntry, ReconciliationBreak
from hedge_fund.engine.attribution import (
    DayModel,
    Trade,
    aggregate,
    attribute_day,
    day_model,
    holdings,
    trades,
    trading_days,
)
from hedge_fund.engine.execution.window import MARKET_TZ
from hedge_fund.engine.ledger import build_lots, canonical_symbol, reconcile, signed_quantity
from hedge_fund.engine.portfolio.risk_model import (
    RiskParams,
    simple_returns,
    validate_risk_params,
)
from hedge_fund.ports.attribution import AttributionReportStore
from hedge_fund.ports.broker import Broker
from hedge_fund.ports.data import DataProvider
from hedge_fund.ports.ledger import FillStore, OpeningBalance
from hedge_fund.services.portfolio import BENCHMARK
from hedge_fund.services.snapshots import calendar_window
from hedge_fund.universe import Universe


@dataclass(frozen=True)
class AttributeDeps:
    """The ports one attribution runs over, the estimation universe and the risk params."""

    broker: Broker
    prices: DataProvider
    universe: Universe
    risk: RiskParams
    fills: FillStore
    opening: OpeningBalance
    report_store: AttributionReportStore


@dataclass(frozen=True)
class _Market:
    """What the fetch stage read: the window, closes on the trading-day calendar, sectors
    and the canonical universe."""

    start: date
    end: date
    calendar: list[date]
    closes: pd.DataFrame
    sectors: dict[str, str | None]
    universe: list[str]


@dataclass
class _Attribute:
    """What the run has reached so far; ``stage`` names the stage now running."""

    started_at: datetime
    start: date | None
    end: date | None
    stage: str = "load"
    days: tuple[DayAttribution, ...] = ()
    by_ticker: Mapping[str, AttributionTotals] = field(default_factory=dict)
    by_sector: Mapping[str, AttributionTotals] = field(default_factory=dict)
    total: AttributionTotals = NO_PNL
    breaks: tuple[ReconciliationBreak, ...] = ()
    no_beta: Mapping[str, tuple[date, ...]] = field(default_factory=dict)
    unregressed_days: tuple[date, ...] = ()
    excluded: Mapping[str, str] = field(default_factory=dict)

    def report(self, error: BaseException | None) -> AttributionReport:
        """FAILED with ``failed_stage`` and ``error`` when ``error`` is set, else BREAK when
        a ticker broke, else CLEAN."""
        if error is not None:
            outcome = AttributeOutcome.FAILED
        else:
            outcome = AttributeOutcome.BREAK if self.breaks else AttributeOutcome.CLEAN
        return AttributionReport(
            start=self.start,
            end=self.end,
            started_at=self.started_at,
            outcome=outcome,
            failed_stage=None if error is None else self.stage,
            error=None if error is None else f"{type(error).__name__}: {error}",
            days=self.days,
            by_ticker=self.by_ticker,
            by_sector=self.by_sector,
            total=self.total,
            breaks=self.breaks,
            no_beta=self.no_beta,
            unregressed_days=self.unregressed_days,
            excluded=self.excluded,
        )


def run_attribute(
    deps: AttributeDeps,
    *,
    start: date | None,
    end: date | None,
    now: Callable[[], datetime],
) -> AttributionReport:
    """Attribute the book's P&L over ``[start, end]``, save the report and return it.

    ``start`` defaults to the first opening or fill date (New York), ``end`` to the last
    SPY close up to today in New York (``now`` is read once, for ``started_at``); an empty
    ledger gives the one-day window ``[end, end]``.
    """
    run = _Attribute(started_at=now(), start=start, end=end)
    _run(deps, run)
    report = run.report(None)
    deps.report_store.save(report)
    return report


def _run(deps: AttributeDeps, run: _Attribute) -> None:
    """The stages load, reconcile, fetch, attribute and aggregate; the caller saves."""
    validate_risk_params(deps.risk)
    fills = deps.fills.load_all()
    opening = deps.opening.load()
    if run.start is None:
        run.start = _first_day(opening, fills)
    run.stage = "reconcile"
    lots, _ = build_lots(opening, fills)
    run.breaks = reconcile(lots, deps.broker.positions(), exclude=())
    run.stage = "fetch"
    market = _fetch(deps, run, opening, fills)
    run.stage = "attribute"
    _attribute_days(deps, run, opening, fills, market)
    run.stage = "aggregate"
    skipped = set(run.excluded) | {brk.ticker for brk in run.breaks}
    run.by_ticker, run.by_sector, run.total = aggregate(run.days, skipped)


def _first_day(opening: Sequence[OpeningEntry], fills: Sequence[Fill]) -> date | None:
    """The earliest opening date or New York fill date (fills with shares), None if none."""
    days = [entry.on for entry in opening]
    for fill in fills:
        if signed_quantity(fill):
            days.append(cast(datetime, fill.filled_at).astimezone(MARKET_TZ).date())
    return min(days, default=None)


def _held(opening: Sequence[OpeningEntry], fills: Sequence[Fill]) -> list[str]:
    """Canonical tickers of the opening entries and of the fills with shares, sorted."""
    tickers = {canonical_symbol(entry.ticker) for entry in opening}
    tickers |= {canonical_symbol(fill.ticker) for fill in fills if signed_quantity(fill)}
    return sorted(tickers)


def _fetch(
    deps: AttributeDeps,
    run: _Attribute,
    opening: Sequence[OpeningEntry],
    fills: Sequence[Fill],
) -> _Market:
    """Closes for universe, held and SPY from ``lookback + 1`` trading rows before the start
    (``calendar_window``) to the end, reindexed to SPY's dates (no forward fill), and the
    sectors of universe and held names; resolves the window's defaults.
    """
    last = run.end or run.started_at.astimezone(MARKET_TZ).date()
    anchor = run.start or last
    universe = sorted({canonical_symbol(ticker) for ticker in deps.universe.tickers})
    names = list(dict.fromkeys([*universe, *_held(opening, fills)]))
    frame = deps.prices.closes(
        list(dict.fromkeys([*names, BENCHMARK])),
        anchor - calendar_window(deps.risk.lookback + 1),
        last,
    )
    calendar = trading_days(frame[BENCHMARK], date.min, last)
    if not calendar:
        raise ValueError(f"no {BENCHMARK} closes up to {last}")
    end = run.end or calendar[-1]
    start = run.start or end
    run.start, run.end = start, end
    sectors = deps.prices.sectors(names)
    closes = frame.reindex(pd.DatetimeIndex([pd.Timestamp(day) for day in calendar]))
    return _Market(start, end, calendar, closes, sectors, universe)


def _window(calendar: Sequence[date], start: date, end: date) -> range:
    """Calendar positions of the window's trading days; the first has a previous one."""
    if start > end:
        raise ValueError(f"start {start} is after end {end}")
    positions = [index for index, day in enumerate(calendar) if start <= day <= end]
    if not positions:
        raise ValueError(f"no trading day between {start} and {end}")
    if positions[0] == 0:
        raise ValueError(f"no trading day before {calendar[0]} in the closes")
    return range(positions[0], positions[-1] + 1)


def _attribute_days(
    deps: AttributeDeps,
    run: _Attribute,
    opening: Sequence[OpeningEntry],
    fills: Sequence[Fill],
    market: _Market,
) -> None:
    """Per window day: the day model (only when something is held at the previous close),
    then ``attribute_day``. Keeps each excluded ticker's first reason."""
    returns = simple_returns(market.closes)
    benchmark = returns[BENCHMARK]
    book = holdings(opening, fills, market.calendar)
    traded: dict[date, list[Trade]] = {}
    for trade in trades(opening, fills, market.calendar):
        traded.setdefault(trade.day, []).append(trade)
    rows: list[DayAttribution] = []
    excluded: dict[str, str] = {}
    no_beta: dict[str, list[date]] = {}
    unregressed: list[date] = []
    for position in _window(market.calendar, market.start, market.end):
        day = market.calendar[position]
        held = book[market.calendar[position - 1]]
        model = None
        if held:
            model = day_model(returns, benchmark, market.sectors, market.universe, deps.risk, day)
            if model is None:
                unregressed.append(day)
        day_rows, day_excluded = attribute_day(
            day, held, traded.get(day, []), market.closes, model, market.sectors
        )
        rows.extend(day_rows)
        for ticker, reason in day_excluded.items():
            excluded.setdefault(ticker, reason)
        _note_no_beta(no_beta, model, day_rows)
    run.days = tuple(rows)
    run.excluded = dict(sorted(excluded.items()))
    run.no_beta = {ticker: tuple(days) for ticker, days in sorted(no_beta.items())}
    run.unregressed_days = tuple(unregressed)


def _note_no_beta(
    no_beta: dict[str, list[date]], model: DayModel | None, rows: Sequence[DayAttribution]
) -> None:
    """On a regressed day, a held row that was not regressed is a no-beta day."""
    if model is None:
        return
    for row in rows:
        if row.prior_shares and not row.regressed:
            no_beta.setdefault(row.ticker, []).append(row.day)
