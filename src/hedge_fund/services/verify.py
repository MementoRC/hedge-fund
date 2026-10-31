"""Layer 7d thesis and rating verification: every stored research rating scored against the
excess return over SPY that followed it, over 5, 21 and 63 trading days.

Read-only and stateless: each run rescans the research dates in its window, the 7a fills
(only to mark which observations were executed) and adjusted closes, and writes one report.
Every stage runs over injected ports; the CLI (``hedge_fund.cli``) is the only composition
root. Whatever happens one VerificationReport is saved: a run that raises (an Exception or an
interrupt) saves a FAILED report naming the stage, then re-raises.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta

import pandas as pd

from hedge_fund.domain.research import ResearchResult
from hedge_fund.domain.verification import (
    HORIZONS,
    ExclusionReason,
    HorizonSummary,
    Observation,
    VerificationReport,
    VerifyOutcome,
)
from hedge_fund.engine.attribution import trading_days
from hedge_fund.engine.execution.window import MARKET_TZ
from hedge_fund.engine.ledger import canonical_symbol
from hedge_fund.engine.verification import observe, summarise
from hedge_fund.ports.data import DataProvider
from hedge_fund.ports.ledger import FillStore
from hedge_fund.ports.research import ResearchStore
from hedge_fund.ports.verification import VerificationReportStore
from hedge_fund.services.portfolio import BENCHMARK


class VerificationSaveError(Exception):
    """The verification finished but its report could not be saved; carries ``report``.

    ``__cause__`` is the store's exception.
    """

    def __init__(self, report: VerificationReport) -> None:
        super().__init__(f"the {report.outcome.value} verification report could not be saved")
        self.report = report


_BASE_PAD = timedelta(days=10)  # closes are fetched this far before the earliest research date


@dataclass(frozen=True)
class VerifyDeps:
    """The ports one verification runs over."""

    research: ResearchStore
    fills: FillStore
    data: DataProvider
    reports: VerificationReportStore


@dataclass(frozen=True)
class _Market:
    """What the fetch stage read: SPY's trading days and every close reindexed to them."""

    calendar: list[date]
    closes: pd.DataFrame


@dataclass
class _Verify:
    """What the run has reached so far; ``stage`` names the stage now running."""

    started_at: datetime
    today: date
    start: date | None
    end: date | None
    stage: str = "load"
    observations: tuple[Observation, ...] = ()
    summaries: tuple[HorizonSummary, ...] = ()

    def report(self, error: BaseException | None) -> VerificationReport:
        """FAILED with ``failed_stage`` and ``error`` when ``error`` is set, else GAPS when an
        observation has no price, else CLEAN."""
        if error is not None:
            outcome = VerifyOutcome.FAILED
        elif any(row.reason is ExclusionReason.NO_PRICE for row in self.observations):
            outcome = VerifyOutcome.GAPS
        else:
            outcome = VerifyOutcome.CLEAN
        return VerificationReport(
            start=self.start,
            end=self.end,
            started_at=self.started_at,
            outcome=outcome,
            failed_stage=None if error is None else self.stage,
            error=None if error is None else f"{type(error).__name__}: {error}",
            observations=self.observations,
            summaries=self.summaries,
        )


def run_verify(
    deps: VerifyDeps,
    *,
    start: date | None,
    end: date | None,
    now: Callable[[], datetime],
) -> VerificationReport:
    """Verify the research dated in ``[start, end]``, save the report and return it.

    ``start`` defaults to the earliest research date, ``end`` to today in New York (``now``
    is read once, for ``started_at`` and today). With no research in the window no prices are
    fetched and the report has no observations and one zeroed summary per horizon. Any
    exception (KeyboardInterrupt included) saves a FAILED report with ``failed_stage`` and
    ``error`` and is re-raised; if that save fails too, the failure is added to it with
    ``add_note``. A completed run whose report cannot be saved raises VerificationSaveError
    carrying the report.
    """
    started_at = now()
    run = _Verify(
        started_at=started_at,
        today=started_at.astimezone(MARKET_TZ).date(),
        start=start,
        end=end,
    )
    try:
        _run(deps, run)
    except BaseException as error:
        try:
            deps.reports.save(run.report(error))
        except Exception as save_error:
            error.add_note(
                "the verification report could not be saved: "
                f"{type(save_error).__name__}: {save_error}"
            )
        raise
    report = run.report(None)
    try:
        deps.reports.save(report)
    except Exception as save_error:
        raise VerificationSaveError(report) from save_error
    return report


def _run(deps: VerifyDeps, run: _Verify) -> None:
    """The stages load, fetch, observe and summarise; the caller saves."""
    results, executed = _load(deps, run)
    if results:
        run.stage = "fetch"
        market = _fetch(deps, run, results)
        run.stage = "observe"
        run.observations = _observe(results, executed, market)
    run.stage = "summarise"
    run.summaries = tuple(summarise(run.observations, horizon) for horizon in HORIZONS)


def _load(deps: VerifyDeps, run: _Verify) -> tuple[list[ResearchResult], set[tuple[date, str]]]:
    """The research results dated in the window (resolving its bounds) and the executed set:
    ``(cycle_as_of, canonical ticker)`` of every fill with shares."""
    end = run.end if run.end is not None else run.today
    run.end = end
    if run.start is not None and run.start > end:
        raise ValueError(f"start {run.start} is after end {end}")
    days = [
        day
        for day in deps.research.dates()
        if day <= end and (run.start is None or day >= run.start)
    ]
    if run.start is None:
        run.start = min(days, default=None)
    results: list[ResearchResult] = []
    for day in days:
        result = deps.research.load(day)
        if result is None:
            raise ValueError(f"research for {day} is listed but could not be loaded")
        results.append(result)
    executed = {
        (fill.cycle_as_of, canonical_symbol(fill.ticker))
        for fill in deps.fills.load_all()
        if fill.filled_shares > 0
    }
    return results, executed


def _fetch(deps: VerifyDeps, run: _Verify, results: Sequence[ResearchResult]) -> _Market:
    """One ``closes`` call for every researched name and SPY, from ten calendar days before the
    earliest research date to today, reindexed to SPY's trading days (no forward fill)."""
    names = sorted(
        {canonical_symbol(name.candidate.ticker) for result in results for name in result.names}
    )
    first = min(result.as_of for result in results)
    frame = deps.data.closes(list(dict.fromkeys([*names, BENCHMARK])), first - _BASE_PAD, run.today)
    calendar = trading_days(frame[BENCHMARK], date.min, run.today)
    if not calendar:
        raise ValueError(f"no {BENCHMARK} closes up to {run.today}")
    return _Market(calendar, frame.reindex(pd.DatetimeIndex([pd.Timestamp(d) for d in calendar])))


def _observe(
    results: Sequence[ResearchResult],
    executed: set[tuple[date, str]],
    market: _Market,
) -> tuple[Observation, ...]:
    """Every researched name at every horizon, in research-date, shortlist, horizon order."""
    spy = market.closes[BENCHMARK]
    rows: list[Observation] = []
    for result in results:
        for name in result.names:
            ticker = canonical_symbol(name.candidate.ticker)
            done = (result.as_of, ticker) in executed
            closes = market.closes[ticker]
            rows.extend(
                observe(name, result.as_of, closes, spy, market.calendar, horizon, done)
                for horizon in HORIZONS
            )
    return tuple(rows)
