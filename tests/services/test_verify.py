"""Layer 7d verification over fakes: stages, the window, the executed join and the report."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from typing import Any

import pandas as pd
import pytest

from hedge_fund.domain.ledger import Fill, FillStatus
from hedge_fund.domain.orders import OrderAction
from hedge_fund.domain.research import NameResearch, ResearchResult, Verdict
from hedge_fund.domain.shortlist import Candidate, Side
from hedge_fund.domain.verification import (
    HORIZONS,
    Bucket,
    ExclusionReason,
    ObservationStatus,
    VerificationReport,
    VerifyOutcome,
)
from hedge_fund.services.verify import VerifyDeps, run_verify
from tests.factories import make_closes
from tests.fakes import (
    FakeProvider,
    InMemoryFillStore,
    InMemoryResearchStore,
    InMemoryVerificationStore,
)

NOW = datetime(2026, 9, 30, 22, 0, tzinfo=UTC)  # 18:00 in New York
LATE = datetime(2026, 10, 1, 1, 30, tzinfo=UTC)  # 21:30 on 09-30 in New York
AUG, SEP, END = date(2026, 8, 14), date(2026, 9, 1), date(2026, 9, 30)
FRAME = make_closes(["AAA", "BBB", "BRK-B", "CCC", "SPY"], n_days=80)
CANONICAL = ("AAA", "BBB", "BRK-B", "CCC")


def _name(
    ticker: str, side: Side, mean: float | None, verdict: Verdict, valid: int = 3
) -> NameResearch:
    return NameResearch(
        candidate=Candidate(ticker, "Tech", side, 1, 0.5),
        runs=(),
        valid=valid,
        mean=mean,
        std=None if mean is None else 0.0,
        agreement=None if mean is None else 1.0,
        verdict=verdict,
    )


SEP_NAMES = (
    _name("AAA", Side.LONG, 1.0, Verdict.CONFIRM),
    _name("BBB", Side.SHORT, 0.0, Verdict.NEUTRAL),
    _name("BRK.B", Side.LONG, -1.0, Verdict.CONTRADICT),
    _name("CCC", Side.LONG, None, Verdict.INSUFFICIENT, valid=1),
)
RESEARCH_SEP = ResearchResult(as_of=SEP, names=SEP_NAMES)
RESEARCH_AUG = ResearchResult(as_of=AUG, names=(_name("AAA", Side.LONG, 2.0, Verdict.CONFIRM),))


def _fill(cid: str, ticker: str, cycle: date, action: OrderAction, shares: int) -> Fill:
    at = datetime(2026, 9, 2, 13, 30, tzinfo=UTC)
    return Fill(
        client_order_id=cid,
        broker_order_id=f"b-{cid}",
        cycle_as_of=cycle,
        ticker=ticker,
        action=action,
        requested_shares=5,
        filled_shares=shares,
        avg_price=50.0 if shares else None,
        filled_at=at if shares else None,
        status=FillStatus.FILLED if shares else FillStatus.CANCELED,
        recorded_at=at,
    )


FILLS = (
    _fill("hf-a", "BRK-B", SEP, OrderAction.BUY, 5),  # executed (dash form joins BRK.B)
    _fill("hf-b", "AAA", date(2026, 9, 2), OrderAction.BUY, 5),  # another cycle date
    _fill("hf-c", "BBB", SEP, OrderAction.BUY_TO_COVER, 3),  # a cover fill counts
    _fill("hf-d", "AAA", SEP, OrderAction.BUY, 0),  # no shares filled
)


@dataclass
class World:
    """The fakes one verification runs over; every field may be replaced by a test."""

    research: Any = field(default_factory=lambda: InMemoryResearchStore([RESEARCH_SEP]))
    fills: Any = field(default_factory=lambda: InMemoryFillStore(FILLS))
    data: Any = field(default_factory=lambda: FakeProvider(FRAME, {}))
    reports: InMemoryVerificationStore = field(default_factory=InMemoryVerificationStore)

    @property
    def deps(self) -> VerifyDeps:
        return VerifyDeps(
            research=self.research, fills=self.fills, data=self.data, reports=self.reports
        )


def _verify(
    world: World,
    *,
    start: date | None = None,
    end: date | None = None,
    now: datetime = NOW,
) -> VerificationReport:
    return run_verify(world.deps, start=start, end=end, now=lambda: now)


def _by_key(report: VerificationReport) -> dict[tuple[str, int], Any]:
    return {(row.ticker, row.horizon): row for row in report.observations}


def _ret(ticker: str, first: date, last: date) -> float:
    column = FRAME[ticker]
    return float(column[pd.Timestamp(last)] / column[pd.Timestamp(first)] - 1.0)


# --- completed runs ----------------------------------------------------------------------


def test_a_clean_run_scores_every_name_at_every_horizon() -> None:
    world = World()
    report = _verify(world)
    assert report.outcome is VerifyOutcome.CLEAN
    assert (report.start, report.end, report.started_at) == (SEP, END, NOW)
    assert (report.failed_stage, report.error) == (None, None)
    assert [(row.ticker, row.horizon) for row in report.observations] == [
        (ticker, horizon) for ticker in ("AAA", "BBB", "BRK.B", "CCC") for horizon in HORIZONS
    ]
    rows = _by_key(report)
    for ticker in ("AAA", "BBB", "BRK.B"):
        assert (rows[ticker, 5].status, rows[ticker, 5].end_date) == (
            ObservationStatus.SCORED,
            date(2026, 9, 8),
        )
        assert (rows[ticker, 21].status, rows[ticker, 21].end_date) == (
            ObservationStatus.SCORED,
            END,  # exactly the last calendar date
        )
        assert (rows[ticker, 63].status, rows[ticker, 63].end_date) == (
            ObservationStatus.PENDING,
            None,
        )
    for horizon in HORIZONS:
        assert (rows["CCC", horizon].status, rows["CCC", horizon].reason) == (
            ObservationStatus.EXCLUDED,
            ExclusionReason.INSUFFICIENT_RESEARCH,
        )
    assert [(s.horizon, s.scored, s.pending, s.excluded) for s in report.summaries] == [
        (5, 3, 0, 1),
        (21, 3, 0, 1),
        (63, 0, 3, 1),
    ]
    assert world.reports.saved == [report]
    assert world.data.closes_calls == [((*CANONICAL, "SPY"), date(2026, 8, 22), END)]


def test_returns_and_hits_follow_the_definitions() -> None:
    rows = _by_key(_verify(World()))
    spy = _ret("SPY", SEP, date(2026, 9, 8))
    aaa = rows["AAA", 5]
    assert aaa.base_date == SEP
    assert aaa.name_return == pytest.approx(_ret("AAA", SEP, date(2026, 9, 8)))
    assert aaa.spy_return == pytest.approx(spy)
    assert aaa.aligned_excess == pytest.approx(aaa.name_return - spy)
    assert aaa.hit is (aaa.aligned_excess > 0)  # long, mean +1
    bbb = rows["BBB", 5]  # short, mean 0: no call
    assert bbb.aligned_excess == pytest.approx(-(_ret("BBB", SEP, date(2026, 9, 8)) - spy))
    assert bbb.hit is None
    brk = rows["BRK.B", 5]  # long, mean -1; looked up as BRK-B
    assert brk.name_return == pytest.approx(_ret("BRK-B", SEP, date(2026, 9, 8)))
    assert brk.hit is (brk.aligned_excess < 0)


def test_the_executed_flag_joins_on_cycle_date_and_canonical_ticker() -> None:
    rows = _by_key(_verify(World()))
    executed = {ticker: rows[ticker, 5].executed for ticker in ("AAA", "BBB", "BRK.B", "CCC")}
    assert executed == {"AAA": False, "BBB": True, "BRK.B": True, "CCC": False}


def test_a_name_without_a_price_is_a_gap_and_the_outcome_says_so() -> None:
    names = (*SEP_NAMES, _name("ZZZ", Side.LONG, 1.0, Verdict.CONFIRM))
    world = World(research=InMemoryResearchStore([ResearchResult(as_of=SEP, names=names)]))
    report = _verify(world)
    assert report.outcome is VerifyOutcome.GAPS
    rows = _by_key(report)
    assert (rows["ZZZ", 5].status, rows["ZZZ", 5].reason) == (
        ObservationStatus.EXCLUDED,
        ExclusionReason.NO_PRICE,
    )
    assert rows["ZZZ", 63].status is ObservationStatus.PENDING  # pending wins over no price
    assert world.data.closes_calls == [((*CANONICAL, "ZZZ", "SPY"), date(2026, 8, 22), END)]
    assert world.reports.saved == [report]


def test_today_is_the_new_york_date() -> None:
    world = World()
    report = _verify(world, now=LATE)  # 10-01 in UTC, 09-30 in New York
    assert report.end == END
    assert world.data.closes_calls[0][2] == END


def test_the_default_window_spans_all_the_research() -> None:
    world = World(research=InMemoryResearchStore([RESEARCH_SEP, RESEARCH_AUG]))
    report = _verify(world)
    assert (report.start, report.end) == (AUG, END)
    assert [row.as_of for row in report.observations] == [AUG] * 3 + [SEP] * 12
    assert world.data.closes_calls == [((*CANONICAL, "SPY"), date(2026, 8, 4), END)]


@pytest.mark.parametrize(
    ("window", "bounds", "dates"),
    [
        pytest.param({"start": SEP, "end": SEP}, (SEP, SEP), {SEP}, id="one-day"),
        pytest.param({"end": date(2026, 8, 31)}, (AUG, date(2026, 8, 31)), {AUG}, id="end-only"),
        pytest.param(
            {"start": date(2026, 8, 20)}, (date(2026, 8, 20), END), {SEP}, id="start-only"
        ),
    ],
)
def test_an_explicit_window_filters_the_research(
    window: dict[str, date], bounds: tuple[date, date], dates: set[date]
) -> None:
    world = World(research=InMemoryResearchStore([RESEARCH_SEP, RESEARCH_AUG]))
    report = _verify(world, **window)
    assert (report.start, report.end) == bounds
    assert {row.as_of for row in report.observations} == dates


@pytest.mark.parametrize(
    ("research", "window", "start"),
    [
        pytest.param(InMemoryResearchStore(), {}, None, id="no-research"),
        pytest.param(
            InMemoryResearchStore([RESEARCH_SEP]),
            {"start": date(2026, 9, 2)},
            date(2026, 9, 2),
            id="research-before-the-window",
        ),
    ],
)
def test_no_research_in_the_window_gives_zeroed_summaries_without_fetching(
    research: InMemoryResearchStore, window: dict[str, date], start: date | None
) -> None:
    world = World(research=research)
    report = _verify(world, **window)
    assert report.outcome is VerifyOutcome.CLEAN
    assert (report.start, report.end, report.observations) == (start, END, ())
    assert world.data.closes_calls == []
    assert [summary.horizon for summary in report.summaries] == list(HORIZONS)
    for summary in report.summaries:
        assert (summary.scored, summary.pending, summary.excluded) == (0, 0, 0)
        assert summary.overall == Bucket("all", 0, 0, None, None, None)
        assert summary.executed == Bucket("executed", 0, 0, None, None, None)
        assert (summary.by_verdict, summary.by_score, summary.by_sector) == ({}, {}, {})
        assert (summary.ic, summary.no_call_mean_abs_excess) == (None, None)
    assert world.reports.saved == [report]


def test_a_start_after_the_end_raises() -> None:
    with pytest.raises(ValueError, match="start 2026-09-30 is after end 2026-09-01"):
        _verify(World(), start=END, end=SEP)


def test_nan_is_not_used_to_build_the_calendar() -> None:
    frame = FRAME.copy()
    frame.loc[pd.Timestamp("2026-09-30"), "SPY"] = math.nan  # the last SPY row is missing
    report = _verify(World(data=FakeProvider(frame, {})))
    rows = _by_key(report)
    assert rows["AAA", 21].status is ObservationStatus.PENDING  # 09-30 is no longer a session
    assert rows["AAA", 5].status is ObservationStatus.SCORED
