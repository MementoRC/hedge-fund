"""Tests for the research service."""

import time
from datetime import date

import pytest

from hedge_fund.domain.research import Rating, RunRecord, Verdict
from hedge_fund.domain.shortlist import Candidate, Shortlist, Side
from hedge_fund.engine.research.conviction import ConvictionParams
from hedge_fund.services.research import _previous_weekday, research_shortlist
from tests.fakes import ScriptedResearcher

AS_OF = date(2026, 9, 30)  # Wednesday
TODAY = date(2026, 9, 30)


def _candidate(ticker: str, side: Side = Side.LONG) -> Candidate:
    return Candidate(ticker=ticker, sector="Tech", side=side, depth=1, composite=1.0)


def _shortlist(*candidates: Candidate, as_of: date = AS_OF) -> Shortlist:
    return Shortlist(as_of=as_of, candidates=candidates, trimmed={})


# --- _previous_weekday -----------------------------------------------------


@pytest.mark.parametrize(
    ("today", "expected"),
    [
        (date(2026, 10, 6), date(2026, 10, 5)),  # Tuesday -> Monday
        (date(2026, 10, 2), date(2026, 10, 1)),  # Friday -> Thursday
        (date(2026, 10, 3), date(2026, 10, 2)),  # Saturday -> Friday
        (date(2026, 10, 4), date(2026, 10, 2)),  # Sunday -> Friday
        (date(2026, 10, 5), date(2026, 10, 2)),  # Monday -> Friday
    ],
)
def test_previous_weekday(today: date, expected: date) -> None:
    assert _previous_weekday(today) == expected


# --- orchestration ---------------------------------------------------------


def test_names_follow_shortlist_order_and_runs_k_times() -> None:
    researcher = ScriptedResearcher(
        {
            ("A", 0): Rating.BUY,
            ("A", 1): Rating.BUY,
            ("B", 0): Rating.SELL,
            ("B", 1): Rating.SELL,
        }
    )
    shortlist = _shortlist(_candidate("B", Side.SHORT), _candidate("A"))
    out = research_shortlist(researcher, shortlist, k=2, max_workers=2, today=TODAY)
    assert out.as_of == AS_OF
    assert [name.candidate.ticker for name in out.names] == ["B", "A"]
    assert [name.verdict for name in out.names] == [Verdict.CONFIRM, Verdict.CONFIRM]
    assert [[run.run_index for run in name.runs] for name in out.names] == [[0, 1], [0, 1]]
    assert sorted(researcher.calls) == [
        ("A", AS_OF, 0),
        ("A", AS_OF, 1),
        ("B", AS_OF, 0),
        ("B", AS_OF, 1),
    ]


def test_default_k_is_three() -> None:
    researcher = ScriptedResearcher()
    research_shortlist(researcher, _shortlist(_candidate("A")), today=TODAY)
    assert len(researcher.calls) == 3


def test_params_are_passed_to_conviction() -> None:
    researcher = ScriptedResearcher({("A", 0): Rating.BUY})
    out = research_shortlist(
        researcher,
        _shortlist(_candidate("A")),
        k=1,
        params=ConvictionParams(min_valid=1),
        today=TODAY,
    )
    assert out.names[0].verdict is Verdict.CONFIRM


def test_exception_becomes_error_record() -> None:
    researcher = ScriptedResearcher({("A", 0): RuntimeError("boom")})
    out = research_shortlist(researcher, _shortlist(_candidate("A")), k=2, today=TODAY)
    failed, ok = out.names[0].runs
    assert failed.rating is None
    assert failed.error == "RuntimeError: boom"
    assert failed.decision == ""
    assert failed.reports == {}
    assert failed.seconds == 0.0
    assert ok.valid
    assert out.names[0].valid == 1
    assert out.names[0].verdict is Verdict.INSUFFICIENT


def test_all_runs_failing_is_all_insufficient() -> None:
    researcher = ScriptedResearcher(
        {("A", 0): ValueError("x"), ("A", 1): ValueError("y"), ("B", 0): OSError("z")}
    )
    shortlist = _shortlist(_candidate("A"), _candidate("B"))
    out = research_shortlist(researcher, shortlist, k=2, today=TODAY)
    assert out.names[0].valid == 0
    assert out.names[0].mean is None
    assert out.names[1].runs[0].error == "OSError: z"
    assert [name.verdict for name in out.names] == [Verdict.INSUFFICIENT, Verdict.INSUFFICIENT]


def test_keyboard_interrupt_propagates() -> None:
    researcher = ScriptedResearcher({("A", 1): KeyboardInterrupt()})
    with pytest.raises(KeyboardInterrupt):
        research_shortlist(researcher, _shortlist(_candidate("A")), k=2, today=TODAY)


class _InterruptingResearcher:
    """First call raises KeyboardInterrupt; later calls are slow so queued runs stay queued."""

    def __init__(self) -> None:
        self.calls: list[int] = []

    def run(self, ticker: str, as_of: date, run_index: int) -> RunRecord:
        self.calls.append(run_index)
        if len(self.calls) == 1:
            raise KeyboardInterrupt
        time.sleep(0.05)
        return RunRecord(
            ticker=ticker,
            as_of=as_of,
            run_index=run_index,
            rating=Rating.HOLD,
            decision="",
            reports={},
            seconds=0.0,
            error=None,
        )


def test_keyboard_interrupt_cancels_queued_runs() -> None:
    researcher = _InterruptingResearcher()
    shortlist = _shortlist(_candidate("A"), _candidate("B"))
    with pytest.raises(KeyboardInterrupt):
        research_shortlist(researcher, shortlist, k=3, max_workers=1, today=TODAY)
    assert len(researcher.calls) < 6


def test_empty_shortlist_never_calls_the_researcher() -> None:
    researcher = ScriptedResearcher()
    out = research_shortlist(researcher, _shortlist(), today=TODAY)
    assert out.as_of == AS_OF
    assert out.names == ()
    assert researcher.calls == []


# --- validation ------------------------------------------------------------


@pytest.mark.parametrize("field", ["k", "max_workers"])
@pytest.mark.parametrize("value", [0, -1, 1.5, True])
def test_rejects_invalid_counts(field: str, value: object) -> None:
    with pytest.raises(ValueError, match=f"{field} must be an int >= 1"):
        research_shortlist(
            ScriptedResearcher(),
            _shortlist(_candidate("A")),
            today=TODAY,
            **{field: value},  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("as_of", "today"),
    [
        (date(2026, 9, 30), date(2026, 9, 30)),  # same day
        (date(2026, 9, 29), date(2026, 9, 30)),  # previous weekday
        (date(2026, 10, 16), date(2026, 10, 17)),  # Friday run on Saturday
        (date(2026, 10, 16), date(2026, 10, 18)),  # Friday run on Sunday
        (date(2026, 10, 16), date(2026, 10, 19)),  # Friday run on Monday
    ],
)
def test_live_guard_accepts_recent_dates(as_of: date, today: date) -> None:
    researcher = ScriptedResearcher()
    out = research_shortlist(researcher, _shortlist(_candidate("A"), as_of=as_of), today=today)
    assert out.as_of == as_of


@pytest.mark.parametrize(
    ("as_of", "today"),
    [
        (date(2026, 10, 1), date(2026, 9, 30)),  # future
        (date(2026, 9, 28), date(2026, 9, 30)),  # older than the previous weekday
        (date(2026, 10, 16), date(2026, 10, 20)),  # Friday run on Tuesday
    ],
)
def test_live_guard_rejects_other_dates(as_of: date, today: date) -> None:
    researcher = ScriptedResearcher()
    with pytest.raises(ValueError, match="as_of"):
        research_shortlist(researcher, _shortlist(_candidate("A"), as_of=as_of), today=today)
    assert researcher.calls == []
