"""Tests for the research domain types."""

import dataclasses
from datetime import date

import pytest

from hedge_fund.domain.research import NameResearch, Rating, ResearchResult, RunRecord, Verdict
from hedge_fund.domain.shortlist import Candidate, Side

AS_OF = date(2026, 9, 30)


def _run(rating: Rating | None, error: str | None = None) -> RunRecord:
    return RunRecord(
        ticker="A",
        as_of=AS_OF,
        run_index=0,
        rating=rating,
        decision="d",
        reports={},
        seconds=1.0,
        error=error,
    )


def test_rating_values_are_plain_strings() -> None:
    assert [str(rating) for rating in Rating] == [
        "Buy",
        "Overweight",
        "Hold",
        "Underweight",
        "Sell",
        "REVIEW",
    ]


def test_rating_scores() -> None:
    assert [rating.score for rating in Rating] == [2, 1, 0, -1, -2, None]


def test_verdict_values_are_plain_strings() -> None:
    assert [str(verdict) for verdict in Verdict] == [
        "confirm",
        "neutral",
        "contradict",
        "insufficient",
    ]


@pytest.mark.parametrize(
    ("rating", "error", "expected"),
    [
        (Rating.BUY, None, True),
        (Rating.SELL, None, True),
        (Rating.REVIEW, None, False),
        (None, None, False),
        (Rating.BUY, "RuntimeError: boom", False),
    ],
)
def test_run_record_valid(rating: Rating | None, error: str | None, expected: bool) -> None:
    assert _run(rating, error).valid is expected


def test_research_types_are_frozen() -> None:
    run = _run(Rating.BUY)
    candidate = Candidate(ticker="A", sector="Tech", side=Side.LONG, depth=1, composite=1.5)
    name = NameResearch(
        candidate=candidate,
        runs=(run,),
        valid=1,
        mean=2.0,
        std=0.0,
        agreement=1.0,
        verdict=Verdict.CONFIRM,
    )
    result = ResearchResult(as_of=AS_OF, names=(name,))
    with pytest.raises(dataclasses.FrozenInstanceError):
        run.seconds = 2.0  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        name.valid = 2  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.names = ()  # type: ignore[misc]
