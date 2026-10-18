"""Tests for the conviction engine."""

import dataclasses
import math
from datetime import date

import pytest

from hedge_fund.domain.research import Rating, RunRecord, Verdict
from hedge_fund.domain.shortlist import Candidate, Side
from hedge_fund.engine.research.conviction import (
    DEFAULT_CONVICTION,
    ConvictionParams,
    conviction,
)

AS_OF = date(2026, 9, 30)
NAN = float("nan")
INF = float("inf")


def _candidate(side: Side = Side.LONG) -> Candidate:
    return Candidate(ticker="A", sector="Tech", side=side, depth=1, composite=1.0)


def _run(
    index: int,
    rating: Rating | None,
    *,
    error: str | None = None,
    ticker: str = "A",
    as_of: date = AS_OF,
) -> RunRecord:
    return RunRecord(
        ticker=ticker,
        as_of=as_of,
        run_index=index,
        rating=rating,
        decision="",
        reports={},
        seconds=1.0,
        error=error,
    )


def _runs(*ratings: Rating | None) -> tuple[RunRecord, ...]:
    return tuple(_run(i, rating) for i, rating in enumerate(ratings))


# --- params ----------------------------------------------------------------


def test_default_params() -> None:
    expected = ConvictionParams(min_valid=2, confirm=0.5, contradict=-0.5, max_std=1.0)
    assert expected == DEFAULT_CONVICTION


@pytest.mark.parametrize("value", [0, -1, 1.5, NAN, True])
def test_rejects_invalid_min_valid(value: object) -> None:
    params = dataclasses.replace(DEFAULT_CONVICTION, min_valid=value)
    with pytest.raises(ValueError, match="min_valid must be an int >= 1"):
        conviction(_candidate(), (), params=params)


@pytest.mark.parametrize("field", ["confirm", "contradict", "max_std"])
@pytest.mark.parametrize("value", [NAN, INF, -INF, True, "x"])
def test_rejects_non_finite_or_non_numeric(field: str, value: object) -> None:
    params = dataclasses.replace(DEFAULT_CONVICTION, **{field: value})
    with pytest.raises(ValueError, match=f"{field} must be a finite number"):
        conviction(_candidate(), (), params=params)


@pytest.mark.parametrize(
    "overrides",
    [{"confirm": 0.0}, {"confirm": -1.0}, {"contradict": 0.0}, {"contradict": 0.5}],
)
def test_rejects_misordered_thresholds(overrides: dict[str, float]) -> None:
    params = dataclasses.replace(DEFAULT_CONVICTION, **overrides)
    with pytest.raises(ValueError, match="contradict < 0 < confirm"):
        conviction(_candidate(), (), params=params)


@pytest.mark.parametrize("value", [0.0, -1.0])
def test_rejects_non_positive_max_std(value: float) -> None:
    params = dataclasses.replace(DEFAULT_CONVICTION, max_std=value)
    with pytest.raises(ValueError, match="max_std must be > 0"):
        conviction(_candidate(), (), params=params)


# --- maths and verdict -----------------------------------------------------


def test_params_accept_int_thresholds() -> None:
    params = ConvictionParams(min_valid=1, confirm=1, contradict=-1, max_std=2)
    out = conviction(_candidate(), _runs(Rating.BUY), params=params)
    assert out.verdict is Verdict.CONFIRM


def test_long_stats() -> None:
    # aligned scores [2, 0, -1, 1]: mean 0.5, population variance 5/4, 2 of 4 positive
    out = conviction(
        _candidate(), _runs(Rating.BUY, Rating.HOLD, Rating.UNDERWEIGHT, Rating.OVERWEIGHT)
    )
    assert out.valid == 4
    assert out.mean == pytest.approx(0.5)
    assert out.std == pytest.approx(math.sqrt(1.25))
    assert out.agreement == pytest.approx(0.5)
    assert out.verdict is Verdict.NEUTRAL  # mean >= 0.5 but std > 1.0


def test_short_sign_flip_confirms_on_sells() -> None:
    out = conviction(_candidate(Side.SHORT), _runs(Rating.SELL, Rating.SELL))
    assert out.mean == pytest.approx(2.0)
    assert out.std == pytest.approx(0.0)
    assert out.agreement == pytest.approx(1.0)
    assert out.verdict is Verdict.CONFIRM


def test_short_sign_flip_contradicts_on_buys() -> None:
    out = conviction(_candidate(Side.SHORT), _runs(Rating.BUY, Rating.BUY))
    assert out.mean == pytest.approx(-2.0)
    assert out.agreement == pytest.approx(0.0)
    assert out.verdict is Verdict.CONTRADICT


def test_review_and_error_runs_are_excluded() -> None:
    runs = (
        _run(0, Rating.BUY),
        _run(1, Rating.BUY, error="RuntimeError: boom"),
        _run(2, Rating.REVIEW),
        _run(3, None),
        _run(4, Rating.BUY),
    )
    out = conviction(_candidate(), runs)
    assert out.valid == 2
    assert out.mean == pytest.approx(2.0)
    assert out.verdict is Verdict.CONFIRM
    assert out.runs == runs


def test_runs_are_sorted_by_run_index() -> None:
    shuffled = (_run(2, Rating.HOLD), _run(0, Rating.BUY), _run(1, Rating.BUY))
    out = conviction(_candidate(), shuffled)
    assert [run.run_index for run in out.runs] == [0, 1, 2]


def test_confirm_threshold_is_inclusive() -> None:
    # aligned [1, 0]: mean exactly 0.5, std 0.5
    out = conviction(_candidate(), _runs(Rating.OVERWEIGHT, Rating.HOLD))
    expected = 0.5
    assert out.mean == expected
    assert out.verdict is Verdict.CONFIRM


def test_contradict_threshold_is_inclusive() -> None:
    # aligned [-1, 0]: mean exactly -0.5, std 0.5
    out = conviction(_candidate(), _runs(Rating.UNDERWEIGHT, Rating.HOLD))
    expected = -0.5
    assert out.mean == expected
    assert out.verdict is Verdict.CONTRADICT


def test_just_below_confirm_is_neutral() -> None:
    # aligned [1, 0, 0]: mean 1/3, std sqrt(2)/3
    out = conviction(_candidate(), _runs(Rating.OVERWEIGHT, Rating.HOLD, Rating.HOLD))
    assert out.mean == pytest.approx(1 / 3)
    assert out.std == pytest.approx(math.sqrt(2) / 3)
    assert out.verdict is Verdict.NEUTRAL


def test_max_std_is_inclusive() -> None:
    # aligned [2, 0]: mean 1.0, std exactly 1.0
    out = conviction(_candidate(), _runs(Rating.BUY, Rating.HOLD))
    expected = 1.0
    assert out.std == expected
    assert out.verdict is Verdict.CONFIRM


def test_dispersion_downgrades_to_neutral() -> None:
    # aligned [2, 2, -2, 2]: mean 1.0, variance 12/4 = 3, std sqrt(3) > 1.0
    out = conviction(_candidate(), _runs(Rating.BUY, Rating.BUY, Rating.SELL, Rating.BUY))
    assert out.mean == pytest.approx(1.0)
    assert out.std == pytest.approx(math.sqrt(3))
    assert out.verdict is Verdict.NEUTRAL


def test_dispersion_downgrades_a_contradiction_too() -> None:
    # short with ratings [2, 2, -2, 2] -> aligned [-2, -2, 2, -2]: mean -1.0, std sqrt(3)
    out = conviction(_candidate(Side.SHORT), _runs(Rating.BUY, Rating.BUY, Rating.SELL, Rating.BUY))
    assert out.mean == pytest.approx(-1.0)
    assert out.verdict is Verdict.NEUTRAL


def test_fewer_than_min_valid_is_insufficient_but_keeps_stats() -> None:
    out = conviction(_candidate(), _runs(Rating.BUY, Rating.REVIEW))
    assert out.valid == 1
    assert out.mean == pytest.approx(2.0)
    assert out.std == pytest.approx(0.0)
    assert out.agreement == pytest.approx(1.0)
    assert out.verdict is Verdict.INSUFFICIENT


def test_no_valid_runs_gives_none_stats() -> None:
    out = conviction(_candidate(), _runs(Rating.REVIEW, None))
    assert out.valid == 0
    assert (out.mean, out.std, out.agreement) == (None, None, None)
    assert out.verdict is Verdict.INSUFFICIENT


def test_empty_runs_are_insufficient() -> None:
    out = conviction(_candidate(), ())
    assert out.runs == ()
    assert out.valid == 0
    assert out.verdict is Verdict.INSUFFICIENT


def test_candidate_is_carried_through() -> None:
    candidate = _candidate(Side.SHORT)
    assert conviction(candidate, ()).candidate == candidate


# --- input validation ------------------------------------------------------


def test_rejects_ticker_mismatch() -> None:
    with pytest.raises(ValueError, match="expected 'A'"):
        conviction(_candidate(), (_run(0, Rating.BUY, ticker="B"),))


def test_rejects_as_of_mismatch() -> None:
    runs = (_run(0, Rating.BUY), _run(1, Rating.BUY, as_of=date(2026, 9, 29)))
    with pytest.raises(ValueError, match="as_of"):
        conviction(_candidate(), runs)


def test_rejects_duplicate_run_index() -> None:
    runs = (_run(0, Rating.BUY), _run(0, Rating.HOLD))
    with pytest.raises(ValueError, match="duplicate run_index 0"):
        conviction(_candidate(), runs)
