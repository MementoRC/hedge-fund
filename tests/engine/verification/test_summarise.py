"""Tests for summarise: the per-horizon buckets, the IC and the left-out observations."""

from __future__ import annotations

import pytest

import hedge_fund.engine.verification as verification
from hedge_fund.domain.research import Verdict
from hedge_fund.domain.verification import (
    Bucket,
    ExclusionReason,
    HorizonSummary,
    Observation,
    ObservationStatus,
)
from hedge_fund.engine.verification import summarise
from tests.factories import make_observation


def _scored(
    verdict: Verdict,
    mean: float,
    sector: str,
    aligned: float,
    hit: bool | None,
    executed: bool = False,
    horizon: int = 5,
) -> Observation:
    return make_observation(
        verdict=verdict,
        mean=mean,
        sector=sector,
        aligned_excess=aligned,
        hit=hit,
        executed=executed,
        horizon=horizon,
    )


SCORED = [
    _scored(Verdict.CONFIRM, 2.0, "Tech", 0.10, True, executed=True),
    _scored(Verdict.CONFIRM, 1.0, "Tech", -0.04, False),
    _scored(Verdict.NEUTRAL, 0.0, "Energy", 0.02, None),
    _scored(Verdict.CONTRADICT, -1.0, "Energy", -0.06, True, executed=True),
]
PENDING = make_observation(
    status=ObservationStatus.PENDING,
    end_date=None,
    name_return=None,
    spy_return=None,
    aligned_excess=None,
    hit=None,
)
NO_PRICE = make_observation(
    status=ObservationStatus.EXCLUDED,
    reason=ExclusionReason.NO_PRICE,
    name_return=None,
    spy_return=None,
    aligned_excess=None,
    hit=None,
)
INSUFFICIENT = make_observation(
    status=ObservationStatus.EXCLUDED,
    reason=ExclusionReason.INSUFFICIENT_RESEARCH,
    verdict=Verdict.INSUFFICIENT,
    mean=None,
    end_date=None,
    name_return=None,
    spy_return=None,
    aligned_excess=None,
    hit=None,
)
ELSEWHERE = _scored(Verdict.CONFIRM, 2.0, "Tech", 0.90, True, executed=True, horizon=21)
EVERYTHING = [*SCORED, PENDING, NO_PRICE, INSUFFICIENT, ELSEWHERE]


def _close(actual: float | None, expected: float | None) -> None:
    if expected is None:
        assert actual is None
    else:
        assert actual == pytest.approx(expected)


def _check(
    bucket: Bucket,
    n: int,
    hits: int,
    rate: float | None,
    mean: float | None,
    median: float | None,
) -> None:
    assert (bucket.n, bucket.hits) == (n, hits)
    _close(bucket.hit_rate, rate)
    _close(bucket.mean_excess, mean)
    _close(bucket.median_excess, median)


def test_counts_cover_the_horizon_only() -> None:
    summary = summarise(EVERYTHING, 5)
    assert (summary.horizon, summary.scored, summary.pending, summary.excluded) == (5, 4, 1, 2)


def test_overall() -> None:
    summary = summarise(EVERYTHING, 5)
    assert summary.overall.key == "all"
    _check(summary.overall, 4, 2, 2 / 3, 0.005, -0.01)  # 2 hits of the 3 calls


def test_by_verdict_has_only_the_scored_verdicts() -> None:
    by_verdict = summarise(EVERYTHING, 5).by_verdict
    assert list(by_verdict) == ["confirm", "neutral", "contradict"]
    _check(by_verdict["confirm"], 2, 1, 0.5, 0.03, 0.03)
    _check(by_verdict["neutral"], 1, 0, None, 0.02, 0.02)
    _check(by_verdict["contradict"], 1, 1, 1.0, -0.06, -0.06)
    assert by_verdict["confirm"].key == "confirm"


def test_by_score_is_keyed_by_the_rounded_aligned_mean_in_numeric_order() -> None:
    by_score = summarise(EVERYTHING, 5).by_score
    assert list(by_score) == ["-1", "0", "1", "2"]
    _check(by_score["0"], 1, 0, None, 0.02, 0.02)
    _check(by_score["2"], 1, 1, 1.0, 0.10, 0.10)
    _check(by_score["-1"], 1, 1, 1.0, -0.06, -0.06)


def test_by_sector_is_sorted() -> None:
    by_sector = summarise(EVERYTHING, 5).by_sector
    assert list(by_sector) == ["Energy", "Tech"]
    _check(by_sector["Energy"], 2, 1, 1.0, -0.02, -0.02)
    _check(by_sector["Tech"], 2, 1, 0.5, 0.03, 0.03)


def test_executed_bucket_counts_only_executed_scored_rows() -> None:
    summary = summarise(EVERYTHING, 5)
    assert summary.executed.key == "executed"
    _check(summary.executed, 2, 2, 1.0, 0.02, 0.02)


def test_ic_is_the_spearman_of_aligned_mean_and_aligned_excess() -> None:
    # means rank 4, 3, 2, 1 and excess rank 4, 2, 3, 1: r = 4 / 5
    _close(summarise(EVERYTHING, 5).ic, 0.8)


def test_no_call_mean_abs_excess_is_over_the_zero_mean_rows() -> None:
    _close(summarise(EVERYTHING, 5).no_call_mean_abs_excess, 0.02)


def test_another_horizon_is_summarised_on_its_own() -> None:
    summary = summarise(EVERYTHING, 21)
    assert (summary.scored, summary.pending, summary.excluded) == (1, 0, 0)
    _check(summary.overall, 1, 1, 1.0, 0.90, 0.90)


def test_an_empty_input_is_a_zeroed_summary() -> None:
    zero = Bucket("all", 0, 0, None, None, None)
    assert summarise([], 63) == HorizonSummary(
        horizon=63,
        scored=0,
        pending=0,
        excluded=0,
        overall=zero,
        by_verdict={},
        by_score={},
        by_sector={},
        executed=Bucket("executed", 0, 0, None, None, None),
        ic=None,
        no_call_mean_abs_excess=None,
    )


def test_pending_and_excluded_alone_leave_every_statistic_empty() -> None:
    summary = summarise([PENDING, NO_PRICE, INSUFFICIENT], 5)
    assert (summary.scored, summary.pending, summary.excluded) == (0, 1, 2)
    assert summary.overall.n == 0
    assert (summary.by_verdict, summary.by_score, summary.by_sector) == ({}, {}, {})


def test_ic_and_the_no_call_figure_are_none_without_the_data() -> None:
    summary = summarise(SCORED[:2], 5)
    assert summary.ic is None  # two observations
    assert summary.no_call_mean_abs_excess is None  # no mean-0 row


def test_package_exports() -> None:
    assert verification.__all__ == ["make_bucket", "observe", "score_key", "spearman", "summarise"]
