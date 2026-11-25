"""Tests for the verification statistics: Spearman IC, score keys and buckets."""

from __future__ import annotations

import pytest

from hedge_fund.domain.verification import Bucket
from hedge_fund.engine.verification.stats import make_bucket, score_key, spearman
from tests.factories import make_observation

# --- score_key -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("mean", "key"),
    [
        (0.0, 0),
        (0.49, 0),
        (-0.49, 0),
        (0.5, 1),
        (-0.5, -1),
        (1.4999, 1),
        (1.5, 2),
        (-1.5, -2),
        (2.0, 2),
        (-2.0, -2),
    ],
)
def test_score_key_rounds_half_away_from_zero(mean: float, key: int) -> None:
    assert score_key(mean) == key


# --- spearman ------------------------------------------------------------------------------


def test_spearman_of_a_perfect_monotone_relation() -> None:
    assert spearman([1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 45.0]) == pytest.approx(1.0)


def test_spearman_of_a_reversed_relation() -> None:
    assert spearman([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]) == pytest.approx(-1.0)


def test_spearman_uses_average_ranks_for_ties() -> None:
    # ranks of x: 1, 2.5, 2.5, 4; ranks of y: 1, 2, 3, 4; r = 4.5 / sqrt(4.5 * 5)
    assert spearman([1.0, 2.0, 2.0, 3.0], [1.0, 2.0, 3.0, 4.0]) == pytest.approx(0.9**0.5)


@pytest.mark.parametrize(
    ("xs", "ys"),
    [([], []), ([1.0], [2.0]), ([1.0, 2.0], [2.0, 1.0])],
    ids=["empty", "one", "two"],
)
def test_spearman_is_none_for_fewer_than_three_pairs(xs: list[float], ys: list[float]) -> None:
    assert spearman(xs, ys) is None


@pytest.mark.parametrize(
    ("xs", "ys"),
    [
        ([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]),
        ([1.0, 2.0, 3.0], [5.0, 5.0, 5.0]),
        ([1.0, 1.0, 1.0], [5.0, 5.0, 5.0]),
    ],
    ids=["constant-x", "constant-y", "both"],
)
def test_spearman_is_none_when_a_side_is_constant(xs: list[float], ys: list[float]) -> None:
    assert spearman(xs, ys) is None


def test_spearman_refuses_unequal_lengths() -> None:
    with pytest.raises(ValueError, match="2 values against 3"):
        spearman([1.0, 2.0], [1.0, 2.0, 3.0])


# --- make_bucket ---------------------------------------------------------------------------


def test_bucket_arithmetic() -> None:
    scored = [
        make_observation(aligned_excess=0.10, hit=True),
        make_observation(aligned_excess=-0.04, hit=False),
        make_observation(aligned_excess=0.02, hit=None),
    ]
    bucket = make_bucket("all", scored)
    assert (bucket.key, bucket.n, bucket.hits) == ("all", 3, 1)
    assert bucket.hit_rate == pytest.approx(0.5)  # 1 hit of the 2 with a call
    assert bucket.mean_excess == pytest.approx(0.08 / 3)
    assert bucket.median_excess == pytest.approx(0.02)


def test_an_empty_bucket_has_no_statistics() -> None:
    assert make_bucket("none", []) == Bucket("none", 0, 0, None, None, None)


def test_a_bucket_of_no_calls_has_no_hit_rate() -> None:
    scored = [
        make_observation(mean=0.0, aligned_excess=0.02, hit=None),
        make_observation(mean=0.0, aligned_excess=-0.06, hit=None),
    ]
    bucket = make_bucket("0", scored)
    assert (bucket.n, bucket.hits, bucket.hit_rate) == (2, 0, None)
    assert bucket.mean_excess == pytest.approx(-0.02)
    assert bucket.median_excess == pytest.approx(-0.02)


def test_a_bucket_of_misses_has_a_zero_hit_rate() -> None:
    bucket = make_bucket("x", [make_observation(aligned_excess=-0.01, hit=False)])
    assert (bucket.n, bucket.hits, bucket.hit_rate) == (1, 0, 0.0)
