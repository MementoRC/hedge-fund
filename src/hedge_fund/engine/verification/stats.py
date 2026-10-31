"""Statistics for verification: rank correlation, score keys and buckets. Pure, no scipy."""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence

from hedge_fund.domain.verification import Bucket, Observation


def score_key(mean: float) -> int:
    """``mean`` rounded half away from zero: 0.5 -> 1, -0.5 -> -1, 1.5 -> 2, 0.49 -> 0."""
    return int(math.copysign(math.floor(abs(mean) + 0.5), mean))


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    """Spearman rank correlation, ties given their average rank.

    None for fewer than 3 pairs or when either side is constant. ValueError when the
    sequences differ in length.
    """
    if len(xs) != len(ys):
        raise ValueError(f"{len(xs)} values against {len(ys)}")
    if len(xs) < 3:
        return None
    return _pearson(_ranks(xs), _ranks(ys))


def make_bucket(key: str, scored: Sequence[Observation]) -> Bucket:
    """The Bucket of SCORED observations: size, hits among those with a call, hit rate, and
    mean and median ``aligned_excess``."""
    excess = [row.aligned_excess for row in scored if row.aligned_excess is not None]
    called = [row for row in scored if row.hit is not None]
    hits = sum(1 for row in called if row.hit)
    return Bucket(
        key=key,
        n=len(scored),
        hits=hits,
        hit_rate=hits / len(called) if called else None,
        mean_excess=math.fsum(excess) / len(excess) if excess else None,
        median_excess=statistics.median(excess) if excess else None,
    )


def _ranks(values: Sequence[float]) -> list[float]:
    """1-based ranks, equal values sharing the average of the ranks they span."""
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    first = 0
    while first < len(order):
        last = first
        while last + 1 < len(order) and values[order[last + 1]] == values[order[first]]:
            last += 1
        average = (first + last) / 2 + 1
        for position in range(first, last + 1):
            ranks[order[position]] = average
        first = last + 1
    return ranks


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    """Pearson correlation clamped to [-1, 1]; None when either side has no variance."""
    mean_x, mean_y = math.fsum(xs) / len(xs), math.fsum(ys) / len(ys)
    sxx = math.fsum((x - mean_x) ** 2 for x in xs)
    syy = math.fsum((y - mean_y) ** 2 for y in ys)
    if sxx == 0 or syy == 0:
        return None
    sxy = math.fsum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True))
    return max(-1.0, min(1.0, sxy / math.sqrt(sxx * syy)))
