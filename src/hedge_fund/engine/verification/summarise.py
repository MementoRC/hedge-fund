"""The per-horizon summary of verification observations. Pure."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import cast

from hedge_fund.domain.research import Verdict
from hedge_fund.domain.verification import (
    Bucket,
    HorizonSummary,
    Observation,
    ObservationStatus,
)
from hedge_fund.engine.verification.stats import make_bucket, score_key, spearman


def summarise(observations: Sequence[Observation], horizon: int) -> HorizonSummary:
    """The summary of the ``observations`` of ``horizon`` (others are ignored).

    PENDING and EXCLUDED observations are counted and left out of every statistic. The
    buckets are "all" and "executed" (SCORED rows with ``executed``), one per verdict
    (confirm, neutral, contradict that occur), per score key ("-2" .. "2", numeric order)
    and per sector (sorted). ``ic`` is the Spearman correlation of aligned mean and aligned
    excess; ``no_call_mean_abs_excess`` the mean ``|aligned_excess|`` of the ``mean == 0``
    rows. An empty input gives counts 0, empty buckets (n 0) and no statistics.
    """
    rows = [row for row in observations if row.horizon == horizon]
    scored = [row for row in rows if row.status is ObservationStatus.SCORED]
    pairs = [
        (row.mean, row.aligned_excess)
        for row in scored
        if row.mean is not None and row.aligned_excess is not None
    ]
    no_call = [abs(excess) for mean, excess in pairs if mean == 0]
    return HorizonSummary(
        horizon=horizon,
        scored=len(scored),
        pending=sum(1 for row in rows if row.status is ObservationStatus.PENDING),
        excluded=sum(1 for row in rows if row.status is ObservationStatus.EXCLUDED),
        overall=make_bucket("all", scored),
        by_verdict=_buckets(scored, lambda row: row.verdict.value, [v.value for v in Verdict]),
        by_score=_buckets(scored, _score_label, sorted({_score_label(r) for r in scored}, key=int)),
        by_sector=_buckets(scored, lambda row: row.sector, sorted({row.sector for row in scored})),
        executed=make_bucket("executed", [row for row in scored if row.executed]),
        ic=spearman([mean for mean, _ in pairs], [excess for _, excess in pairs]),
        no_call_mean_abs_excess=math.fsum(no_call) / len(no_call) if no_call else None,
    )


def _score_label(row: Observation) -> str:
    # a SCORED observation always carries a mean
    return str(score_key(cast(float, row.mean)))


def _buckets(
    scored: Sequence[Observation],
    label: Callable[[Observation], str],
    order: Sequence[str],
) -> dict[str, Bucket]:
    """One Bucket per label in ``order`` that has rows, in that order."""
    groups: dict[str, list[Observation]] = {}
    for row in scored:
        groups.setdefault(label(row), []).append(row)
    return {key: make_bucket(key, groups[key]) for key in order if key in groups}
