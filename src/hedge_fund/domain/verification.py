"""Thesis and rating verification types (Layer 7d): one research rating scored against the
excess return over SPY that followed it, bucketed statistics per horizon, and the
verification report.

None of these types validate themselves.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime
from enum import StrEnum

from hedge_fund.domain.research import Verdict
from hedge_fund.domain.shortlist import Side

HORIZONS: tuple[int, ...] = (5, 21, 63)


class ObservationStatus(StrEnum):
    """SCORED has returns and (for a directional call) a hit; PENDING's horizon is not reached
    yet; EXCLUDED has a ``reason``."""

    SCORED = "scored"
    PENDING = "pending"
    EXCLUDED = "excluded"


class ExclusionReason(StrEnum):
    """Why an observation is EXCLUDED: the name has no usable close at the base or end date,
    or the research was INSUFFICIENT (too few valid runs, or no mean)."""

    NO_PRICE = "no_price"
    INSUFFICIENT_RESEARCH = "insufficient_research"


@dataclass(frozen=True)
class Observation:
    """One (research ``as_of``, ticker, ``horizon``).

    ``mean``, ``std``, ``agreement``, ``valid``, ``verdict`` and ``side`` are the name's
    ``NameResearch`` figures (``mean`` is side-aligned). ``base_date`` is the last SPY trading
    day on or before ``as_of`` (None only for INSUFFICIENT_RESEARCH exclusions, which look
    nothing up); ``end_date`` is ``horizon`` SPY sessions later (None while PENDING or for
    insufficient research). ``aligned_excess = side * (name_return -
    spy_return)`` with side +1 LONG, -1 SHORT. ``hit`` is None without a directional call
    (``mean == 0``) or unless SCORED; a zero ``aligned_excess`` with a call is a miss.
    ``executed``: the 7a ledger has a fill with shares for the ticker on that research date.
    """

    as_of: date
    ticker: str
    sector: str
    side: Side
    verdict: Verdict
    mean: float | None
    std: float | None
    agreement: float | None
    valid: int
    horizon: int
    base_date: date | None
    end_date: date | None
    name_return: float | None
    spy_return: float | None
    aligned_excess: float | None
    hit: bool | None
    executed: bool
    status: ObservationStatus
    reason: ExclusionReason | None


@dataclass(frozen=True)
class Bucket:
    """Statistics of a group of SCORED observations.

    ``n`` counts the group; ``hits`` counts the hits among those with ``hit is not None`` and
    ``hit_rate`` is ``hits`` over their count (None when there are none). ``mean_excess`` and
    ``median_excess`` are over ``aligned_excess``; all three statistics are None for an empty
    group. ``key`` is "all" / "executed", a verdict value, a score ("-2" .. "2") or a sector.
    """

    key: str
    n: int
    hits: int
    hit_rate: float | None
    mean_excess: float | None
    median_excess: float | None


@dataclass(frozen=True)
class HorizonSummary:
    """All observations of one horizon: counts of SCORED / PENDING / EXCLUDED and the buckets
    of the SCORED ones.

    ``by_score`` is keyed by the aligned mean rounded half away from zero ("-2" .. "2";
    key "0" collects every aligned mean with ``|mean| < 0.5``, so it can have hits; only the
    ``mean == 0`` rows in it have ``hit`` None); ``by_verdict`` has only
    confirm / neutral / contradict keys. ``ic`` is the Spearman rank correlation of aligned
    mean and aligned excess (None for fewer than 3 observations or a constant side);
    ``no_call_mean_abs_excess`` is the mean ``|aligned_excess|`` of the ``mean == 0`` ones.
    """

    horizon: int
    scored: int
    pending: int
    excluded: int
    overall: Bucket
    by_verdict: Mapping[str, Bucket]
    by_score: Mapping[str, Bucket]
    by_sector: Mapping[str, Bucket]
    executed: Bucket
    ic: float | None
    no_call_mean_abs_excess: float | None


class VerifyOutcome(StrEnum):
    """How one verification ended: GAPS when a name had no price; FAILED is the partial report
    of a run that raised."""

    CLEAN = "clean"
    GAPS = "gaps"
    FAILED = "failed"


@dataclass(frozen=True)
class VerificationReport:
    """Audit record of one verification; fields not reached are empty.

    ``start`` / ``end`` are the window's bounds: None only on a run that failed before load
    resolved them (``start`` is also None for a completed run with no research and no
    ``--start``). ``started_at`` is tz-aware. ``failed_stage`` and ``error``
    (``"<Type>: <message>"``) are set only for FAILED. ``observations`` holds every
    (research date, name, horizon) in research-date, shortlist, horizon order; exclusions
    are read from each observation's ``status`` and ``reason``. ``summaries`` has one entry
    per horizon of ``HORIZONS`` once the run got that far.
    """

    start: date | None
    end: date | None
    started_at: datetime
    outcome: VerifyOutcome
    failed_stage: str | None
    error: str | None
    observations: tuple[Observation, ...]
    summaries: tuple[HorizonSummary, ...]
