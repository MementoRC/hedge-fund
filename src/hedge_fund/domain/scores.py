"""Factor scoring results."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class FactorScore:
    """One ticker's factor values. ``raw`` and ``z`` are keyed by factor name.

    ``piotroski`` and ``altman_z`` are set when quality gates ran, else None.
    """

    ticker: str
    sector: str
    raw: Mapping[str, float]
    z: Mapping[str, float]
    composite: float
    fallback_universe_z: bool
    piotroski: int | None = None
    altman_z: float | None = None


@dataclass(frozen=True)
class ScoreResult:
    """Scores sorted by composite descending (ties by ticker), plus excluded tickers."""

    as_of: date
    scores: tuple[FactorScore, ...]
    excluded: Mapping[str, str]


@dataclass(frozen=True)
class FactorCrowding:
    """Excess mean residual correlation of one factor's long and short legs over the baseline.

    An excess is None when the leg has fewer than ``min_leg`` names or is not computable;
    ``crowded`` is True when any non-None excess reaches the threshold.
    """

    factor: str
    long_excess: float | None
    short_excess: float | None
    long_n: int
    short_n: int
    crowded: bool


@dataclass(frozen=True)
class CrowdingReport:
    """Per-factor crowding in scoring order; ``baseline_corr`` is the universe mean pair corr."""

    as_of: date
    baseline_corr: float | None
    factors: tuple[FactorCrowding, ...]
