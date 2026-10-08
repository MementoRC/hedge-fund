"""Factor scoring results."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class FactorScore:
    """One ticker's factor values. ``raw`` and ``z`` are keyed by factor name."""

    ticker: str
    sector: str
    raw: Mapping[str, float]
    z: Mapping[str, float]
    composite: float
    fallback_universe_z: bool


@dataclass(frozen=True)
class ScoreResult:
    """Scores sorted by composite descending (ties by ticker), plus excluded tickers."""

    as_of: date
    scores: tuple[FactorScore, ...]
    excluded: Mapping[str, str]
