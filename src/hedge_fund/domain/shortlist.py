"""Candidate shortlist handed from the factor engine to the research layer."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from enum import StrEnum

TRIMMED_CAP = "cap"


class Side(StrEnum):
    """Which book a candidate is researched for."""

    LONG = "long"
    SHORT = "short"


@dataclass(frozen=True)
class Candidate:
    """One shortlisted name; ``depth`` is its 1-based rank within (sector, side)."""

    ticker: str
    sector: str
    side: Side
    depth: int
    composite: float


@dataclass(frozen=True)
class Shortlist:
    """Candidates in selection order, plus qualified names cut by the cap (ticker -> reason)."""

    as_of: date
    candidates: tuple[Candidate, ...]
    trimmed: Mapping[str, str]
