"""Target portfolio types: signed weights per name, plus every name left out and why."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date

from hedge_fund.domain.shortlist import Side

EXCLUDED_HISTORY = "history"
EXCLUDED_THIN_BOOK = "thin-book"
EXCLUDED_OPTIMIZER_ZERO = "optimizer:zero"
EXCLUDED_INFEASIBLE = "infeasible"


@dataclass(frozen=True)
class Position:
    """One sized name; ``weight`` is a signed fraction of capital (+ long, - short)."""

    ticker: str
    sector: str
    side: Side
    weight: float
    alpha: float
    beta: float


@dataclass(frozen=True)
class TargetPortfolio:
    """Longs then shorts (each by descending abs(weight), then ticker) plus exclusions.

    ``excluded`` maps ticker -> reason, sorted by ticker. An empty portfolio has
    ``positions=()`` and ``gross = net = beta = expected_vol = 0.0``.
    """

    as_of: date
    positions: tuple[Position, ...]
    excluded: Mapping[str, str]
    gross: float
    net: float
    beta: float
    expected_vol: float
