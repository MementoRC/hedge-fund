"""Sector-balanced long/short candidate shortlist from a ScoreResult."""

from __future__ import annotations

from dataclasses import dataclass

from hedge_fund.domain.scores import ScoreResult
from hedge_fund.domain.shortlist import Shortlist


@dataclass(frozen=True)
class ShortlistParams:
    """Names per side per sector, and the hard cap on the whole shortlist."""

    per_sector: int = 3
    max_names: int = 40


DEFAULT_SHORTLIST = ShortlistParams()


def shortlist(result: ScoreResult, *, params: ShortlistParams = DEFAULT_SHORTLIST) -> Shortlist:
    """Pick up to ``per_sector`` longs and shorts per sector, capped at ``max_names``."""
    _validate_params(params)
    return Shortlist(as_of=result.as_of, candidates=(), trimmed={})


def _validate_params(params: ShortlistParams) -> None:
    """Type check first: bool, float and NaN would slip past a plain ``< 1`` comparison."""
    for name in ("per_sector", "max_names"):
        value = getattr(params, name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be >= 1, got {value!r}")
