"""Side-aligned conviction verdict from K research ratings of one candidate."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from hedge_fund.domain.research import NameResearch, RunRecord
from hedge_fund.domain.shortlist import Candidate


@dataclass(frozen=True)
class ConvictionParams:
    """Minimum valid runs, mean-score thresholds and the dispersion ceiling."""

    min_valid: int = 2
    confirm: float = 0.5
    contradict: float = -0.5
    max_std: float = 1.0


DEFAULT_CONVICTION = ConvictionParams()


def conviction(
    candidate: Candidate,
    runs: Sequence[RunRecord],
    *,
    params: ConvictionParams = DEFAULT_CONVICTION,
) -> NameResearch:
    """Validate ``params``; the verdict maths arrives in the next task."""
    _validate_params(params)
    raise NotImplementedError


def _validate_params(params: ConvictionParams) -> None:
    """Type check first: bool, float and NaN would slip past plain range comparisons."""
    min_valid = params.min_valid
    if isinstance(min_valid, bool) or not isinstance(min_valid, int) or min_valid < 1:
        raise ValueError(f"min_valid must be an int >= 1, got {min_valid!r}")
    for name in ("confirm", "contradict", "max_std"):
        value = getattr(params, name)
        if (
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(value)
        ):
            raise ValueError(f"{name} must be a finite number, got {value!r}")
    if not params.contradict < 0 < params.confirm:
        raise ValueError(
            "thresholds must satisfy contradict < 0 < confirm, "
            f"got contradict={params.contradict!r}, confirm={params.confirm!r}"
        )
    if not params.max_std > 0:
        raise ValueError(f"max_std must be > 0, got {params.max_std!r}")
