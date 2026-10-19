"""Market-neutral mean-variance optimizer with beta, sector, gross and per-name limits."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd

from hedge_fund.domain.portfolio import EXCLUDED_THIN_BOOK
from hedge_fund.domain.shortlist import Side


class PortfolioInfeasibleError(ValueError):
    """The QP did not converge or its solution breaks a limit; no book is returned."""


@dataclass(frozen=True)
class OptimizeParams:
    """Risk aversion, book gross, per-name cap, minimum names per side and hard limits."""

    risk_aversion: float = 1.0
    gross: float = 2.0
    max_weight: float = 0.10
    min_names_per_side: int = 3
    beta_band: float = 0.10
    sector_net: float = 0.10
    sector_gross: float = 0.50


DEFAULT_OPTIMIZE = OptimizeParams()


@dataclass(frozen=True)
class Asset:
    """One optimizer input; ``side`` fixes the sign of its weight."""

    ticker: str
    sector: str
    side: Side
    alpha: float
    beta: float


@dataclass(frozen=True)
class Allocation:
    """Non-zero signed weights by ticker, plus every other input name with its reason."""

    weights: dict[str, float]
    excluded: dict[str, str]


def optimize_weights(
    assets: Sequence[Asset],
    sigma: pd.DataFrame,
    *,
    params: OptimizeParams = DEFAULT_OPTIMIZE,
) -> Allocation:
    """Validate ``params`` and apply the thin-book rule; sizing arrives in the next tasks."""
    _validate_params(params)
    n_long = sum(1 for asset in assets if asset.side is Side.LONG)
    if min(n_long, len(assets) - n_long) < params.min_names_per_side:
        return _excluded_all(assets, EXCLUDED_THIN_BOOK)
    raise NotImplementedError


def _excluded_all(assets: Sequence[Asset], reason: str) -> Allocation:
    return Allocation(weights={}, excluded=dict.fromkeys(sorted(a.ticker for a in assets), reason))


_FLOAT_FIELDS = ("risk_aversion", "gross", "max_weight", "beta_band", "sector_net", "sector_gross")


def _validate_params(params: OptimizeParams) -> None:
    """Type check first: bool, float and NaN would slip past plain range comparisons."""
    for name in _FLOAT_FIELDS:
        value = getattr(params, name)
        if (
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(value)
        ):
            raise ValueError(f"{name} must be a finite number, got {value!r}")
    count = params.min_names_per_side
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError(f"min_names_per_side must be an int >= 1, got {count!r}")
    _validate_ranges(params)


def _validate_ranges(params: OptimizeParams) -> None:
    half = params.gross / 2
    if not params.risk_aversion > 0:
        raise ValueError(f"risk_aversion must be > 0, got {params.risk_aversion!r}")
    if not 0 < params.max_weight <= half:
        raise ValueError(
            "max_weight must satisfy 0 < max_weight <= gross / 2, "
            f"got max_weight={params.max_weight!r}, gross={params.gross!r}"
        )
    if params.max_weight * params.min_names_per_side > half:
        raise ValueError(
            "max_weight * min_names_per_side must be <= gross / 2, got "
            f"{params.max_weight!r} * {params.min_names_per_side!r} > {half!r}"
        )
    if params.beta_band < 0:
        raise ValueError(f"beta_band must be >= 0, got {params.beta_band!r}")
    if not 0 < params.sector_net <= params.sector_gross:
        raise ValueError(
            "sector limits must satisfy 0 < sector_net <= sector_gross, "
            f"got sector_net={params.sector_net!r}, sector_gross={params.sector_gross!r}"
        )
