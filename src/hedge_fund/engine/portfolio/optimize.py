"""Market-neutral mean-variance optimizer with beta, sector, gross and per-name limits."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.optimize import linprog

from hedge_fund.domain.portfolio import EXCLUDED_INFEASIBLE, EXCLUDED_THIN_BOOK
from hedge_fund.domain.shortlist import Side

_SLACK = 1e-6  # relative shrink of g* when a limit binds below the cap
_CAP_TOLERANCE = 1e-9
_MIN_SIDE_GROSS = 1e-6


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
    """Thin-book rule and the LP feasible side gross; the QP arrives in the next task."""
    _validate_params(params)
    n_long = sum(1 for asset in assets if asset.side is Side.LONG)
    if min(n_long, len(assets) - n_long) < params.min_names_per_side:
        return _excluded_all(assets, EXCLUDED_THIN_BOOK)
    problem = _problem(assets, sigma, params)
    start = _feasible_start(problem, params)
    if start is None:
        return _excluded_all(assets, EXCLUDED_INFEASIBLE)
    raise NotImplementedError


def _excluded_all(assets: Sequence[Asset], reason: str) -> Allocation:
    return Allocation(weights={}, excluded=dict.fromkeys(sorted(a.ticker for a in assets), reason))


@dataclass(frozen=True, eq=False)
class _Problem:
    """Arrays for one solve. Linear limits are rows of ``g @ w <= h``, named by ``labels``."""

    tickers: list[str]
    alpha: NDArray[np.float64]
    sigma: NDArray[np.float64]
    bounds: list[tuple[float, float]]
    long: NDArray[np.float64]
    short: NDArray[np.float64]
    g: NDArray[np.float64]
    h: NDArray[np.float64]
    labels: list[str]


def _problem(assets: Sequence[Asset], sigma: pd.DataFrame, params: OptimizeParams) -> _Problem:
    """Bounds, side indicators, and the beta and per-sector limits as linear rows."""
    tickers = [asset.ticker for asset in assets]
    sign = np.array([1.0 if asset.side is Side.LONG else -1.0 for asset in assets])
    beta = np.array([asset.beta for asset in assets])
    rows = [beta, -beta]
    limits = [params.beta_band, params.beta_band]
    labels = ["beta <= beta_band", "beta >= -beta_band"]
    for sector in sorted({asset.sector for asset in assets}):
        member = np.array([1.0 if asset.sector == sector else 0.0 for asset in assets])
        rows += [member, -member, sign * member]
        limits += [params.sector_net, params.sector_net, params.sector_gross]
        labels += [
            f"sector {sector} net <= sector_net",
            f"sector {sector} net >= -sector_net",
            f"sector {sector} gross <= sector_gross",
        ]
    cap = params.max_weight
    return _Problem(
        tickers=tickers,
        alpha=np.array([asset.alpha for asset in assets]),
        sigma=sigma.loc[tickers, tickers].to_numpy(dtype=float),
        bounds=[(0.0, cap) if s > 0 else (-cap, 0.0) for s in sign],
        long=(sign > 0).astype(float),
        short=(sign < 0).astype(float),
        g=np.array(rows),
        h=np.array(limits),
        labels=labels,
    )


def _feasible_start(
    problem: _Problem, params: OptimizeParams
) -> tuple[float, NDArray[np.float64]] | None:
    """Largest side gross the limits allow (LP), and the LP weights scaled to it.

    ``g*`` maximizes ``g`` subject to the bounds, ``0 <= g <= cap``, the side sums ``+g`` /
    ``-g`` and every linear limit. Below the cap a limit binds, so ``g*`` is shrunk by
    ``1e-6`` relative slack to keep SLSQP's active set non-singular. None when the LP solver
    fails or the side gross is below ``1e-6``.
    """
    n = len(problem.tickers)
    cap = min(
        params.gross / 2,
        problem.long.sum() * params.max_weight,
        problem.short.sum() * params.max_weight,
    )
    result = linprog(
        c=np.r_[np.zeros(n), -1.0],
        A_ub=np.c_[problem.g, np.zeros(len(problem.h))],
        b_ub=problem.h,
        A_eq=np.vstack([np.r_[problem.long, -1.0], np.r_[problem.short, 1.0]]),
        b_eq=np.zeros(2),
        bounds=[*problem.bounds, (0.0, cap)],
        method="highs",
    )
    if not result.success:
        return None
    best = float(result.x[-1])
    side_gross = best * (1 - _SLACK) if best < cap - _CAP_TOLERANCE else best
    if side_gross < _MIN_SIDE_GROSS:
        return None
    return side_gross, np.asarray(result.x[:n], dtype=float) * (side_gross / best)


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
