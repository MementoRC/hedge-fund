"""Risk model: daily returns window, Ledoit-Wolf covariance, beta and vol versus a benchmark."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

TRADING_DAYS = 252


@dataclass(frozen=True)
class RiskParams:
    """Return rows in the window, and the fewest non-NaN returns a series needs."""

    lookback: int = 252
    min_obs: int = 200


DEFAULT_RISK = RiskParams()


@dataclass(frozen=True, eq=False)
class RiskModel:
    """Estimates for the names that kept enough history, as ticker-labelled frames and dicts.

    ``sigma`` is the annualized Ledoit-Wolf covariance, ``corr`` the unshrunk sample
    correlation (NaN for a zero-variance series), ``betas`` the unshrunk OLS slopes on the
    benchmark and ``vols`` the square roots of ``sigma``'s diagonal. ``excluded`` maps
    short-history tickers to ``"history"`` (sorted); ``delta`` is the shrinkage intensity and
    ``observations`` the aligned row count.
    """

    sigma: pd.DataFrame
    corr: pd.DataFrame
    betas: dict[str, float]
    vols: dict[str, float]
    excluded: dict[str, str]
    delta: float
    observations: int


def risk_model(
    closes: pd.DataFrame,
    benchmark: pd.Series,
    *,
    params: RiskParams = DEFAULT_RISK,
) -> RiskModel:
    """Validate ``params``; the estimation arrives in the next task."""
    _validate_params(params)
    raise NotImplementedError


def ledoit_wolf(returns: NDArray[np.float64]) -> tuple[NDArray[np.float64], float]:
    """Ledoit-Wolf (2004) shrinkage of the sample covariance toward a scaled identity.

    ``returns`` is T rows x N names; columns are demeaned here. With ``S = X'X / T``,
    ``m = tr(S) / N``, ``d2 = ||S - mI||_F^2 / N`` and
    ``b2_bar = sum_t ||x_t x_t' - S||_F^2 / (N T^2)``, the intensity is
    ``delta = min(b2_bar, d2) / d2`` (0 when ``d2 == 0``). Returns the daily (not annualized)
    ``delta m I + (1 - delta) S`` and ``delta``.
    """
    rows, n = returns.shape
    x = returns - returns.mean(axis=0)
    sample = x.T @ x / rows
    target = (float(np.trace(sample)) / n) * np.eye(n)
    d2 = float(np.sum((sample - target) ** 2)) / n
    b2_sum = float(np.sum(np.sum(x**2, axis=1) ** 2) - rows * np.sum(sample**2))
    b2_bar = max(b2_sum, 0.0) / (n * rows**2)
    delta = min(b2_bar, d2) / d2 if d2 > 0 else 0.0
    return delta * target + (1 - delta) * sample, delta


def _validate_params(params: RiskParams) -> None:
    """Type check first: bool, float and NaN would slip past plain range comparisons."""
    for name in ("lookback", "min_obs"):
        value = getattr(params, name)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must be an int, got {value!r}")
    if not 2 <= params.min_obs <= params.lookback:
        raise ValueError(
            "min_obs must satisfy 2 <= min_obs <= lookback, "
            f"got min_obs={params.min_obs!r}, lookback={params.lookback!r}"
        )
