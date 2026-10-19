"""Risk model: daily returns window, Ledoit-Wolf covariance, beta and vol versus a benchmark."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from hedge_fund.domain.portfolio import EXCLUDED_HISTORY

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
    """Estimate covariance, correlation, beta and vol from the last ``lookback`` returns.

    Returns are ``close_t / close_{t-1} - 1`` on ``closes``' date index with no forward
    fill; ``benchmark`` is reindexed to that index first (its other dates are ignored). A
    name with fewer than ``min_obs`` returns in the window is excluded as ``"history"``. The
    rest are aligned with the benchmark on rows where all have returns; while that leaves
    fewer than ``min_obs`` rows, the name with the fewest returns (ties: alphabetically
    last) is dropped as ``"history"``. Raises ValueError on invalid params, on a benchmark
    with fewer than ``min_obs`` returns, or on zero benchmark variance in the aligned rows.
    """
    _validate_params(params)
    returns = _returns(closes).iloc[-params.lookback :]
    spy = _returns(benchmark.reindex(closes.index)).iloc[-params.lookback :]
    available = int(spy.notna().sum())
    if available < params.min_obs:
        raise ValueError(
            f"benchmark has {available} returns in the window, needs min_obs={params.min_obs}"
        )
    counts = {str(t): int(n) for t, n in returns.notna().sum().items()}
    kept = [t for t, n in counts.items() if n >= params.min_obs]
    excluded = {t: EXCLUDED_HISTORY for t, n in counts.items() if n < params.min_obs}
    rows = _shared_rows(returns, spy, kept)
    while kept and rows.sum() < params.min_obs:
        worst = max(kept, key=lambda t: (-counts[t], t))
        kept.remove(worst)
        excluded[worst] = EXCLUDED_HISTORY
        rows = _shared_rows(returns, spy, kept)
    excluded = dict(sorted(excluded.items()))
    if not kept:
        empty = pd.DataFrame()
        return RiskModel(empty, empty, {}, {}, excluded, delta=0.0, observations=0)
    return _estimate(returns.loc[rows, kept], spy[rows], excluded)


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


def _returns(prices: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Simple returns; a NaN close makes the returns on both adjacent rows NaN.

    Non-positive closes are treated as missing, which leaves NaN returns on both adjacent rows.
    """
    prices = prices.where(prices > 0)
    return (prices / prices.shift(1) - 1).replace([np.inf, -np.inf], np.nan)


def _shared_rows(returns: pd.DataFrame, spy: pd.Series, kept: list[str]) -> pd.Series:
    """Boolean row mask: the benchmark and every kept name have a return."""
    return returns[kept].notna().all(axis=1) & spy.notna()


def _estimate(window: pd.DataFrame, spy: pd.Series, excluded: dict[str, str]) -> RiskModel:
    x = window.to_numpy(dtype=float)
    s = spy.to_numpy(dtype=float)
    s = s - s.mean()
    spy_var = float(s @ s) / len(s)
    if not spy_var > 0:
        raise ValueError("benchmark returns have zero variance in the aligned window")
    shrunk, delta = ledoit_wolf(x)
    sigma = TRADING_DAYS * shrunk
    xc = x - x.mean(axis=0)
    sample = xc.T @ xc / len(xc)
    std = np.sqrt(np.diag(sample))
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.clip(sample / np.outer(std, std), -1.0, 1.0)
    betas = xc.T @ s / len(s) / spy_var
    tickers = [str(t) for t in window.columns]
    return RiskModel(
        sigma=pd.DataFrame(sigma, index=tickers, columns=tickers),
        corr=pd.DataFrame(corr, index=tickers, columns=tickers),
        betas=dict(zip(tickers, betas.tolist(), strict=True)),
        vols=dict(zip(tickers, np.sqrt(np.diag(sigma)).tolist(), strict=True)),
        excluded=excluded,
        delta=delta,
        observations=len(x),
    )


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
