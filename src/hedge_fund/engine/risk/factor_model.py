"""Factor risk model: market beta plus sector dummies, daily cross-sectional OLS, F and D."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from hedge_fund.domain.risk import RiskBreakdown
from hedge_fund.engine.portfolio.risk_model import (
    DEFAULT_RISK,
    TRADING_DAYS,
    RiskParams,
    simple_returns,
    validate_risk_params,
)

MARKET = "market"


@dataclass(frozen=True, eq=False)
class Regression:
    """Usable days only: factor returns (days x factors) and residuals (days x names).

    A residual is NaN on a usable day where its name had no return.
    """

    factor_returns: pd.DataFrame
    residuals: pd.DataFrame


@dataclass(frozen=True, eq=False)
class FactorModel:
    """Exposures (estimation names x factors), annualised F, and D for assessable names.

    ``exposures`` columns are ``market`` then the sorted sectors; ``factor_cov`` is labelled
    by factor on both axes. ``specific_var`` only holds names with at least ``min_obs``
    residuals: exactly the names a book can be assessed on.
    """

    exposures: pd.DataFrame
    factor_cov: pd.DataFrame
    specific_var: dict[str, float]
    usable_days: int


def betas(returns: pd.DataFrame, spy: pd.Series) -> dict[str, float]:
    """OLS slope (ddof=0) of each column of ``returns`` on ``spy``, pairwise per name.

    Only rows where both the name and ``spy`` have a return are used, and ``spy`` is
    demeaned over those rows. Raises ValueError when ``spy`` has zero variance on a name's
    rows (no shared row counts as zero variance).
    """
    out: dict[str, float] = {}
    for ticker in returns.columns:
        rows = returns[ticker].notna() & spy.notna()
        slope = _slope(
            returns.loc[rows, ticker].to_numpy(dtype=float), spy[rows].to_numpy(dtype=float)
        )
        if slope is None:
            raise ValueError(f"benchmark returns have zero variance on {ticker!r}'s rows")
        out[str(ticker)] = slope
    return out


def _slope(x: NDArray[np.float64], s: NDArray[np.float64]) -> float | None:
    if len(s) == 0:
        return None
    s = s - s.mean()
    var = float(s @ s) / len(s)
    if not var > 0:
        return None
    return float((x - x.mean()) @ s) / len(s) / var


def exposures(betas: Mapping[str, float], sectors: Mapping[str, str]) -> pd.DataFrame:
    """``X``: rows are ``betas``' tickers (sorted); columns ``market`` then sector dummies.

    The ``market`` column holds the beta; there is one 0/1 column per sector present among
    the tickers, sorted. ``sectors`` must cover every ticker; extra keys are ignored.
    """
    tickers = sorted(betas)
    names = sorted({sectors[t] for t in tickers})
    rows = [[betas[t], *(1.0 if sectors[t] == s else 0.0 for s in names)] for t in tickers]
    return pd.DataFrame(rows, index=tickers, columns=[MARKET, *names], dtype=float)


def estimation_universe(
    returns: pd.DataFrame, sectors: Mapping[str, str], *, min_obs: int
) -> list[str]:
    """Sorted tickers with >= ``min_obs`` returns whose sector keeps >= 2 such names."""
    counts = returns.notna().sum()
    kept = [str(t) for t in returns.columns if counts[t] >= min_obs]
    size = Counter(sectors[t] for t in kept)
    return sorted(t for t in kept if size[sectors[t]] >= 2)


def regress(returns: pd.DataFrame, exposures: pd.DataFrame, *, min_obs: int) -> Regression:
    """Daily cross-sectional OLS ``r_t = X_t f_t + u_t`` over the names with a return.

    ``exposures`` comes from ``exposures()``; ``returns`` must hold its tickers as columns.
    A day is usable only when every sector has >= 2 names with a return and ``rank(X_t)``
    equals the factor count; other days are skipped whole. Raises ValueError with fewer
    than ``min_obs`` usable days.
    """
    names = list(exposures.index)
    x = exposures.to_numpy(dtype=float)
    days: list[object] = []
    factor_rows: list[NDArray[np.float64]] = []
    residual_rows: list[NDArray[np.float64]] = []
    for day, row in zip(returns.index, returns[names].to_numpy(dtype=float), strict=True):
        present = ~np.isnan(row)
        x_t = x[present]
        if not _usable(x_t):
            continue
        f_t = np.linalg.lstsq(x_t, row[present], rcond=None)[0]
        u = np.full(len(names), np.nan)
        u[present] = row[present] - x_t @ f_t
        days.append(day)
        factor_rows.append(f_t)
        residual_rows.append(u)
    if len(days) < min_obs:
        raise ValueError(f"{len(days)} usable regression days, needs min_obs={min_obs}")
    index = pd.Index(days)
    return Regression(
        factor_returns=pd.DataFrame(factor_rows, index=index, columns=exposures.columns),
        residuals=pd.DataFrame(residual_rows, index=index, columns=names),
    )


def _usable(x_t: NDArray[np.float64]) -> bool:
    """Every sector column has >= 2 names and the day's exposures have full column rank."""
    k = int(x_t.shape[1])
    if len(x_t) < k or bool(np.any(x_t[:, 1:].sum(axis=0) < 2)):
        return False
    return int(np.linalg.matrix_rank(x_t)) == k


def covariances(regression: Regression, *, min_obs: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """``F = 252 cov(f)`` over usable days and ``D_i = 252 var(u_i)``, both ddof=0.

    A name with fewer than ``min_obs`` residuals gets no ``D_i`` (it is unassessable).
    """
    f = regression.factor_returns.to_numpy(dtype=float)
    centred = f - f.mean(axis=0)
    factors = list(regression.factor_returns.columns)
    factor_cov = pd.DataFrame(
        TRADING_DAYS * (centred.T @ centred) / len(f), index=factors, columns=factors
    )
    residuals = regression.residuals
    counts = residuals.notna().sum()
    variances = residuals.var(ddof=0)
    specific = {
        str(t): TRADING_DAYS * float(variances[t])
        for t in residuals.columns
        if counts[t] >= min_obs
    }
    return factor_cov, specific


def fit_factor_model(
    closes: pd.DataFrame,
    benchmark: pd.Series,
    sectors: Mapping[str, str],
    *,
    params: RiskParams = DEFAULT_RISK,
) -> FactorModel:
    """Fit the model on every name in ``closes`` over the last ``lookback`` returns.

    Returns follow ``simple_returns``; ``benchmark`` is reindexed to ``closes``' dates first
    (its other dates are ignored). Raises ValueError on invalid params, zero benchmark
    variance on a name's rows, or fewer than ``min_obs`` usable regression days.
    """
    validate_risk_params(params)
    returns = simple_returns(closes).iloc[-params.lookback :]
    spy = simple_returns(benchmark.reindex(closes.index)).iloc[-params.lookback :]
    names = estimation_universe(returns, sectors, min_obs=params.min_obs)
    x = exposures(betas(returns[names], spy), {t: sectors[t] for t in names})
    regression = regress(returns, x, min_obs=params.min_obs)
    factor_cov, specific = covariances(regression, min_obs=params.min_obs)
    return FactorModel(
        exposures=x,
        factor_cov=factor_cov,
        specific_var=specific,
        usable_days=len(regression.factor_returns),
    )


def empty_breakdown(unassessed: tuple[str, ...] = ()) -> RiskBreakdown:
    """All-zero breakdown for a book with no assessable name."""
    return RiskBreakdown(
        total_vol=0.0,
        factor_vol=0.0,
        specific_vol=0.0,
        factor_share=0.0,
        by_factor={},
        by_name={},
        unassessed=unassessed,
    )


def decompose(model: FactorModel, weights: Mapping[str, float]) -> RiskBreakdown:
    """Euler decomposition of the book's risk over its assessable names.

    With ``x = X'w``: ``factor_var = x'Fx``, ``specific_var = sum w_i^2 D_i``,
    ``by_factor[k] = x_k (Fx)_k`` and ``by_name[i] = w_i (Sigma w)_i / total`` with
    ``Sigma = X F X' + D``. A ticker without ``specific_var`` is unassessed and left out.
    Zero total variance gives zero shares.
    """
    unassessed = tuple(sorted(t for t in weights if t not in model.specific_var))
    tickers = sorted(t for t in weights if t in model.specific_var)
    if not tickers:
        return empty_breakdown(unassessed)
    w = np.array([weights[t] for t in tickers], dtype=float)
    x_names = model.exposures.loc[tickers].to_numpy(dtype=float)
    d = np.array([model.specific_var[t] for t in tickers], dtype=float)
    x = x_names.T @ w
    fx = model.factor_cov.to_numpy(dtype=float) @ x
    raw_factor_var = float(x @ fx)
    specific_var = float((w * w) @ d)
    scale = max(abs(raw_factor_var), specific_var, 1e-300)
    if raw_factor_var < -1e-12 * scale or raw_factor_var + specific_var < -1e-12 * scale:
        raise ValueError(
            f"negative factor/total variance (factor {raw_factor_var:.3e}, "
            f"specific {specific_var:.3e}): the factor covariance is not positive semidefinite"
        )
    factor_var = max(raw_factor_var, 0.0)
    total = factor_var + specific_var
    shares = w * (x_names @ fx + d * w) / total if total > 0 else np.zeros_like(w)
    return RiskBreakdown(
        total_vol=math.sqrt(total),
        factor_vol=math.sqrt(factor_var),
        specific_vol=math.sqrt(specific_var),
        factor_share=min(max(factor_var / total, 0.0), 1.0) if total > 0 else 0.0,
        by_factor=dict(zip(map(str, model.exposures.columns), (x * fx).tolist(), strict=True)),
        by_name=dict(zip(tickers, shares.tolist(), strict=True)),
        unassessed=unassessed,
    )
