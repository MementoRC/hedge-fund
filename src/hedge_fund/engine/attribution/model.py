"""The risk model one trading day at a time, ex-post and point-in-time. Pure.

Betas and exposures come from the ``lookback`` returns ending the day before; the day's
own row is then regressed alone, giving its market and sector factor returns and each
regressed name's residual. Only universe names enter; held names never move the betas.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date

import pandas as pd

from hedge_fund.engine.portfolio.risk_model import RiskParams
from hedge_fund.engine.risk.factor_model import (
    MARKET,
    betas,
    estimation_universe,
    exposures,
    regress,
)


@dataclass(frozen=True)
class DayModel:
    """One day's regression: ``market`` and ``sector_returns`` (sector -> f) are the day's
    factor returns; ``betas``, ``sectors`` and ``residuals`` cover the names regressed that
    day (a name in the exposures without a return that day has no residual and is not
    regressed). For each regressed name ``r = beta * market + sector_returns[sector] + u``.
    """

    market: float
    sector_returns: dict[str, float]
    betas: dict[str, float]
    sectors: dict[str, str]
    residuals: dict[str, float]


def day_model(
    returns: pd.DataFrame,
    benchmark: pd.Series,
    sectors: Mapping[str, str | None],
    universe: Iterable[str],
    params: RiskParams,
    day: date,
) -> DayModel | None:
    """The model for ``day`` (a row of ``returns``), or None when the day is unregressed.

    ``returns`` and ``benchmark`` are simple returns on the trading-day index. The window is
    the last ``params.lookback`` rows before ``day``. A universe name enters only with a
    sector, at least ``params.min_obs`` returns in the window and non-zero benchmark
    variance on its rows (``betas`` would raise); ``estimation_universe`` then keeps sectors
    with >= 2 names. One bad name never unregresses the day. None when no name survives, or
    ``regress`` (``min_obs=1``) refuses or drops the day.
    """
    at = pd.Timestamp(day)
    window = returns.loc[returns.index < at].tail(params.lookback)
    found = _candidates(window, benchmark.reindex(window.index), sectors, universe, params)
    sector_of = {ticker: sector for ticker, (_, sector) in found.items()}
    kept = estimation_universe(window[list(found)], sector_of, min_obs=params.min_obs)
    if not kept:
        return None
    x = exposures({ticker: found[ticker][0] for ticker in kept}, sector_of)
    try:
        regression = regress(returns.loc[[at], kept], x, min_obs=1)
    except ValueError:  # the day is not usable: a sector short of names, or rank-deficient
        return None
    factors = regression.factor_returns.iloc[0]
    residuals = regression.residuals.iloc[0]
    return DayModel(
        market=float(factors[MARKET]),
        sector_returns={str(k): float(v) for k, v in factors.items() if k != MARKET},
        betas={ticker: found[ticker][0] for ticker in kept},
        sectors={ticker: sector_of[ticker] for ticker in kept},
        residuals={str(k): float(v) for k, v in residuals.items() if not math.isnan(v)},
    )


def _candidates(
    window: pd.DataFrame,
    spy: pd.Series,
    sectors: Mapping[str, str | None],
    universe: Iterable[str],
    params: RiskParams,
) -> dict[str, tuple[float, str]]:
    """Universe ticker -> (beta, sector) for the names fit to enter the estimation set."""
    found: dict[str, tuple[float, str]] = {}
    for ticker in sorted(set(universe)):
        sector = sectors.get(ticker)
        if sector is None or ticker not in window.columns:
            continue
        if int(window[ticker].notna().sum()) < params.min_obs:
            continue
        try:
            beta = betas(window[[ticker]], spy)[ticker]
        except ValueError:  # zero benchmark variance on the name's rows
            continue
        found[ticker] = (beta, sector)
    return found
