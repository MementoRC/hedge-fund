"""Macro-sensitivity factors: per-ticker OLS betas of daily log returns on FRED series changes.

Point in time: a value dated d is known from d + 1, so every row dated on or after ``as_of``
is cut from both the closes and the series before they are aligned. This is a cut-off, not a
row shift: the regression stays contemporaneous, return(d) on the series change on d. Missing
values are dropped, never filled, so a gap removes its interval instead of adding a zero
return or a zero change.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from hedge_fund.domain.macro import RATE_SERIES, RISKOFF_SERIES
from hedge_fund.engine.factors.definitions import TRADING_DAYS

_FLAT_TOLERANCE = 1e-12  # relative to the series level: smaller change std is "does not move"
MIN_PAIRS = 200  # fewer paired returns give NaN, excluded as nonfinite_factor


@dataclass(frozen=True)
class MacroFactorSpec:
    """A named macro factor over (unfilled closes, raw macro frame, as_of).

    ``sign`` is +1 if higher is better, -1 if lower is better.
    """

    name: str
    fn: Callable[[pd.DataFrame, pd.DataFrame, date], pd.Series]
    sign: int
    min_history: int


def macro_beta(closes: pd.DataFrame, series: pd.Series, as_of: date) -> pd.Series:
    """OLS slope of each ticker's daily log returns on ``series``' daily changes before ``as_of``.

    Per ticker, its non-NaN closes are intersected with the non-NaN series dates, and returns
    and changes are taken over consecutive rows of that intersection. NaN when fewer than
    MIN_PAIRS returns pair up, or when the series does not move.
    """
    cut = pd.Timestamp(as_of)
    window = _as_ns(closes.loc[closes.index < cut])
    known = _as_ns(series.loc[series.index < cut].dropna())
    return pd.Series(
        {ticker: _slope(window[ticker].dropna(), known) for ticker in window.columns},
        index=window.columns,
        dtype="float64",
    )


def macro_series_problem(series: pd.Series, as_of: date) -> str | None:
    """Why ``series`` gives a NaN beta for every ticker, or None if it may not.

    Only rows dated strictly before ``as_of`` with a finite value (NaN and inf dropped) count.
    The checks are exact sufficient conditions for NaN on every ticker: fewer than
    MIN_PAIRS + 1 finite observations (no ticker can pair up enough returns) or all of them
    exactly equal (no change in any intersection). The near-flat tolerance of ``_slope`` is
    deliberately not applied: it runs on each ticker's own intersected dates, so a
    whole-series version would be neither necessary nor sufficient. A caller rejects only
    what fails for every ticker.
    """
    cut = pd.Timestamp(as_of)
    values = series.loc[series.index < cut].to_numpy(dtype="float64")
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return f"has no finite value before {as_of}"
    if finite.size < MIN_PAIRS + 1:
        return f"has {finite.size} finite observations before {as_of}, needs {MIN_PAIRS + 1}"
    if (finite == finite[0]).all():
        return f"is constant before {as_of}"
    return None


def _as_ns[T: (pd.Series, pd.DataFrame)](frame: T) -> T:
    """Normalise the datetime unit so closes and series on different units still align."""
    return frame.set_axis(pd.DatetimeIndex(frame.index).as_unit("ns"))


def _slope(prices: pd.Series, known: pd.Series) -> float:
    """Slope for one ticker. Non-positive closes are dropped like missing ones (log undefined)."""
    dates = prices.loc[prices > 0].index.intersection(known.index)
    if len(dates) < MIN_PAIRS + 1:
        return math.nan
    returns = np.diff(np.log(prices.loc[dates].to_numpy(dtype="float64")))
    levels = known.loc[dates].to_numpy(dtype="float64")
    changes = np.diff(levels)
    centred = changes - changes.mean()
    spread = float(centred @ centred)
    if spread <= len(changes) * (_FLAT_TOLERANCE * max(float(np.abs(levels).max()), 1.0)) ** 2:
        return math.nan
    return float(centred @ (returns - returns.mean()) / spread)


def rate_beta(closes: pd.DataFrame, macro: pd.DataFrame, as_of: date) -> pd.Series:
    """Minus the absolute beta on DGS10 changes: rate-insensitive names score higher.

    Positive (banks) and negative (utilities, REITs) rate sensitivity are penalised alike.
    """
    return -macro_beta(closes, macro[RATE_SERIES], as_of).abs()


def riskoff_beta(closes: pd.DataFrame, macro: pd.DataFrame, as_of: date) -> pd.Series:
    """The beta on VIXCLS changes: a less negative beta, a name that holds up when VIX jumps,
    scores higher."""
    return macro_beta(closes, macro[RISKOFF_SERIES], as_of)


MACRO_FACTORS: tuple[MacroFactorSpec, ...] = (
    MacroFactorSpec("rate_beta", rate_beta, sign=1, min_history=TRADING_DAYS + 1),
    MacroFactorSpec("riskoff_beta", riskoff_beta, sign=1, min_history=TRADING_DAYS + 1),
)
