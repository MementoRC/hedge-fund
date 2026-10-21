"""Factor risk model: market beta plus sector dummies, daily cross-sectional OLS, F and D."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
from numpy.typing import NDArray

MARKET = "market"


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
