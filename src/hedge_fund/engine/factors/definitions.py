"""Factor definitions: pure functions from a price window to a raw value per ticker."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS = 252
SKIP_DAYS = 21


@dataclass(frozen=True)
class FactorSpec:
    """A named factor. ``sign`` is +1 if higher is better, -1 if lower is better."""

    name: str
    fn: Callable[[pd.DataFrame], pd.Series]
    sign: int
    min_history: int


def momentum_12_1(window: pd.DataFrame) -> pd.Series:
    """Return from t-252 to t-21, skipping the most recent month."""
    return window.iloc[-1 - SKIP_DAYS] / window.iloc[-1 - TRADING_DAYS] - 1.0


def low_vol(window: pd.DataFrame) -> pd.Series:
    """Negated annualized volatility of daily log returns (higher is calmer)."""
    log_returns = np.log(window).diff().iloc[1:]
    return -log_returns.std(ddof=0) * np.sqrt(TRADING_DAYS)


def reversal_1m(window: pd.DataFrame) -> pd.Series:
    """Unsigned return over the last month (t-21 to t)."""
    return window.iloc[-1] / window.iloc[-1 - SKIP_DAYS] - 1.0


def high_52w_proximity(window: pd.DataFrame) -> pd.Series:
    """Close at t over the highest daily close of the last 252 rows (closes only, not intraday)."""
    return window.iloc[-1] / window.iloc[-TRADING_DAYS:].max()


FACTORS: tuple[FactorSpec, ...] = (
    FactorSpec("momentum_12_1", momentum_12_1, sign=1, min_history=TRADING_DAYS + 1),
    FactorSpec("low_vol", low_vol, sign=1, min_history=TRADING_DAYS + 1),
    FactorSpec("reversal_1m", reversal_1m, sign=-1, min_history=SKIP_DAYS + 1),
    FactorSpec("high_52w_proximity", high_52w_proximity, sign=1, min_history=TRADING_DAYS),
)
