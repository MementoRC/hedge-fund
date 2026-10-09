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


FACTORS: tuple[FactorSpec, ...] = (
    FactorSpec("momentum_12_1", momentum_12_1, sign=1, min_history=TRADING_DAYS + 1),
    FactorSpec("low_vol", low_vol, sign=1, min_history=TRADING_DAYS + 1),
)
