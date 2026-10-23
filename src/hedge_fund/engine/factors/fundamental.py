"""Fundamental factor definitions: pure functions from fundamentals and close to a raw value."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True, eq=False)
class FundamentalInputs:
    """Current and prior metrics plus the close at as_of, all indexed by the same tickers."""

    current: pd.DataFrame
    prior: pd.DataFrame
    close: pd.Series


@dataclass(frozen=True)
class FundamentalFactorSpec:
    """A named fundamental factor. ``sign`` is +1 if higher is better, -1 if lower is better."""

    name: str
    fn: Callable[[FundamentalInputs], pd.Series]
    sign: int


def _positive(values: pd.Series) -> pd.Series:
    """``values`` with entries <= 0 (and NaN) as NaN, so they are safe as a denominator."""
    return values.where(values > 0)


def market_cap(inputs: FundamentalInputs) -> pd.Series:
    """Shares outstanding times close; NaN when the product is <= 0 or missing."""
    return _positive(inputs.current["shares_outstanding"] * inputs.close)


def earnings_yield(inputs: FundamentalInputs) -> pd.Series:
    """TTM net income over market cap; negative earnings stay negative."""
    return inputs.current["net_income"] / market_cap(inputs)


def gross_profitability(inputs: FundamentalInputs) -> pd.Series:
    """TTM gross profit over total assets; NaN when total assets <= 0."""
    return inputs.current["gross_profit"] / _positive(inputs.current["total_assets"])


def revenue_growth(inputs: FundamentalInputs) -> pd.Series:
    """TTM revenue over year-ago TTM revenue, minus one; NaN when prior revenue <= 0 or NaN."""
    return inputs.current["revenue"] / _positive(inputs.prior["revenue"]) - 1.0


def size(inputs: FundamentalInputs) -> pd.Series:
    """Natural log of market cap; NaN market cap stays NaN."""
    return np.log(market_cap(inputs))


FUNDAMENTAL_FACTORS: tuple[FundamentalFactorSpec, ...] = (
    FundamentalFactorSpec("earnings_yield", earnings_yield, sign=1),
    FundamentalFactorSpec("gross_profitability", gross_profitability, sign=1),
    FundamentalFactorSpec("revenue_growth", revenue_growth, sign=1),
    FundamentalFactorSpec("size", size, sign=-1),
)
