"""Prices as a vendor showed them on a past date, rebuilt from freeze-date history (pure).

The frozen store keeps yfinance ``auto_adjust=False`` prices: adjusted for every split up
to the freeze date, for no dividend. As seen on ``as_of``:

- Splits. A split ex-dated after ``as_of`` had not happened, so every row up to ``as_of``
  is multiplied by ``split_factor(ticker, as_of, splits)``, the product of the ratios of
  the ticker's splits with ``ex_date > as_of``.
- Dividends. Yahoo's back-adjustment (the CRSP method behind its Adj Close), restricted to
  ex-dates on or before ``as_of``: a dividend ``D`` ex-dated ``e`` has the factor
  ``f_e = 1 - D / P``, ``P`` the last close before ``e``, and the adjusted close on day
  ``t`` is ``close_t * prod(f_e for t < e <= as_of)``. ``D`` and ``P`` share a split basis,
  so ``f_e`` does not depend on it.

The adjusted close on ``as_of`` itself is therefore its split-unwound raw close.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date

import pandas as pd

from hedge_fund.domain.backtest import Dividend
from hedge_fund.domain.ledger import Split
from hedge_fund.engine.ledger.splits import split_factor


def unwound(prices: pd.Series, ticker: str, splits: Sequence[Split], as_of: date) -> pd.Series:
    """``prices`` (an Open or Close column) up to ``as_of``, on ``as_of``'s split basis.

    Rows after ``as_of`` are dropped and splits ex-dated after it are undone.
    """
    visible = prices.loc[prices.index <= pd.Timestamp(as_of)]
    return visible * float(split_factor(ticker, as_of, splits))


def dividend_factors(
    closes: pd.Series, ticker: str, dividends: Sequence[Dividend], as_of: date
) -> pd.Series:
    """Back-adjustment factor per row of ``closes`` up to ``as_of`` (1.0 where none applies).

    Only ``ticker``'s dividends with ``ex_date <= as_of`` count; one with no non-NaN close
    before its ex-date is skipped, as there is nothing before it to adjust.

    Raises ValueError when a dividend is not below its prior close (``f_e <= 0``).
    """
    visible = closes.loc[closes.index <= pd.Timestamp(as_of)]
    factors = pd.Series(1.0, index=visible.index, dtype="float64")
    for dividend in dividends:
        if dividend.ticker != ticker or dividend.ex_date > as_of:
            continue
        ex = pd.Timestamp(dividend.ex_date)
        prior = visible.loc[visible.index < ex].dropna()
        if prior.empty:
            continue
        prior_close = float(prior.iloc[-1])
        factor = 1.0 - dividend.amount / prior_close
        if factor <= 0.0:
            raise ValueError(
                f"{ticker}: dividend {dividend.amount} ex {dividend.ex_date} is not below "
                f"the prior close {prior_close}"
            )
        factors.loc[factors.index < ex] *= factor
    return factors


def closes_as_of(
    closes: pd.Series,
    ticker: str,
    splits: Sequence[Split],
    dividends: Sequence[Dividend],
    as_of: date,
) -> pd.Series:
    """Dividend- and split-adjusted closes up to ``as_of``, as a vendor showed them on it."""
    factors = dividend_factors(closes, ticker, dividends, as_of)
    return unwound(closes, ticker, splits, as_of) * factors
