"""Deterministic price and fundamentals fixtures for engine tests."""

from collections.abc import Callable, Mapping, Sequence
from datetime import date

import numpy as np
import pandas as pd

from hedge_fund.domain.fundamentals import METRIC_NAMES, PERIOD_COLUMNS, FundamentalsSnapshot
from hedge_fund.domain.snapshot import PriceSnapshot
from hedge_fund.ports.fundamentals import FACT_COLUMNS, coerce_facts

END_DATE = "2026-09-30"
FUNDAMENTALS_AS_OF = date(2026, 9, 30)


def make_closes(tickers: Sequence[str], n_days: int = 300, seed: int = 0) -> pd.DataFrame:
    """Geometric random-walk closes on business days ending END_DATE, one column per ticker.

    Each ticker gets its own drift and volatility, so ties are practically impossible.
    """
    rng = np.random.default_rng(seed)
    index = pd.bdate_range(end=END_DATE, periods=n_days)
    data = {}
    for ticker in tickers:
        drift = rng.uniform(-0.001, 0.001)
        vol = rng.uniform(0.005, 0.03)
        data[ticker] = 100.0 * np.exp(np.cumsum(rng.normal(drift, vol, n_days)))
    return pd.DataFrame(data, index=index)


def snapshot_from(closes: pd.DataFrame, sectors: Mapping[str, str]) -> PriceSnapshot:
    """Build a snapshot whose as_of is the date of the last row."""
    return PriceSnapshot(as_of=closes.index[-1].date(), closes=closes, sectors=dict(sectors))


def make_fundamentals(
    tickers: Sequence[str] = ("AAA", "BBB", "CCC"), seed: int = 0
) -> FundamentalsSnapshot:
    """Valid snapshot: positive random metrics, E 2026-06-30, E′ 2025-06-30, filed 2026-08-01."""
    rng = np.random.default_rng(seed)
    index = sorted(tickers)
    names = list(METRIC_NAMES)
    shape = (len(index), len(names))
    current = pd.DataFrame(rng.uniform(1.0, 1000.0, shape), index=index, columns=names)
    prior = pd.DataFrame(rng.uniform(1.0, 1000.0, shape), index=index, columns=names)
    dates = dict(zip(PERIOD_COLUMNS, ["2026-06-30", "2025-06-30", "2026-08-01"], strict=True))
    periods = pd.DataFrame(
        {column: pd.Timestamp(value) for column, value in dates.items()}, index=index
    ).astype("datetime64[ns]")
    return FundamentalsSnapshot(
        as_of=FUNDAMENTALS_AS_OF, current=current, prior=prior, periods=periods
    )


_PASSING_CURRENT: dict[str, Callable[[float], float]] = {
    "net_income": lambda k: 100.0 * k * k,
    "operating_cash_flow": lambda k: 150.0 * k * k,
    "total_assets": lambda k: 1000.0,
    "long_term_debt": lambda k: 100.0,
    "current_assets": lambda k: 500.0,
    "current_liabilities": lambda k: 200.0,
    "shares_outstanding": lambda k: 100.0 * k,
    "gross_profit": lambda k: 400.0 * k,
    "revenue": lambda k: 800.0 * k,
    "retained_earnings": lambda k: 300.0,
    "operating_income": lambda k: 150.0 * k,
    "total_liabilities": lambda k: 500.0,
}
_PASSING_PRIOR: dict[str, Callable[[float], float]] = {
    "net_income": lambda k: 80.0 * k * k,
    "total_assets": lambda k: 1000.0,
    "long_term_debt": lambda k: 200.0,
    "current_assets": lambda k: 400.0,
    "current_liabilities": lambda k: 250.0,
    "shares_outstanding": lambda k: 100.0 * k,
    "gross_profit": lambda k: 300.0,
    "revenue": lambda k: 700.0,
}


def make_passing_fundamentals(
    tickers: Sequence[str] = ("AAA", "BBB", "CCC"),
) -> FundamentalsSnapshot:
    """``make_fundamentals`` with gate inputs set so every ticker has F = 9 and Z > 1.81.

    Values scale with k = 1 + 0.1 * i (i = sorted ticker position) so fundamental factor
    raw values still differ across tickers.
    """
    base = make_fundamentals(tickers)
    current, prior = base.current.copy(), base.prior.copy()
    for i, ticker in enumerate(current.index):
        k = 1.0 + 0.1 * i
        current.loc[ticker, list(_PASSING_CURRENT)] = [f(k) for f in _PASSING_CURRENT.values()]
        prior.loc[ticker, list(_PASSING_PRIOR)] = [f(k) for f in _PASSING_PRIOR.values()]
    return FundamentalsSnapshot(base.as_of, current, prior, base.periods)


def make_facts(tickers: Sequence[str]) -> pd.DataFrame:
    """EDGAR-like facts per ticker; at FUNDAMENTALS_AS_OF, E = 2026-06-30 and E′ = 2025-06-30.

    Assets: 950 (2025-06-30), 1000 (2025-12-31), 1100 (2026-06-30); dei shares 50 filed with
    the 2026-06-30 10-Q (2026-08-01); FY2025 revenue 400 in the 10-K filed 2026-02-15.
    """
    shares = "dei:EntityCommonStockSharesOutstanding"
    rows = []
    for t in tickers:
        rows += [
            (t, "us-gaap:Assets", None, "2025-06-30", 950.0, "2025-08-01", "10-Q", f"{t}-q225"),
            (
                t,
                "us-gaap:Revenues",
                "2025-01-01",
                "2025-12-31",
                400.0,
                "2026-02-15",
                "10-K",
                f"{t}-k25",
            ),
            (t, "us-gaap:Assets", None, "2025-12-31", 1000.0, "2026-02-15", "10-K", f"{t}-k25"),
            (t, "us-gaap:Assets", None, "2026-06-30", 1100.0, "2026-08-01", "10-Q", f"{t}-q226"),
            (t, shares, None, "2026-07-25", 50.0, "2026-08-01", "10-Q", f"{t}-q226"),
        ]
    return coerce_facts(pd.DataFrame(rows, columns=list(FACT_COLUMNS)))
