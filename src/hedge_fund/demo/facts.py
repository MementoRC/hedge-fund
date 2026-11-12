"""EDGAR-like facts that pass every quality gate, for the demo world and for tests. Pure.

``make_fundamentals`` draws its random metrics from ``np.random.default_rng(seed)``; for
metrics outside ``PASSING_PRIOR`` those draws reach the prior-period facts rows, so the stream
is part of the pinned golden store. The demo world passes ``seed - 1`` (seed 1 -> stream 0,
today's bytes); the default ``seed=0`` is what every other test keeps using.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import date

import numpy as np
import pandas as pd

from hedge_fund.domain.fundamentals import METRIC_NAMES, PERIOD_COLUMNS, FundamentalsSnapshot
from hedge_fund.engine.fundamentals.metrics import METRICS, SHARES_DEI_TAG
from hedge_fund.ports.fundamentals import FACT_COLUMNS, coerce_facts

FUNDAMENTALS_AS_OF = date(2026, 9, 30)


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


PASSING_CURRENT: dict[str, Callable[[float], float]] = {
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
PASSING_PRIOR: dict[str, Callable[[float], float]] = {
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
    tickers: Sequence[str] = ("AAA", "BBB", "CCC"), seed: int = 0
) -> FundamentalsSnapshot:
    """``make_fundamentals`` with gate inputs set so every ticker has F = 9 and Z > 1.81.

    Values scale with k = 1 + 0.1 * i (i = sorted ticker position) so fundamental factor
    raw values still differ across tickers.
    """
    base = make_fundamentals(tickers, seed)
    current, prior = base.current.copy(), base.prior.copy()
    for i, ticker in enumerate(current.index):
        k = 1.0 + 0.1 * i
        current.loc[ticker, list(PASSING_CURRENT)] = [f(k) for f in PASSING_CURRENT.values()]
        prior.loc[ticker, list(PASSING_PRIOR)] = [f(k) for f in PASSING_PRIOR.values()]
    return FundamentalsSnapshot(base.as_of, current, prior, base.periods)


def make_passing_facts(tickers: Sequence[str], seed: int = 0) -> pd.DataFrame:
    """EDGAR-like facts deriving, as of FUNDAMENTALS_AS_OF, to ``make_passing_fundamentals``.

    Only the metrics the fundamental factors and quality gates read (the ``PASSING_CURRENT``
    keys) are reported, which keeps derivation fast; the others derive to NaN. Each is
    reported under its first tag (shares on the dei cover tag) by two 10-K filings:
    E = 2026-06-30 filed 2026-08-01 and E′ = 2025-06-30 filed 2025-08-01. Flows are annual
    facts ending at E / E′, stocks are instants at E / E′.
    """
    target = make_passing_fundamentals(tickers, seed)
    rows = []
    for ticker in target.current.index:
        filings = (
            (target.current, "2026-06-30", "2026-08-01", f"{ticker}-e"),
            (target.prior, "2025-06-30", "2025-08-01", f"{ticker}-p"),
        )
        for values, end, filed, accn in filings:
            for spec in METRICS:
                if spec.name not in PASSING_CURRENT:
                    continue
                tag = SHARES_DEI_TAG if spec.name == "shares_outstanding" else spec.tags[0]
                start = None
                if spec.kind == "flow":
                    start = pd.Timestamp(end) - pd.DateOffset(years=1) + pd.Timedelta(days=1)
                value = float(values.at[ticker, spec.name])
                rows.append((ticker, tag, start, end, value, filed, "10-K", accn))
    return coerce_facts(pd.DataFrame(rows, columns=list(FACT_COLUMNS)))
