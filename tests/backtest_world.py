"""A small frozen store for the golden backtest: 18 members in 3 sectors, SPY, 13 months of prices.

Every fact is filed before the first rebalance (2024-02-01), save POISON_TICKER's FY2023 10-K
filed 2024-02-20 (after it, before the second); prices are a seeded random walk. The book is
this wide because the default risk limits (name_risk 0.15) veto any book of fewer than about
a dozen names. Events:
- SPLIT_TICKER splits 2:1 ex 2024-02-14 (stored prices are post-split throughout);
- DIVIDEND_TICKER pays DIVIDEND ex 2024-02-21;
- DELIST_TICKER's prices stop after 2024-02-27 (its membership ends 2024-02-28);
- REMOVED leaves the index and ADDED joins it on the rebalance day 2024-03-01;
- POISON_TICKER's FY2023 10-K (the late one) reports a deep accumulated deficit, so the
  Altman gate drops it from every cycle that can see that filing, and from no earlier one.
"""

from __future__ import annotations

import random
from datetime import date
from fractions import Fraction

import numpy as np
import pandas as pd

from hedge_fund.domain.backtest import Dividend, FrozenData, MemberSpan
from hedge_fund.domain.ledger import Split
from tests.factories import make_passing_facts

SEED = 1  # picked so both rebalances trade and every event below hits a held name
SECTORS = {
    f"{prefix}{number}": sector
    for prefix, sector in (("AE", "Energy"), ("BT", "Tech"), ("CH", "Health"))
    for number in range(1, 7)
}
REMOVED = "AE6"
ADDED = "BT6"
SPLIT_TICKER = "AE1"
DIVIDEND_TICKER = "AE2"
DELIST_TICKER = "BT1"
POISON_TICKER = "BT2"
DIVIDEND = 0.4
FIRST_DAY = date(2023, 2, 1)
LAST_DAY = date(2024, 3, 8)
SPLIT_EX = date(2024, 2, 14)
DIVIDEND_EX = date(2024, 2, 21)
LAST_TRADED = date(2024, 2, 27)
DELIST_ON = date(2024, 2, 28)
MEMBERSHIP_CHANGE = date(2024, 3, 1)
LATE_FILED = pd.Timestamp("2024-02-20")
SHIFT_TO_2023 = pd.Timedelta(days=-1004)  # E 2026-06-30 -> 2023-09-30, filed 2023-11-01
POISONED_METRIC_TAG = "us-gaap:RetainedEarningsAccumulatedDeficit"


def _prices(days: pd.DatetimeIndex) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Seeded random walks: a market factor plus per-name drift and noise; opens near closes."""
    rng = random.Random(SEED)
    market = np.array([rng.gauss(0.0003, 0.008) for _ in days])
    closes = {"SPY": 400.0 * np.exp(np.cumsum(market))}
    for ticker in SECTORS:
        drift, vol = rng.uniform(-0.0015, 0.0015), rng.uniform(0.008, 0.016)
        noise = np.array([rng.gauss(drift, vol) for _ in days])
        closes[ticker] = 100.0 * np.exp(np.cumsum(market + noise))
    frame = pd.DataFrame(closes, index=days, dtype="float64")
    gaps = pd.DataFrame(
        {t: [rng.uniform(-0.004, 0.004) for _ in days] for t in frame.columns}, index=days
    )
    opens = frame * (1.0 + gaps)
    stopped = days > pd.Timestamp(LAST_TRADED)
    frame.loc[stopped, DELIST_TICKER] = np.nan
    opens.loc[stopped, DELIST_TICKER] = np.nan
    return opens, frame


def _facts() -> pd.DataFrame:
    """Every name's passing filings, all filed 2023-11-01, plus POISON_TICKER's late 10-K."""
    tickers = sorted(SECTORS)
    early = make_passing_facts(tickers)
    for column in ("start", "end", "filed"):
        early[column] = early[column] + SHIFT_TO_2023
    late = early[early["ticker"] == POISON_TICKER].copy()
    for column in ("start", "end"):
        late[column] = late[column] + pd.Timedelta(days=92)
    late["filed"] = LATE_FILED
    late["accn"] = late["accn"] + "-fy23"
    deficit = (late["ticker"] == POISON_TICKER) & (late["tag"] == POISONED_METRIC_TAG)
    late.loc[deficit & (late["end"] == late["end"].max()), "value"] = -1_000_000.0
    return pd.concat([early, late], ignore_index=True)


def make_golden_data() -> FrozenData:
    """The golden store (see the module docstring)."""
    days = pd.bdate_range(FIRST_DAY, LAST_DAY)
    opens, closes = _prices(days)
    start = date(2020, 1, 2)
    membership = tuple(
        MemberSpan(
            ticker,
            MEMBERSHIP_CHANGE if ticker == ADDED else start,
            {REMOVED: MEMBERSHIP_CHANGE, DELIST_TICKER: DELIST_ON}.get(ticker),
        )
        for ticker in SECTORS
    )
    return FrozenData(
        start=FIRST_DAY,
        end=LAST_DAY,
        opens=opens,
        closes=closes,
        splits=(Split(SPLIT_TICKER, SPLIT_EX, Fraction(2)),),
        dividends=(Dividend(DIVIDEND_TICKER, DIVIDEND_EX, DIVIDEND),),
        facts=_facts(),
        sectors=dict(SECTORS),
        ciks={ticker: f"{i + 1:010d}" for i, ticker in enumerate(SECTORS)},
        membership=membership,
        unpriced=(),
        no_sector=(),
        no_cik=(),
        sic_sectors=(),
    )
