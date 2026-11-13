"""Deterministic price and fundamentals fixtures for engine tests."""

import dataclasses
from collections.abc import Mapping, Sequence
from datetime import date
from fractions import Fraction
from typing import Any

import numpy as np
import pandas as pd

from hedge_fund.demo.facts import FUNDAMENTALS_AS_OF as FUNDAMENTALS_AS_OF
from hedge_fund.demo.facts import make_fundamentals as make_fundamentals
from hedge_fund.demo.facts import make_passing_facts as make_passing_facts
from hedge_fund.demo.facts import make_passing_fundamentals as make_passing_fundamentals
from hedge_fund.domain.backtest import Dividend, FrozenData, MemberSpan
from hedge_fund.domain.ledger import Split
from hedge_fund.domain.research import Verdict
from hedge_fund.domain.shortlist import Side
from hedge_fund.domain.snapshot import PriceSnapshot
from hedge_fund.domain.verification import Observation, ObservationStatus
from hedge_fund.ports.fundamentals import FACT_COLUMNS, coerce_facts

END_DATE = "2026-09-30"


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


def make_crowded_closes(
    tickers: Sequence[str], crowded: Sequence[str], strength: float = 1.0, seed: int = 0
) -> pd.DataFrame:
    """``make_closes`` paths with one shared N(0, 0.02) log-return shock added to ``crowded``.

    The shock comes from a separate RNG seeded ``seed + 1``; prices are rebuilt from the
    first close, so non-crowded columns are unchanged (up to float round-off).
    """
    closes = make_closes(tickers, seed=seed)
    log_returns = np.log(closes).diff().iloc[1:]
    latent = np.random.default_rng(seed + 1).normal(0.0, 0.02, len(log_returns))
    for ticker in crowded:
        log_returns[ticker] = log_returns[ticker] + strength * latent
    rebuilt = np.exp(log_returns.cumsum()) * closes.iloc[0]
    return pd.concat([closes.iloc[:1], rebuilt]).set_axis(closes.index)


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


def make_frozen_data(**changes: Any) -> FrozenData:
    """A small frozen store: AAA, BBB and SPY on the 30 business days to 2024-02-09.

    AAA splits 2:1 ex 2024-01-22 (the stored prices are post-split throughout) and pays 0.5
    ex 2024-02-01. BBB leaves the index effective 2024-01-29, when CCC (unpriced, no
    sector, no CIK) joins. ``changes`` replace fields.
    """
    days = pd.bdate_range("2024-01-01", periods=30)
    closes = make_closes(["AAA", "BBB", "SPY"], n_days=30).set_axis(days, axis=0)
    base = FrozenData(
        start=date(2024, 1, 1),
        end=date(2024, 2, 9),
        opens=closes * 0.99,
        closes=closes,
        splits=(Split("AAA", date(2024, 1, 22), Fraction(2)),),
        dividends=(Dividend("AAA", date(2024, 2, 1), 0.5),),
        facts=make_facts(["AAA", "BBB"]),
        sectors={"AAA": "Energy", "BBB": "Financials"},
        ciks={"AAA": "0000000001", "BBB": "0000000002"},
        membership=(
            MemberSpan("AAA", date(2020, 1, 2), None),
            MemberSpan("BBB", date(2020, 1, 2), date(2024, 1, 29)),
            MemberSpan("CCC", date(2024, 1, 29), None),
        ),
        unpriced=("CCC",),
        no_sector=("CCC",),
        no_cik=("CCC",),
        sic_sectors=(),
    )
    return dataclasses.replace(base, **changes)


def make_observation(**changes: Any) -> Observation:
    """A SCORED long CONFIRM observation at the 5-day horizon; ``changes`` replace fields."""
    base = Observation(
        as_of=date(2026, 9, 1),
        ticker="AAA",
        sector="Tech",
        side=Side.LONG,
        verdict=Verdict.CONFIRM,
        mean=1.0,
        std=0.0,
        agreement=1.0,
        valid=3,
        horizon=5,
        base_date=date(2026, 9, 1),
        end_date=date(2026, 9, 8),
        name_return=0.06,
        spy_return=0.01,
        aligned_excess=0.05,
        hit=True,
        executed=False,
        status=ObservationStatus.SCORED,
        reason=None,
    )
    return dataclasses.replace(base, **changes)
