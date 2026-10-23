"""Composition roots for the default DataProvider and FundamentalsProvider stacks."""

from __future__ import annotations

from pathlib import Path

from hedge_fund.adapters.data.caching import CachingProvider
from hedge_fund.adapters.data.edgar_provider import EdgarProvider
from hedge_fund.adapters.data.fundamentals_caching import CachingFundamentalsProvider
from hedge_fund.adapters.data.static_sectors import StaticSectorProvider
from hedge_fund.adapters.data.yfinance_provider import YFinanceProvider
from hedge_fund.ports.data import DataProvider
from hedge_fund.ports.fundamentals import FundamentalsProvider
from hedge_fund.universe import load_universe


def default_provider(
    cache_dir: Path,
    universe: str = "sp500",
    *,
    vendor: DataProvider | None = None,
) -> DataProvider:
    """Universe sectors over a parquet cache over `vendor` (Yahoo Finance by default)."""
    cached = CachingProvider(vendor if vendor is not None else YFinanceProvider(), cache_dir)
    return StaticSectorProvider(cached, load_universe(universe))


def default_fundamentals_provider(
    cache_dir: Path,
    universe: str = "sp500",
    *,
    vendor: FundamentalsProvider | None = None,
) -> FundamentalsProvider:
    """Parquet cache over `vendor` (SEC EDGAR with the universe's CIKs by default).

    The default vendor needs SEC_USER_AGENT; see EdgarProvider.
    """
    ciks = load_universe(universe).ciks
    inner: FundamentalsProvider = vendor if vendor is not None else EdgarProvider(ciks)
    return CachingFundamentalsProvider(inner, cache_dir)
