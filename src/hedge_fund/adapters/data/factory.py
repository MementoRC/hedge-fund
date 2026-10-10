"""Composition root for the default DataProvider stack."""

from __future__ import annotations

from pathlib import Path

from hedge_fund.adapters.data.caching import CachingProvider
from hedge_fund.adapters.data.static_sectors import StaticSectorProvider
from hedge_fund.adapters.data.yfinance_provider import YFinanceProvider
from hedge_fund.ports.data import DataProvider
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
