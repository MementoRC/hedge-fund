"""Live yfinance smoke test. Excluded by default; run with: pytest -m network."""

from collections.abc import Sequence
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

from hedge_fund.adapters.data.factory import default_provider
from hedge_fund.adapters.data.yfinance_provider import YFinanceProvider
from hedge_fund.engine.factors.score import score
from hedge_fund.services.snapshots import build_snapshot
from hedge_fund.universe import load_universe
from tests.fakes import assert_closes_equal

pytestmark = pytest.mark.network

TICKERS = ["AAPL", "MSFT", "NVDA"]


class _CountingVendor:
    """YFinanceProvider that counts closes calls."""

    def __init__(self) -> None:
        self._inner = YFinanceProvider()
        self.closes_calls = 0

    def closes(self, tickers: Sequence[str], start: date, end: date) -> pd.DataFrame:
        self.closes_calls += 1
        return self._inner.closes(tickers, start, end)

    def sectors(self, tickers: Sequence[str]) -> dict[str, str | None]:
        return self._inner.sectors(tickers)


def _past_weekday(days_ago: int = 3) -> date:
    """Most recent weekday at least `days_ago` days back, so the cache fully covers it."""
    day = date.today() - timedelta(days=days_ago)
    while day.weekday() >= 5:
        day -= timedelta(days=1)
    return day


def test_real_tickers_flow_from_yfinance_to_scores(tmp_path: Path) -> None:
    vendor = _CountingVendor()
    provider = default_provider(tmp_path, vendor=vendor)
    as_of = _past_weekday()
    build = build_snapshot(provider, TICKERS, as_of)
    assert build.unresolved == {}
    assert len(build.snapshot.closes) >= 253
    universe_sectors = load_universe("sp500").sectors
    assert dict(build.snapshot.sectors) == {t: universe_sectors[t] for t in TICKERS}
    result = score(build.snapshot)
    assert {s.ticker for s in result.scores} | set(result.excluded) == set(TICKERS)
    assert (tmp_path / "prices" / "AAPL.parquet").is_file()
    calls_after_first = vendor.closes_calls
    again = build_snapshot(provider, TICKERS, as_of)
    assert vendor.closes_calls == calls_after_first
    assert_closes_equal(again.snapshot.closes, build.snapshot.closes)
