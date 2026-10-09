"""Live yfinance smoke test. Excluded by default; run with: pytest -m network."""

from datetime import date
from pathlib import Path

import pytest

from hedge_fund.adapters.data.caching import CachingProvider
from hedge_fund.adapters.data.yfinance_provider import YFinanceProvider
from hedge_fund.engine.factors.score import score
from hedge_fund.services.snapshots import build_snapshot

pytestmark = pytest.mark.network

TICKERS = ["AAPL", "MSFT", "NVDA"]


def test_real_tickers_flow_from_yfinance_to_scores(tmp_path: Path) -> None:
    provider = CachingProvider(YFinanceProvider(), tmp_path)
    build = build_snapshot(provider, TICKERS, date.today())
    assert build.unresolved == {}
    assert len(build.snapshot.closes) >= 253
    result = score(build.snapshot)
    assert {s.ticker for s in result.scores} | set(result.excluded) == set(TICKERS)
    assert (tmp_path / "prices" / "AAPL.parquet").is_file()
