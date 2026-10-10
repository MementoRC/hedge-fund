"""Live yfinance smoke test. Excluded by default; run with: pytest -m network."""

from datetime import date
from pathlib import Path

import pytest

from hedge_fund.adapters.data.factory import default_provider
from hedge_fund.engine.factors.score import score
from hedge_fund.services.snapshots import build_snapshot
from tests.fakes import assert_closes_equal

pytestmark = pytest.mark.network

TICKERS = ["AAPL", "MSFT", "NVDA"]


def test_real_tickers_flow_from_yfinance_to_scores(tmp_path: Path) -> None:
    provider = default_provider(tmp_path)
    build = build_snapshot(provider, TICKERS, date.today())
    assert build.unresolved == {}
    assert len(build.snapshot.closes) >= 253
    result = score(build.snapshot)
    assert {s.ticker for s in result.scores} | set(result.excluded) == set(TICKERS)
    assert (tmp_path / "prices" / "AAPL.parquet").is_file()
    again = build_snapshot(provider, TICKERS, date.today())
    assert_closes_equal(again.snapshot.closes, build.snapshot.closes)
