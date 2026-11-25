"""Live backtest snapshot smoke test. Excluded by default; run with: pytest -m network.

Needs SEC_USER_AGENT ("Name email@example.com"); skipped when it is unset.
"""

import os
from datetime import date
from fractions import Fraction
from pathlib import Path

import pytest

from hedge_fund import cli
from hedge_fund.adapters.data.frozen_providers import (
    FrozenDataProvider,
    FrozenFundamentalsProvider,
)
from hedge_fund.adapters.data.frozen_store import load_store, write_store
from hedge_fund.domain.backtest import MemberSpan
from hedge_fund.domain.ledger import Split
from hedge_fund.services.backtest_snapshot import build_frozen_data
from hedge_fund.universe import load_universe

pytestmark = pytest.mark.network

SPANS = tuple(MemberSpan(ticker, date(2001, 11, 30), None) for ticker in ("AAPL", "MSFT", "NVDA"))


def test_a_one_year_snapshot_rebuilds_the_nvda_split_and_keeps_edgar_filings(
    tmp_path: Path,
) -> None:
    if not os.environ.get("SEC_USER_AGENT"):
        pytest.skip("SEC_USER_AGENT is not set")
    universe = load_universe("sp500")
    data = build_frozen_data(
        cli.build_snapshot_sources(),
        SPANS,
        sectors=universe.sectors,
        ciks=universe.ciks,
        start=date(2024, 1, 2),
        end=date(2024, 12, 31),
    )
    write_store(tmp_path / "store", data, created_at=cli.clock())
    loaded = load_store(tmp_path / "store")
    assert loaded.unpriced == ()
    assert "SPY" in loaded.closes.columns

    before = FrozenDataProvider(loaded, date(2024, 6, 7))
    june_7 = float(before.closes(["NVDA"], date(2024, 6, 3), date(2024, 6, 7))["NVDA"].iloc[-1])
    assert 1150.0 < june_7 < 1250.0  # the real pre-split close was about 1,208.88
    after = before.with_as_of(date(2024, 6, 10))
    seen = float(after.closes(["NVDA"], date(2024, 6, 7), date(2024, 6, 7))["NVDA"].iloc[0])
    assert seen == pytest.approx(june_7 / 10, rel=1e-9)
    assert after.splits(["NVDA"], date(2024, 6, 7), date(2024, 6, 10)) == (
        Split("NVDA", date(2024, 6, 10), Fraction(10)),
    )
    assert before.splits(["NVDA"], date(2024, 1, 1), date(2024, 6, 7)) == ()

    facts = FrozenFundamentalsProvider(loaded).facts(["NVDA"])
    assert facts["filed"].notna().all()
    assert facts["accn"].nunique() > 4
