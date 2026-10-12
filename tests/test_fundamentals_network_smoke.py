"""Live SEC EDGAR smoke test. Excluded by default; run with: pytest -m network.

Needs SEC_USER_AGENT ("Name email@example.com"); skipped when it is unset.
"""

import os
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from hedge_fund.adapters.data.edgar_provider import EdgarProvider
from hedge_fund.engine.fundamentals.derive import MAX_STALENESS_DAYS
from hedge_fund.ports.fundamentals import FACT_COLUMNS
from hedge_fund.services.fundamentals import build_fundamentals_snapshot

pytestmark = pytest.mark.network

CIKS = {"AAPL": "0000320193"}


def test_aapl_flows_from_edgar_to_a_fundamentals_snapshot() -> None:
    if not os.environ.get("SEC_USER_AGENT"):
        pytest.skip("SEC_USER_AGENT is not set")
    provider = EdgarProvider(CIKS)
    facts = provider.facts(["AAPL"])
    assert not facts.empty
    assert list(facts.columns) == list(FACT_COLUMNS)

    as_of = date.today()
    build = build_fundamentals_snapshot(provider, ["AAPL"], as_of, ciks=CIKS)
    assert build.unresolved == {}
    current = build.snapshot.current.loc["AAPL"]
    for name in ("revenue", "net_income", "total_assets", "shares_outstanding"):
        assert np.isfinite(current[name])
    current_end = build.snapshot.periods.loc["AAPL", "current_end"]
    assert current_end >= pd.Timestamp(as_of - timedelta(days=MAX_STALENESS_DAYS))
