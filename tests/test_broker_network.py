"""Live Alpaca paper smoke test, read-only. Excluded by default; run with: pytest -m network.

Needs APCA_API_KEY_ID and APCA_API_SECRET_KEY for a paper account; skipped when either is
unset. Never submits, cancels or modifies orders.
"""

import os

import pytest

from hedge_fund.adapters.broker import AlpacaPaperBroker
from hedge_fund.domain.orders import Holding

pytestmark = [
    pytest.mark.network,
    pytest.mark.skipif(
        not (os.environ.get("APCA_API_KEY_ID") and os.environ.get("APCA_API_SECRET_KEY")),
        reason="APCA_API_KEY_ID / APCA_API_SECRET_KEY unset",
    ),
]


def test_account_equity_is_positive() -> None:
    assert AlpacaPaperBroker().account().equity > 0


def test_positions_parse() -> None:
    holdings = AlpacaPaperBroker().positions()
    assert all(isinstance(holding, Holding) for holding in holdings)
    assert all(isinstance(holding.shares, int) and holding.shares != 0 for holding in holdings)
    tickers = [holding.ticker for holding in holdings]
    assert tickers == sorted(set(tickers))
