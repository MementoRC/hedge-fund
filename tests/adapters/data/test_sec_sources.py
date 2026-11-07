"""SEC lookups for the backtest snapshot (offline: get, clock and sleep injected)."""

from __future__ import annotations

import pytest

from hedge_fund.adapters.data.edgar_provider import EdgarProvider
from hedge_fund.adapters.data.sec_sources import COMPANY_TICKERS_URL, SUBMISSIONS_URL, SecLookups
from hedge_fund.ports.fundamentals import FundamentalsProviderError
from tests.adapters.data.test_edgar_provider import AGENT, FakeGet, FakeResponse, FakeTime

TICKERS = {
    "0": {"cik_str": 320193, "ticker": "aapl", "title": "Apple Inc."},
    "1": {"cik_str": 999, "ticker": "AAPL", "title": "Later duplicate"},
    "2": {"cik_str": 1067983, "ticker": "BRK-B", "title": "Berkshire Hathaway"},
}


def _lookups(*outcomes: FakeResponse) -> tuple[SecLookups, FakeGet]:
    get = FakeGet(*outcomes)
    fake_time = FakeTime()
    client = EdgarProvider(
        {}, get=get, user_agent=AGENT, clock=fake_time.clock, sleep=fake_time.sleep
    )
    return SecLookups(client), get


def test_company_ciks_upper_case_zero_pad_and_keep_the_first_entry() -> None:
    lookups, get = _lookups(FakeResponse(payload=TICKERS))
    assert lookups.company_ciks() == {"AAPL": "0000320193", "BRK-B": "0001067983"}
    assert get.calls[0][0] == COMPANY_TICKERS_URL


def test_a_missing_company_tickers_file_fails_closed() -> None:
    lookups, _ = _lookups(FakeResponse(404))
    with pytest.raises(FundamentalsProviderError, match="HTTP 404"):
        lookups.company_ciks()


def test_a_malformed_company_tickers_file_fails_closed() -> None:
    lookups, _ = _lookups(FakeResponse(payload={"0": {"ticker": "AAPL"}}))
    with pytest.raises(FundamentalsProviderError, match="malformed company_tickers"):
        lookups.company_ciks()


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        (FakeResponse(payload={"sic": "3571"}), "3571"),
        (FakeResponse(payload={"sic": ""}), None),
        (FakeResponse(payload={"sic": 3571}), None),
        (FakeResponse(payload={}), None),
        (FakeResponse(404), None),
    ],
)
def test_sic_reads_the_submissions_file(response: FakeResponse, expected: str | None) -> None:
    lookups, get = _lookups(response)
    assert lookups.sic("0000320193") == expected
    assert get.calls[0][0] == SUBMISSIONS_URL.format(cik="0000320193")
