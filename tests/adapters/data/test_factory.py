"""Composition roots: default price and fundamentals provider stacks."""

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from hedge_fund.adapters.data.edgar_provider import EdgarProvider
from hedge_fund.adapters.data.factory import default_fundamentals_provider, default_provider
from hedge_fund.adapters.data.fundamentals_caching import CachingFundamentalsProvider
from hedge_fund.ports.fundamentals import FundamentalsProviderError
from hedge_fund.universe import load_universe
from tests.factories import make_facts
from tests.fakes import FakeFundamentalsProvider, FakeProvider, assert_closes_equal

START = date(2026, 1, 2)
END = date(2026, 3, 31)


def _fake() -> FakeProvider:
    index = pd.bdate_range(START, END)
    frame = pd.DataFrame({"AAPL": 100.0, "ZZZZ": 50.0}, index=index)
    return FakeProvider(frame, {"ZZZZ": "Energy"})


def test_universe_sector_wins_and_vendor_is_asked_only_for_the_rest(tmp_path: Path) -> None:
    vendor = _fake()
    provider = default_provider(tmp_path, vendor=vendor)

    expected = load_universe("sp500").sectors["AAPL"]
    assert provider.sectors(["AAPL"]) == {"AAPL": expected}
    assert vendor.sectors_calls == []

    assert provider.sectors(["ZZZZ"]) == {"ZZZZ": "Energy"}
    assert vendor.sectors_calls == [("ZZZZ",)]


def test_closes_are_cached_between_calls(tmp_path: Path) -> None:
    vendor = _fake()
    provider = default_provider(tmp_path, vendor=vendor)

    first = provider.closes(["AAPL"], START, END)
    second = provider.closes(["AAPL"], START, END)

    assert len(vendor.closes_calls) == 1
    assert (tmp_path / "prices" / "AAPL.parquet").is_file()
    assert_closes_equal(second, first)


def test_fundamentals_are_cached_between_calls(tmp_path: Path) -> None:
    vendor = FakeFundamentalsProvider(make_facts(["AAPL"]))
    provider = default_fundamentals_provider(tmp_path, vendor=vendor)
    first = provider.facts(["AAPL"])
    second = provider.facts(["AAPL"])
    assert vendor.calls == [("AAPL",)]
    assert (tmp_path / "fundamentals" / "AAPL.parquet").is_file()
    pd.testing.assert_frame_equal(second, first)


def test_default_fundamentals_vendor_is_edgar_with_universe_ciks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SEC_USER_AGENT", "hedge-fund tests tests@example.com")
    provider = default_fundamentals_provider(tmp_path)
    assert isinstance(provider, CachingFundamentalsProvider)
    assert isinstance(provider._inner, EdgarProvider)
    assert provider._inner._ciks == dict(load_universe("sp500").ciks)


def test_default_fundamentals_provider_requires_sec_user_agent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("SEC_USER_AGENT", raising=False)
    with pytest.raises(FundamentalsProviderError, match="SEC_USER_AGENT"):
        default_fundamentals_provider(tmp_path)


def test_default_fundamentals_provider_rejects_unknown_universe(tmp_path: Path) -> None:
    vendor = FakeFundamentalsProvider(make_facts([]))
    with pytest.raises(ValueError, match="unknown universe"):
        default_fundamentals_provider(tmp_path, "nasdaq", vendor=vendor)
