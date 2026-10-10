"""default_provider wires universe sectors over a caching layer over the vendor."""

from datetime import date
from pathlib import Path

import pandas as pd

from hedge_fund.adapters.data.factory import default_provider
from hedge_fund.universe import load_universe
from tests.fakes import FakeProvider, assert_closes_equal

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
