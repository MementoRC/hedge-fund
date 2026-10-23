"""Tests for the universe-first sector decorator."""

from datetime import date
from types import MappingProxyType

import pandas as pd

from hedge_fund.adapters.data.static_sectors import StaticSectorProvider
from hedge_fund.universe import Universe
from tests.factories import make_closes
from tests.fakes import FakeProvider, assert_closes_equal

UNIVERSE = Universe(
    name="demo",
    list_date=date(2026, 10, 1),
    tickers=("AAA", "BBB"),
    sectors=MappingProxyType({"AAA": "Energy", "BBB": "Financials"}),
)


def test_universe_sectors_win_and_misses_are_delegated() -> None:
    fake = FakeProvider(make_closes(["AAA"]), {"AAA": "Utilities", "CCC": "Materials"})
    result = StaticSectorProvider(fake, UNIVERSE).sectors(["CCC", "AAA", "ZZZ"])
    assert list(result.items()) == [("CCC", "Materials"), ("AAA", "Energy"), ("ZZZ", None)]
    assert fake.sectors_calls == [("CCC", "ZZZ")]


def test_all_universe_hits_skip_inner() -> None:
    fake = FakeProvider(make_closes(["AAA"]), {})
    result = StaticSectorProvider(fake, UNIVERSE).sectors(["BBB", "AAA"])
    assert result == {"BBB": "Financials", "AAA": "Energy"}
    assert fake.sectors_calls == []


def test_closes_pass_through() -> None:
    fake = FakeProvider(make_closes(["AAA", "BBB"]), {})
    start, end = date(2026, 6, 1), date(2026, 9, 30)
    result = StaticSectorProvider(fake, UNIVERSE).closes(["AAA"], start, end)
    assert fake.closes_calls == [(("AAA",), start, end)]
    assert_closes_equal(result, fake.frame.loc[pd.Timestamp(start) : pd.Timestamp(end), ["AAA"]])
