"""Tests for ticker universes: parsing and the shipped S&P 500 file."""

from datetime import date

import pytest

from hedge_fund.universe import GICS_SECTORS, load_universe, parse_universe

VALID = "# list_date: 2026-10-01\nticker,gics_sector\nAAA,Energy\nBRK-B,Financials\n"


def test_parse_universe_reads_list_date_tickers_and_sectors() -> None:
    universe = parse_universe("demo", VALID)
    assert universe.name == "demo"
    assert universe.list_date == date(2026, 10, 1)
    assert universe.tickers == ("AAA", "BRK-B")
    assert dict(universe.sectors) == {"AAA": "Energy", "BRK-B": "Financials"}


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("ticker,gics_sector\nAAA,Energy\n", "list_date"),
        ("# list_date: 2026-10-01\nsymbol,sector\nAAA,Energy\n", "header"),
        ("# list_date: 2026-10-01\nticker,gics_sector\nAAA,Energy\nAAA,Energy\n", "duplicate"),
        ("# list_date: 2026-10-01\nticker,gics_sector\nAAA,Tech\n", "unknown sector"),
    ],
)
def test_parse_universe_rejects_malformed_text(text: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        parse_universe("demo", text)


def test_gics_sectors_has_the_eleven_names() -> None:
    assert len(GICS_SECTORS) == 11
    assert "Information Technology" in GICS_SECTORS


def test_shipped_sp500_is_well_formed() -> None:
    universe = load_universe("sp500")
    assert universe.list_date <= date.today()
    assert len(universe.tickers) >= 490
    assert len(set(universe.tickers)) == len(universe.tickers)
    assert set(universe.tickers) == set(universe.sectors)
    assert set(universe.sectors.values()) <= GICS_SECTORS
    assert all("." not in ticker for ticker in universe.tickers)


@pytest.mark.parametrize("name", ["nasdaq", "../sp500", "SP500"])
def test_unknown_universe_raises(name: str) -> None:
    with pytest.raises(ValueError, match="unknown universe"):
        load_universe(name)
