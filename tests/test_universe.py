"""Tests for ticker universes: parsing and the shipped S&P 500 file."""

from datetime import date
from types import MappingProxyType

import pytest

from hedge_fund.universe import GICS_SECTORS, Universe, load_universe, parse_universe

VALID = "# list_date: 2026-10-01\nticker,gics_sector\nAAA,Energy\nBRK-B,Financials\n"
VALID_WITH_CIK = (
    "# list_date: 2026-10-01\n"
    "ticker,gics_sector,cik\n"
    "AAPL,Information Technology,0000320193\n"
    "NEWCO,Energy,\n"
)


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


def test_parse_universe_names_universe_and_value_for_bad_list_date() -> None:
    text = "# list_date: soon\nticker,gics_sector\nAAA,Energy\n"
    with pytest.raises(ValueError, match=r"universe 'demo'.*list_date.*'soon'"):
        parse_universe("demo", text)


def test_parse_universe_names_universe_and_line_for_row_missing_a_column() -> None:
    text = "# list_date: 2026-10-01\nticker,gics_sector\nAAA,Energy\nXYZ\n"
    with pytest.raises(ValueError, match=r"universe 'demo'.*line 4"):
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


def test_parse_universe_reads_cik_column_and_omits_empty_cells() -> None:
    universe = parse_universe("demo", VALID_WITH_CIK)
    assert universe.tickers == ("AAPL", "NEWCO")
    assert dict(universe.sectors) == {"AAPL": "Information Technology", "NEWCO": "Energy"}
    assert dict(universe.ciks) == {"AAPL": "0000320193"}
    assert isinstance(universe.ciks, MappingProxyType)


def test_two_column_universe_has_no_ciks() -> None:
    assert dict(parse_universe("demo", VALID).ciks) == {}


def test_universe_ciks_default_to_an_empty_read_only_mapping() -> None:
    universe = Universe(
        name="demo",
        list_date=date(2026, 10, 1),
        tickers=("AAA",),
        sectors=MappingProxyType({"AAA": "Energy"}),
    )
    assert isinstance(universe.ciks, MappingProxyType)
    assert dict(universe.ciks) == {}


@pytest.mark.parametrize("cik", ["320193", "00003201930", "000032019A", "٠" * 10])
def test_parse_universe_rejects_malformed_cik(cik: str) -> None:
    text = f"# list_date: 2026-10-01\nticker,gics_sector,cik\nAAPL,Information Technology,{cik}\n"
    with pytest.raises(ValueError, match=r"universe 'demo'.*cik"):
        parse_universe("demo", text)


def test_three_column_row_missing_the_cik_column_is_rejected() -> None:
    text = "# list_date: 2026-10-01\nticker,gics_sector,cik\nAAPL,Information Technology\n"
    with pytest.raises(ValueError, match=r"universe 'demo'.*line 3"):
        parse_universe("demo", text)
