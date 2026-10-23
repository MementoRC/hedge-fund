"""Shipped ticker universes: GICS sectors and SEC CIKs (package data in resources/universes/)."""

from __future__ import annotations

import csv
import io
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date
from importlib import resources
from types import MappingProxyType

GICS_SECTORS: frozenset[str] = frozenset(
    {
        "Communication Services",
        "Consumer Discretionary",
        "Consumer Staples",
        "Energy",
        "Financials",
        "Health Care",
        "Industrials",
        "Information Technology",
        "Materials",
        "Real Estate",
        "Utilities",
    }
)

_LIST_DATE_PREFIX = "# list_date:"
_HEADER = ["ticker", "gics_sector"]
_HEADER_WITH_CIK = [*_HEADER, "cik"]
_CIK = re.compile(r"\d{10}", re.ASCII)
_UNIVERSE_NAME = re.compile(r"[a-z0-9_]+")


def _no_ciks() -> Mapping[str, str]:
    return MappingProxyType({})


@dataclass(frozen=True)
class Universe:
    """A named ticker list as of `list_date`; tickers keep file order.

    `ciks` maps a ticker to its 10-digit SEC CIK; tickers without a CIK are absent.
    """

    name: str
    list_date: date
    tickers: tuple[str, ...]
    sectors: Mapping[str, str]
    ciks: Mapping[str, str] = field(default_factory=_no_ciks)


def parse_universe(name: str, text: str) -> Universe:
    """Parse '# list_date: YYYY-MM-DD', then a 'ticker,gics_sector[,cik]' CSV."""
    first, _, body = text.partition("\n")
    list_date = _list_date(name, first)
    reader = csv.DictReader(io.StringIO(body))
    if reader.fieldnames not in (_HEADER, _HEADER_WITH_CIK):
        raise ValueError(
            f"universe {name!r}: header must be 'ticker,gics_sector' or 'ticker,gics_sector,cik'"
        )
    columns = _HEADER_WITH_CIK if reader.fieldnames == _HEADER_WITH_CIK else _HEADER
    sectors: dict[str, str] = {}
    ciks: dict[str, str] = {}
    for row in reader:
        if any(row[column] is None for column in columns):
            raise ValueError(f"universe {name!r}: line {reader.line_num + 1} is missing a column")
        ticker = row["ticker"].strip()
        sector = row["gics_sector"].strip()
        if ticker in sectors:
            raise ValueError(f"universe {name!r}: duplicate ticker {ticker}")
        if sector not in GICS_SECTORS:
            raise ValueError(f"universe {name!r}: unknown sector {sector!r} for {ticker}")
        sectors[ticker] = sector
        cik = row.get("cik", "").strip()
        if cik:
            ciks[ticker] = _checked_cik(name, ticker, cik)
    return Universe(
        name=name,
        list_date=list_date,
        tickers=tuple(sectors),
        sectors=MappingProxyType(sectors),
        ciks=MappingProxyType(ciks),
    )


def load_universe(name: str) -> Universe:
    """Load resources/universes/<name>.csv shipped with the package."""
    if not _UNIVERSE_NAME.fullmatch(name):
        raise ValueError(f"unknown universe: {name!r}")
    universes = resources.files("hedge_fund").joinpath("resources").joinpath("universes")
    resource = universes.joinpath(f"{name}.csv")
    if not resource.is_file():
        raise ValueError(f"unknown universe: {name!r}")
    return parse_universe(name, resource.read_text(encoding="utf-8"))


def _list_date(name: str, first_line: str) -> date:
    if not first_line.startswith(_LIST_DATE_PREFIX):
        raise ValueError(f"universe {name!r}: first line must be '{_LIST_DATE_PREFIX} YYYY-MM-DD'")
    raw_date = first_line.removeprefix(_LIST_DATE_PREFIX).strip()
    try:
        return date.fromisoformat(raw_date)
    except ValueError:
        raise ValueError(
            f"universe {name!r}: list_date must be YYYY-MM-DD, got {raw_date!r}"
        ) from None


def _checked_cik(name: str, ticker: str, cik: str) -> str:
    if not _CIK.fullmatch(cik):
        raise ValueError(f"universe {name!r}: cik for {ticker} must be 10 digits, got {cik!r}")
    return cik
