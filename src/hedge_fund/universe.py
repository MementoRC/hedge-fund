"""Shipped ticker universes with GICS sectors (package data under resources/universes/)."""

from __future__ import annotations

import csv
import io
import re
from collections.abc import Mapping
from dataclasses import dataclass
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
_UNIVERSE_NAME = re.compile(r"[a-z0-9_]+")


@dataclass(frozen=True)
class Universe:
    """A named ticker list as of `list_date`; tickers keep file order."""

    name: str
    list_date: date
    tickers: tuple[str, ...]
    sectors: Mapping[str, str]


def parse_universe(name: str, text: str) -> Universe:
    """Parse '# list_date: YYYY-MM-DD', then a 'ticker,gics_sector' CSV."""
    first, _, body = text.partition("\n")
    if not first.startswith(_LIST_DATE_PREFIX):
        raise ValueError(f"universe {name!r}: first line must be '{_LIST_DATE_PREFIX} YYYY-MM-DD'")
    list_date = date.fromisoformat(first.removeprefix(_LIST_DATE_PREFIX).strip())
    reader = csv.DictReader(io.StringIO(body))
    if reader.fieldnames != _HEADER:
        raise ValueError(f"universe {name!r}: header must be 'ticker,gics_sector'")
    sectors: dict[str, str] = {}
    for row in reader:
        ticker = row["ticker"].strip()
        sector = row["gics_sector"].strip()
        if ticker in sectors:
            raise ValueError(f"universe {name!r}: duplicate ticker {ticker}")
        if sector not in GICS_SECTORS:
            raise ValueError(f"universe {name!r}: unknown sector {sector!r} for {ticker}")
        sectors[ticker] = sector
    return Universe(
        name=name,
        list_date=list_date,
        tickers=tuple(sectors),
        sectors=MappingProxyType(sectors),
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
