"""Regenerate src/hedge_fund/resources/universes/sp500.csv (dev only, needs network).

Run: pixi run -e dev python scripts/refresh_sp500.py
Source: the datasets/s-and-p-500-companies constituents CSV (current members only; using
it for past dates carries survivorship bias, accepted for now).
"""

from __future__ import annotations

import csv
import io
import sys
from datetime import date
from pathlib import Path

import requests

from hedge_fund.universe import GICS_SECTORS

SOURCE_URL = (
    "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
)
REPO_ROOT = Path(__file__).resolve().parents[1]
TARGET = REPO_ROOT / "src" / "hedge_fund" / "resources" / "universes" / "sp500.csv"
MIN_CONSTITUENTS = 490


def to_yahoo(symbol: str) -> str:
    """Yahoo uses '-' for share classes: BRK.B -> BRK-B."""
    return symbol.strip().upper().replace(".", "-")


def main() -> int:
    response = requests.get(SOURCE_URL, timeout=30)
    response.raise_for_status()
    sectors: dict[str, str] = {}
    for row in csv.DictReader(io.StringIO(response.text)):
        ticker = to_yahoo(row["Symbol"])
        sector = row["GICS Sector"].strip()
        if sector not in GICS_SECTORS:
            print(f"unknown sector {sector!r} for {ticker}", file=sys.stderr)
            return 1
        sectors[ticker] = sector
    if len(sectors) < MIN_CONSTITUENTS:
        print(f"only {len(sectors)} constituents; refusing to write", file=sys.stderr)
        return 1
    lines = [f"# list_date: {date.today().isoformat()}", "ticker,gics_sector"]
    lines += [f"{ticker},{sectors[ticker]}" for ticker in sorted(sectors)]
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    TARGET.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {len(sectors)} tickers to {TARGET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
