"""Regenerate src/hedge_fund/resources/universes/sp500_membership.csv (dev only, needs network).

Run: pixi run -e dev python scripts/build_sp500_membership.py
Source: fja05680/sp500's historical components file at the commit pinned below (see the
Layer 8 PR 1 plan, Task 0). Each row is a date and the full member list from that date;
tickers become Yahoo form (BRK.B -> BRK-B). Spans are [start, end) dataset snapshot dates:
each is the first snapshot on or after the effective date (used as-is, decision D2).
"""

from __future__ import annotations

import csv
import io
import sys
from datetime import date
from pathlib import Path
from urllib.parse import quote

import requests

from hedge_fund.adapters.data._atomic import atomic_write
from hedge_fund.engine.backtest.membership import format_membership, spans_from_compositions

SOURCE_REPO = "https://github.com/fja05680/sp500"
SOURCE_COMMIT = "a2430f2af0c79ddf0748e91de11bdeb1616ab5a7"
SOURCE_FILE = "S&P 500 Historical Components & Changes (Updated).csv"
LICENCE = "MIT"
DATE_COLUMN = "date"
TICKERS_COLUMN = "tickers"
REPO_ROOT = Path(__file__).resolve().parents[1]
TARGET = REPO_ROOT / "src" / "hedge_fund" / "resources" / "universes" / "sp500_membership.csv"
MIN_MEMBERS = 495
MAX_MEMBERS = 510


def to_yahoo(symbol: str) -> str:
    """Yahoo uses '-' for share classes: BRK.B -> BRK-B."""
    return symbol.strip().upper().replace(".", "-")


def compositions(text: str) -> list[tuple[date, frozenset[str]]]:
    """(date, full member list) rows of the source CSV, sorted by date."""
    rows = []
    for row in csv.DictReader(io.StringIO(text)):
        members = frozenset(
            to_yahoo(symbol) for symbol in row[TICKERS_COLUMN].split(",") if symbol.strip()
        )
        rows.append((date.fromisoformat(row[DATE_COLUMN].strip()), members))
    return sorted(rows, key=lambda row: row[0])


def main() -> int:
    url = f"https://raw.githubusercontent.com/fja05680/sp500/{SOURCE_COMMIT}/{quote(SOURCE_FILE)}"
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    rows = compositions(response.text)
    latest = len(rows[-1][1])
    if not MIN_MEMBERS <= latest <= MAX_MEMBERS:
        print(f"latest composition has {latest} members; refusing to write", file=sys.stderr)
        return 1
    spans = spans_from_compositions(rows)
    provenance = [
        f"source: {SOURCE_REPO} @ {SOURCE_COMMIT} ({SOURCE_FILE})",
        f"licence: {LICENCE}; spans are [start, end) first dataset snapshot on or after "
        f"the effective date; built {date.today().isoformat()}",
    ]
    body = format_membership(spans, provenance)
    atomic_write(TARGET, lambda tmp: tmp.write_text(body, encoding="utf-8"))
    tickers = len({span.ticker for span in spans})
    print(f"wrote {len(spans)} spans for {tickers} tickers to {TARGET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
