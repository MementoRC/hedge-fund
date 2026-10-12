"""Regenerate src/hedge_fund/resources/universes/sp500.csv (dev only, needs network).

Run: SEC_USER_AGENT="Your Name you@example.com" pixi run -e dev python scripts/refresh_sp500.py
Sources: the datasets/s-and-p-500-companies constituents CSV (current members only; using
it for past dates carries survivorship bias, accepted for now) and SEC company_tickers.json
for each ticker's 10-digit CIK. SEC tickers use dashes for share classes (BRK-B), as our
Yahoo-format universe does, so the join is on the exact uppercase ticker.
"""

from __future__ import annotations

import csv
import io
import os
import sys
from collections.abc import Mapping, Sequence
from datetime import date
from pathlib import Path
from typing import Any

import requests

from hedge_fund.adapters.data._atomic import atomic_write
from hedge_fund.universe import GICS_SECTORS

SOURCE_URL = (
    "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
)
SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
REPO_ROOT = Path(__file__).resolve().parents[1]
TARGET = REPO_ROOT / "src" / "hedge_fund" / "resources" / "universes" / "sp500.csv"
MIN_CONSTITUENTS = 490
MAX_MISSING_CIK_PERCENT = 2


def to_yahoo(symbol: str) -> str:
    """Yahoo uses '-' for share classes: BRK.B -> BRK-B."""
    return symbol.strip().upper().replace(".", "-")


def sec_ciks(payload: Mapping[str, Mapping[str, Any]]) -> dict[str, str]:
    """Uppercase ticker -> 10-digit zero-padded CIK from company_tickers.json (first entry wins)."""
    ciks: dict[str, str] = {}
    for entry in payload.values():
        ciks.setdefault(str(entry["ticker"]).upper(), f"{int(entry['cik_str']):010d}")
    return ciks


def join_ciks(tickers: Sequence[str], ciks: Mapping[str, str]) -> tuple[dict[str, str], list[str]]:
    """Exact-ticker join: (ticker -> CIK for matches, sorted tickers without a CIK)."""
    found = {ticker: ciks[ticker] for ticker in tickers if ticker in ciks}
    missing = sorted(ticker for ticker in tickers if ticker not in ciks)
    return found, missing


def too_many_missing(missing: int, total: int) -> bool:
    """True when more than MAX_MISSING_CIK_PERCENT percent of `total` lack a CIK."""
    return missing * 100 > MAX_MISSING_CIK_PERCENT * total


def main() -> int:
    user_agent = os.environ.get("SEC_USER_AGENT")
    if not user_agent:
        print("SEC_USER_AGENT is not set (SEC requires 'Name email@example.com')", file=sys.stderr)
        return 1
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
    sec = requests.get(SEC_TICKERS_URL, headers={"User-Agent": user_agent}, timeout=30)
    sec.raise_for_status()
    ciks, missing = join_ciks(sorted(sectors), sec_ciks(sec.json()))
    if too_many_missing(len(missing), len(sectors)):
        print(
            f"{len(missing)} of {len(sectors)} constituents lack a CIK "
            f"(limit {MAX_MISSING_CIK_PERCENT}%): {', '.join(missing)}",
            file=sys.stderr,
        )
        return 1
    if missing:
        print(f"no CIK for: {', '.join(missing)}", file=sys.stderr)
    lines = [f"# list_date: {date.today().isoformat()}", "ticker,gics_sector,cik"]
    lines += [f"{ticker},{sectors[ticker]},{ciks.get(ticker, '')}" for ticker in sorted(sectors)]
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(lines) + "\n"
    atomic_write(TARGET, lambda tmp: tmp.write_text(body, encoding="utf-8"))
    print(f"wrote {len(sectors)} tickers ({len(ciks)} with CIK) to {TARGET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
