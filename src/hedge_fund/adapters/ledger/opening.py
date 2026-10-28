"""The opening balance as JSON: ``<root>/ledger/opening.json``, written by the user.

A list of ``{"ticker", "shares", "price", "date"}``: the holdings before the system's first
fill (not today's broker positions, which would count the fills twice).
"""

from __future__ import annotations

import json
import math
from datetime import date
from pathlib import Path
from typing import Any

from hedge_fund.domain.ledger import OpeningEntry
from hedge_fund.engine.ledger.symbols import canonical_symbol

OPENING_PATH = Path("ledger", "opening.json")

_KEYS = frozenset({"ticker", "shares", "price", "date"})


class JsonOpeningBalance:
    """OpeningBalance reading ``<root>/ledger/opening.json``; ``()`` when there is no file."""

    def __init__(self, root: Path) -> None:
        self._path = Path(root) / OPENING_PATH

    def load(self) -> tuple[OpeningEntry, ...]:
        """The entries in file order, tickers canonicalised (``BRK.B`` -> ``BRK-B``).

        Raises ValueError naming the file for invalid JSON, a body that is not a list, an
        entry without exactly the four keys, an empty ticker, ``shares`` that is not a
        non-zero integer, a ``price`` that is not a finite positive number, a ``date`` that
        is not ISO, or a (canonical) ticker listed twice.
        """
        if not self._path.exists():
            return ()
        name = self._path.name
        try:
            body = json.loads(self._path.read_text(encoding="utf-8"))
        except ValueError as exc:
            raise ValueError(f"{name}: invalid JSON: {exc}") from exc
        if not isinstance(body, list):
            raise ValueError(f"{name}: must be a list of entries")
        entries = tuple(_entry(item, f"{name} entry {index}") for index, item in enumerate(body))
        seen: set[str] = set()
        for entry in entries:
            if entry.ticker in seen:
                raise ValueError(f"{name}: {entry.ticker} appears more than once")
            seen.add(entry.ticker)
        return entries


def _entry(item: Any, where: str) -> OpeningEntry:
    if not isinstance(item, dict) or set(item) != _KEYS:
        raise ValueError(f"{where}: must have exactly the keys date, price, shares, ticker")
    return OpeningEntry(
        ticker=_ticker(item["ticker"], where),
        shares=_shares(item["shares"], where),
        price=_price(item["price"], where),
        on=_day(item["date"], where),
    )


def _ticker(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{where}: ticker must be a non-empty string")
    return canonical_symbol(value)


def _shares(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value == 0:
        raise ValueError(f"{where}: shares must be a non-zero integer, got {value!r}")
    return value


def _price(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{where}: price must be a number, got {value!r}")
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{where}: price must be finite and positive, got {value!r}")
    return float(value)


def _day(value: Any, where: str) -> date:
    if not isinstance(value, str):
        raise ValueError(f"{where}: date must be an ISO date string, got {value!r}")
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{where}: date {value!r} is not an ISO date") from exc
