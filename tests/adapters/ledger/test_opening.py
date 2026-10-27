"""Tests for the JSON opening balance (holdings before the system's first fill)."""

from __future__ import annotations

import json
import math
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from hedge_fund.adapters.ledger.opening import JsonOpeningBalance
from hedge_fund.domain.ledger import OpeningEntry
from hedge_fund.ports.ledger import OpeningBalance

GOOD = {"ticker": "AAA", "shares": 10, "price": 90.5, "date": "2026-09-01"}


def _write(root: Path, body: Any) -> None:
    path = root / "ledger" / "opening.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    text = body if isinstance(body, str) else json.dumps(body)  # json.dumps writes NaN as NaN
    path.write_text(text, encoding="utf-8")


def _entry(**changes: Any) -> dict[str, Any]:
    return {**GOOD, **changes}


def _without(key: str) -> dict[str, Any]:
    entry = dict(GOOD)
    del entry[key]
    return entry


def test_no_file_is_no_entries(tmp_path: Path) -> None:
    opening: OpeningBalance = JsonOpeningBalance(tmp_path)
    assert opening.load() == ()


def test_entries_load_with_canonical_tickers(tmp_path: Path) -> None:
    _write(
        tmp_path,
        [GOOD, {"ticker": "BRK.B", "shares": -5, "price": 400, "date": "2026-08-31"}],
    )
    entries = JsonOpeningBalance(tmp_path).load()
    assert entries == (
        OpeningEntry(ticker="AAA", shares=10, price=90.5, on=date(2026, 9, 1)),
        OpeningEntry(ticker="BRK-B", shares=-5, price=400.0, on=date(2026, 8, 31)),
    )
    assert type(entries[1].price) is float


@pytest.mark.parametrize(
    "body",
    [
        pytest.param("not json", id="not-json"),
        pytest.param({"ticker": "AAA"}, id="not-a-list"),
        pytest.param(["AAA"], id="entry-not-an-object"),
        pytest.param([_without("price")], id="missing-key"),
        pytest.param([_entry(note="x")], id="extra-key"),
        pytest.param([_entry(ticker="")], id="empty-ticker"),
        pytest.param([_entry(ticker=5)], id="non-string-ticker"),
        pytest.param([_entry(shares=0)], id="zero-shares"),
        pytest.param([_entry(shares=1.5)], id="fractional-shares"),
        pytest.param([_entry(shares=True)], id="bool-shares"),
        pytest.param([_entry(shares="10")], id="string-shares"),
        pytest.param([_entry(price=0)], id="zero-price"),
        pytest.param([_entry(price=-1.0)], id="negative-price"),
        pytest.param([_entry(price=math.nan)], id="nan-price"),
        pytest.param([_entry(price=math.inf)], id="inf-price"),
        pytest.param([_entry(price="90.5")], id="string-price"),
        pytest.param([_entry(price=True)], id="bool-price"),
        pytest.param([_entry(date="yesterday")], id="bad-date"),
        pytest.param([_entry(date=20260901)], id="numeric-date"),
    ],
)
def test_a_malformed_file_raises(tmp_path: Path, body: Any) -> None:
    _write(tmp_path, body)
    with pytest.raises(ValueError, match="opening.json"):
        JsonOpeningBalance(tmp_path).load()


@pytest.mark.parametrize("second", ["AAA", "aaa"], ids=["same", "same-canonical"])
def test_a_ticker_may_appear_once(tmp_path: Path, second: str) -> None:
    _write(tmp_path, [GOOD, _entry(ticker=second, shares=3)])
    with pytest.raises(ValueError, match="opening.json: AAA appears more than once"):
        JsonOpeningBalance(tmp_path).load()


def test_dot_and_dash_forms_are_the_same_ticker(tmp_path: Path) -> None:
    _write(tmp_path, [_entry(ticker="BRK.B"), _entry(ticker="BRK-B")])
    with pytest.raises(ValueError, match="BRK-B appears more than once"):
        JsonOpeningBalance(tmp_path).load()
