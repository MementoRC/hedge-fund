"""canonical_symbol: upper case, ``.`` -> ``-`` (Alpaca ``BRK.B`` is Yahoo ``BRK-B``)."""

from __future__ import annotations

import pytest

from hedge_fund.engine.ledger.symbols import canonical_symbol

CASES = [
    pytest.param("BRK.B", "BRK-B", id="dot-to-dash"),
    pytest.param("brk.b", "BRK-B", id="lower-dot"),
    pytest.param("BRK-B", "BRK-B", id="dash-form"),
    pytest.param("aapl", "AAPL", id="lower"),
    pytest.param("AAPL", "AAPL", id="plain"),
]


@pytest.mark.parametrize(("symbol", "expected"), CASES)
def test_canonical_symbol(symbol: str, expected: str) -> None:
    assert canonical_symbol(symbol) == expected


@pytest.mark.parametrize(("symbol", "expected"), CASES)
def test_canonical_symbol_is_idempotent(symbol: str, expected: str) -> None:
    assert canonical_symbol(canonical_symbol(symbol)) == expected
