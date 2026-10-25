"""Ticker normal form for comparing the universe (Yahoo ``BRK-B``) with Alpaca (``BRK.B``)."""

from __future__ import annotations


def canonical_symbol(symbol: str) -> str:
    """``symbol`` upper-cased with ``"."`` replaced by ``"-"``; idempotent."""
    return symbol.upper().replace(".", "-")
