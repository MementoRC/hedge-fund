"""Decoders shared by the report loaders."""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest

from hedge_fund.adapters._json import decode_float, decode_instant, decode_optional


def test_decode_instant_keeps_the_offset() -> None:
    assert decode_instant("2024-01-02T21:30:00+00:00") == datetime(2024, 1, 2, 21, 30, tzinfo=UTC)


def test_decode_instant_refuses_a_naive_time() -> None:
    with pytest.raises(ValueError, match="no UTC offset"):
        decode_instant("2024-01-02T21:30:00")


def test_decode_optional_passes_none_through_and_decodes_the_rest() -> None:
    assert decode_optional(None, date.fromisoformat) is None
    assert decode_optional("2024-01-02", date.fromisoformat) == date(2024, 1, 2)
    assert decode_optional("nan", decode_float) != decode_optional("nan", decode_float)  # nan
