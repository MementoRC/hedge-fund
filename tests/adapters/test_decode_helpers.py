"""Decoders shared by the report loaders."""

from __future__ import annotations

import math
from datetime import UTC, date, datetime, timedelta, timezone
from typing import Any

import pytest

from hedge_fund.adapters._json import decode_float, decode_instant, decode_optional, jsonable


def test_decode_instant_keeps_the_offset() -> None:
    assert decode_instant("2024-01-02T21:30:00+00:00") == datetime(2024, 1, 2, 21, 30, tzinfo=UTC)


def test_decode_instant_refuses_a_naive_time() -> None:
    with pytest.raises(ValueError, match="no UTC offset"):
        decode_instant("2024-01-02T21:30:00")


@pytest.mark.parametrize(
    "instant",
    [
        datetime(2024, 1, 2, 21, 30, tzinfo=UTC),
        datetime(2024, 1, 2, 21, 30, tzinfo=timezone(timedelta(hours=5, minutes=30))),
    ],
)
def test_decode_instant_round_trips_jsonable(instant: datetime) -> None:
    decoded = decode_instant(jsonable(instant))

    assert decoded == instant
    assert decoded.utcoffset() == instant.utcoffset()


def test_decode_instant_reads_a_z_suffix_as_utc() -> None:
    assert decode_instant("2024-01-02T21:30:00Z") == datetime(2024, 1, 2, 21, 30, tzinfo=UTC)


def _must_not_run(_: Any) -> None:
    raise AssertionError("decode called for None")


def test_decode_optional_passes_none_through_and_decodes_the_rest() -> None:
    assert decode_optional(None, _must_not_run) is None
    assert decode_optional("2024-01-02", date.fromisoformat) == date(2024, 1, 2)
    decoded = decode_optional("nan", decode_float)
    assert decoded is not None
    assert math.isnan(decoded)
