"""jsonable: the shared JSON shape of the report stores."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, date, datetime
from enum import Enum

import pytest

from hedge_fund.adapters._json import decode_float, jsonable
from hedge_fund.domain.orders import OrderAction


class _Plain(Enum):
    ONE = 1


@dataclass(frozen=True)
class _Inner:
    when: date
    weight: float


@dataclass(frozen=True)
class _Outer:
    zeta: str
    alpha: _Inner


def test_dataclasses_become_objects_in_field_order() -> None:
    body = jsonable(_Outer(zeta="z", alpha=_Inner(when=date(2026, 9, 30), weight=0.5)))
    assert body == {"zeta": "z", "alpha": {"when": "2026-09-30", "weight": 0.5}}
    assert list(body) == ["zeta", "alpha"]


def test_enums_become_values() -> None:
    assert jsonable([OrderAction.SELL_SHORT, _Plain.ONE]) == ["sell_short", 1]


def test_dates_and_datetimes_become_iso_strings() -> None:
    at = datetime(2026, 10, 1, 13, 30, tzinfo=UTC)
    assert jsonable((date(2026, 10, 1), at)) == ["2026-10-01", "2026-10-01T13:30:00+00:00"]


@pytest.mark.parametrize(
    ("value", "text"), [(math.nan, "nan"), (math.inf, "inf"), (-math.inf, "-inf")]
)
def test_non_finite_floats_become_strings(value: float, text: str) -> None:
    assert jsonable(value) == text


def test_finite_floats_and_other_scalars_pass_through() -> None:
    assert jsonable([1.5, 2, "x", None, True]) == [1.5, 2, "x", None, True]


def test_mappings_and_sequences_recurse() -> None:
    assert jsonable({1: (math.inf, date(2026, 1, 2))}) == {"1": ["inf", "2026-01-02"]}


def test_a_dataclass_type_is_not_expanded() -> None:
    assert jsonable(_Inner) is _Inner


@pytest.mark.parametrize(("value", "expected"), [(1.5, 1.5), (2, 2.0)])
def test_decode_float_takes_json_numbers(value: float, expected: float) -> None:
    decoded = decode_float(value)
    assert decoded == expected
    assert type(decoded) is float


@pytest.mark.parametrize(("text", "expected"), [("inf", math.inf), ("-inf", -math.inf)])
def test_decode_float_turns_infinities_back(text: str, expected: float) -> None:
    assert decode_float(text) == expected


def test_decode_float_turns_nan_back() -> None:
    assert math.isnan(decode_float("nan"))


def test_decode_float_refuses_other_strings() -> None:
    with pytest.raises(ValueError, match="'1.5' is not a float"):
        decode_float("1.5")


@pytest.mark.parametrize("value", [True, None, [1.0]])
def test_decode_float_refuses_non_numbers(value: object) -> None:
    with pytest.raises(TypeError, match="is not a float"):
        decode_float(value)


def test_decode_float_inverts_jsonable() -> None:
    for value in (0.25, math.inf, -math.inf):
        assert decode_float(jsonable(value)) == value
