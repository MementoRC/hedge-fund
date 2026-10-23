"""Number formats and rankings used by the approval summary."""

from __future__ import annotations

import math

import pytest

from hedge_fund.engine.approval.render import _money, _num, _pct, _top

NAN = math.nan
INF = math.inf


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (12.0, "12.00"),
        (9244.0, "9,244.00"),
        (1_234_567.891, "1,234,567.89"),
        (0.005, "0.01"),  # the float 0.005 is just above 0.005
        (-0.004, "0.00"),  # no negative zero
        (NAN, "n/a"),
        (INF, "n/a"),
        (-INF, "n/a"),
    ],
)
def test_money(value: float, expected: str) -> None:
    assert _money(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (2212.6, "+2,212.60"),
        (-1500.0, "-1,500.00"),
        (0.0, "+0.00"),
        (-0.0, "+0.00"),
        (-0.004, "+0.00"),
        (-0.006, "-0.01"),
        (NAN, "n/a"),
        (-INF, "n/a"),
    ],
)
def test_signed_money(value: float, expected: str) -> None:
    assert _money(value, signed=True) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0.118, "11.80%"),
        (0.0727, "7.27%"),
        (1.5, "150.00%"),
        (0.00004, "0.00%"),
        (-0.00004, "0.00%"),
        (-0.021, "-2.10%"),
        (NAN, "n/a"),
        (INF, "n/a"),
        (1e308, "n/a"),  # finite, but x 100 overflows
    ],
)
def test_pct(value: float, expected: str) -> None:
    assert _pct(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0.0042, "+0.42%"),
        (-0.0031, "-0.31%"),
        (0.000051, "+0.01%"),
        (0.0, "+0.00%"),
        (-0.0, "+0.00%"),
        (-0.00004, "+0.00%"),
        (NAN, "n/a"),
        (-INF, "n/a"),
    ],
)
def test_signed_pct(value: float, expected: str) -> None:
    assert _pct(value, signed=True) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0.15, "0.1500"),
        (0.123456, "0.1235"),
        (1234.5, "1234.5000"),
        (-0.25, "-0.2500"),
        (-0.00001, "0.0000"),
        (NAN, "n/a"),
        (INF, "n/a"),
    ],
)
def test_num(value: float, expected: str) -> None:
    assert _num(value) == expected


def test_top_ranks_by_absolute_value_then_name() -> None:
    values = {"B": 0.01, "A": -0.01, "C": -0.03, "D": 0.02}
    assert _top(values, 3) == "C -3.00%, D +2.00%, A -1.00%"


def test_top_cuts_off_at_the_limit() -> None:
    values = {"A": 0.06, "B": 0.05, "C": 0.04, "D": 0.03, "E": 0.02, "F": 0.01}
    assert _top(values, 5) == "A +6.00%, B +5.00%, C +4.00%, D +3.00%, E +2.00%"


def test_top_of_nothing() -> None:
    assert _top({}, 5) == "(none)"
