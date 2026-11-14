"""Display formatting: non-finite and missing numbers read "n/a", windows are named."""

from __future__ import annotations

import math
from datetime import date

import pytest

from hedge_fund.dashboard.format import (
    NOT_AVAILABLE,
    chart_note,
    fmt_date,
    fmt_money,
    fmt_number,
    fmt_percent,
    fmt_window,
    return_note,
)

WINDOW = (date(2024, 1, 3), date(2024, 1, 5))
NON_FINITE = [math.nan, math.inf, -math.inf, None]


def test_the_placeholder_is_n_a() -> None:
    assert NOT_AVAILABLE == "n/a"


@pytest.mark.parametrize("value", NON_FINITE)
def test_fmt_number_shows_n_a_for_nan_infinities_and_none(value: float | None) -> None:
    assert fmt_number(value) == "n/a"
    assert fmt_number(value, ".4f") == "n/a"


def test_fmt_number_formats_finite_values() -> None:
    assert fmt_number(1234.5) == "1,234.50"
    assert fmt_number(0.0) == "0.00"
    assert fmt_number(-2.345, ".1f") == "-2.3"


@pytest.mark.parametrize("value", NON_FINITE)
def test_fmt_percent_shows_n_a_for_nan_infinities_and_none(value: float | None) -> None:
    assert fmt_percent(value) == "n/a"


def test_fmt_percent_formats_finite_values() -> None:
    assert fmt_percent(0.1234) == "12.3%"
    assert fmt_percent(-0.05) == "-5.0%"
    assert fmt_percent(0.2, digits=2) == "20.00%"
    assert fmt_percent(0.0) == "0.0%"


@pytest.mark.parametrize("value", NON_FINITE)
def test_fmt_money_shows_n_a_for_nan_infinities_and_none(value: float | None) -> None:
    assert fmt_money(value) == "n/a"


def test_fmt_money_puts_the_sign_before_the_dollar_sign() -> None:
    assert fmt_money(1234.5) == "$1,234.50"
    assert fmt_money(-1234.5) == "-$1,234.50"
    assert fmt_money(0.0) == "$0.00"


def test_fmt_date_and_window() -> None:
    assert fmt_date(date(2024, 1, 3)) == "2024-01-03"
    assert fmt_date(None) == "n/a"
    assert fmt_window(WINDOW) == "2024-01-03 to 2024-01-05"
    assert fmt_window(None) == "n/a"


def test_chart_note_without_a_benchmark_says_there_is_no_spy_series() -> None:
    note = chart_note(None, partial=False)

    assert "No SPY series" in note
    assert "<root>.store" in note


def test_chart_note_for_a_full_benchmark_says_it_covers_the_whole_history() -> None:
    note = chart_note(WINDOW, partial=False)

    assert "whole NAV history" in note
    assert "2024-01-03 to 2024-01-05" in note
    assert "partial" not in note


def test_chart_note_for_a_partial_benchmark_names_the_window_and_the_rebase() -> None:
    note = chart_note(WINDOW, partial=True)

    assert "partial window" in note
    assert "2024-01-03 to 2024-01-05" in note
    assert "rebases it to the strategy at 2024-01-03" in note
    assert "strategy line stays full length" in note


def test_return_note_without_a_benchmark_is_the_chart_note() -> None:
    assert return_note(None, partial=False, has_return=False) == chart_note(None, partial=False)


def test_return_note_with_a_window_but_no_return_says_fewer_than_three_dates() -> None:
    note = return_note(WINDOW, partial=True, has_return=False)

    assert "partial window" in note
    assert "fewer than 3 dates" in note
    assert "no SPY return is computed" in note


def test_return_note_with_a_return_labels_the_two_spans() -> None:
    note = return_note(WINDOW, partial=False, has_return=True)

    assert "whole NAV history" in note
    assert "(SPY window)" in note
    assert "full-history return" in note
