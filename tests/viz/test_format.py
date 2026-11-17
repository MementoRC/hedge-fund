"""Shared display formatting (viz/format.py): "n/a" for non-finite numbers, named windows."""

from __future__ import annotations

import math
from datetime import date

import numpy as np
import pytest

from hedge_fund.dashboard.format import NO_SPY
from hedge_fund.viz.format import (
    NOT_AVAILABLE,
    WHOLE_HISTORY,
    chart_note,
    escape_markdown,
    fmt_date,
    fmt_money,
    fmt_number,
    fmt_percent,
    fmt_window,
    return_note,
)
from hedge_fund.viz.views import MIN_CURVE_POINTS

WINDOW = (date(2024, 1, 3), date(2024, 1, 5))
NON_FINITE = [math.nan, math.inf, -math.inf, None]
NO_SPY_TEXT = "No SPY here."


def test_the_placeholder_is_n_a() -> None:
    assert NOT_AVAILABLE == "n/a"


def test_escape_markdown_makes_llm_text_literal() -> None:
    assert escape_markdown("$5 *bold*") == r"\$5 \*bold\*"
    assert escape_markdown(r"a\b [x](y) `z` _u_ # h") == r"a\\b \[x\]\(y\) \`z\` \_u\_ \# h"
    assert escape_markdown("Buy: wide moat") == r"Buy\: wide moat"
    assert escape_markdown(":red[x] :rocket:") == r"\:red\[x\] \:rocket\:"
    assert escape_markdown("AT&amp;T") == r"AT\&amp;T"  # no HTML entity is decoded


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


@pytest.mark.parametrize(
    ("value", "number", "percent", "money"),
    [
        (-0.0, "0.00", "0.0%", "$0.00"),
        (-0.001, "0.00", "-0.1%", "$0.00"),
        (-0.0001, "0.00", "0.0%", "$0.00"),
        (-0.004, "0.00", "-0.4%", "$0.00"),
        (-0.006, "-0.01", "-0.6%", "-$0.01"),
        (1e300, f"{1e300:,.2f}", f"{1e300:.1%}", f"${1e300:,.2f}"),
        (-1e300, f"{-1e300:,.2f}", f"{-1e300:.1%}", f"-${1e300:,.2f}"),
        (np.float64(-0.0), "0.00", "0.0%", "$0.00"),
        (np.float64(-0.006), "-0.01", "-0.6%", "-$0.01"),
        (np.float64(1234.5), "1,234.50", "123450.0%", "$1,234.50"),
        (np.float64("nan"), "n/a", "n/a", "n/a"),
        (np.float64("-inf"), "n/a", "n/a", "n/a"),
    ],
)
def test_rounding_to_zero_drops_the_sign_and_exotic_floats_format(
    value: float, number: str, percent: str, money: str
) -> None:
    assert fmt_number(value) == number
    assert fmt_percent(value) == percent
    assert fmt_money(value) == money


def test_fmt_date_and_window() -> None:
    assert fmt_date(date(2024, 1, 3)) == "2024-01-03"
    assert fmt_date(None) == "n/a"
    assert fmt_window(WINDOW) == "2024-01-03 to 2024-01-05"
    assert fmt_window(None) == "n/a"


def test_chart_note_without_a_benchmark_is_exactly_the_no_spy_text_passed_in() -> None:
    assert chart_note(None, partial=False, no_spy=NO_SPY_TEXT) == NO_SPY_TEXT
    assert chart_note(None, partial=True, no_spy=NO_SPY) == NO_SPY


def test_the_dashboard_no_spy_text_names_the_store_and_the_sidebar() -> None:
    assert NO_SPY.startswith("No SPY series for this root.")
    assert "<root>.store" in NO_SPY
    assert "sidebar" in NO_SPY


def test_chart_note_for_a_full_benchmark_says_it_covers_the_whole_history() -> None:
    note = chart_note(WINDOW, partial=False, no_spy=NO_SPY_TEXT)

    assert note == "SPY covers the whole NAV history, 2024-01-03 to 2024-01-05."


def test_chart_note_for_a_partial_benchmark_names_the_window_and_the_rebase() -> None:
    note = chart_note(WINDOW, partial=True, no_spy=NO_SPY_TEXT)

    assert note == (
        "SPY covers only 2024-01-03 to 2024-01-05 (partial window): the chart rebases it to "
        "the strategy at 2024-01-03 while the strategy line stays full length."
    )


def test_return_note_without_a_benchmark_is_exactly_the_no_spy_text() -> None:
    assert return_note(None, partial=False, has_return=False, no_spy=NO_SPY_TEXT) == NO_SPY_TEXT
    assert return_note(None, partial=True, has_return=True, no_spy=NO_SPY) == NO_SPY


def test_return_note_with_a_window_but_no_return_says_fewer_than_three_dates() -> None:
    note = return_note(WINDOW, partial=True, has_return=False, no_spy=NO_SPY_TEXT)

    assert note == (
        f"{chart_note(WINDOW, partial=True, no_spy=NO_SPY_TEXT)} That is fewer than "
        f"{MIN_CURVE_POINTS} dates, so no SPY return is computed."
    )


def test_return_note_with_a_return_labels_the_two_spans() -> None:
    note = return_note(WINDOW, partial=False, has_return=True, no_spy=NO_SPY_TEXT)

    assert note.startswith(WHOLE_HISTORY)
    assert note.endswith(
        "Returns marked (SPY window) are over that span; the full-history return is over the "
        "whole NAV."
    )
