"""Display formatting for the dashboard pages. Pure: no Streamlit, no I/O.

``viz/views.py`` passes NaN and infinities through untouched; how they read is the page
layer's decision, so every number a page prints goes through ``fmt_number`` and friends.
"""

from __future__ import annotations

import math
from datetime import date

from hedge_fund.viz.views import MIN_CURVE_POINTS

NOT_AVAILABLE = "n/a"
Window = tuple[date, date]
NO_SPY = (
    "No SPY series for this root. A demo root takes SPY from its sibling `<root>.store`; "
    "a problem loading it is listed in the sidebar."
)


def fmt_number(value: float | None, spec: str = ",.2f") -> str:
    """``value`` formatted with ``spec``; "n/a" for None, NaN, +inf and -inf.

    A value that rounds to zero at ``spec``'s precision shows no minus sign. ``bool`` is not
    a supported input.
    """
    if value is None or not math.isfinite(value):
        return NOT_AVAILABLE
    text = format(value, spec)
    if text.startswith("-") and not any(char in "123456789" for char in text):
        return text[1:]
    return text


def fmt_percent(value: float | None, digits: int = 1) -> str:
    """``value`` as a percentage with ``digits`` decimals; "n/a" when not finite."""
    return fmt_number(value, f".{digits}%")


def fmt_money(value: float | None) -> str:
    """Dollars with thousands separators; a negative sign leads the dollar sign."""
    text = fmt_number(value)
    if text == NOT_AVAILABLE:
        return text
    return f"-${text[1:]}" if text.startswith("-") else f"${text}"


def fmt_date(day: date | None) -> str:
    """ISO date; "n/a" for None."""
    return NOT_AVAILABLE if day is None else day.isoformat()


def fmt_window(window: Window | None) -> str:
    """``start to end`` for a ``(start, end)`` window; "n/a" for None."""
    return NOT_AVAILABLE if window is None else f"{fmt_date(window[0])} to {fmt_date(window[1])}"


def chart_note(window: Window | None, partial: bool) -> str:
    """What the SPY line on a NAV chart covers (the strategy line is always the full NAV)."""
    if window is None:
        return NO_SPY
    if partial:
        return (
            f"SPY covers only {fmt_window(window)} (partial window): the chart rebases it to "
            f"the strategy at {fmt_date(window[0])} while the strategy line stays full length."
        )
    return f"SPY covers the whole NAV history, {fmt_window(window)}."


def return_note(window: Window | None, partial: bool, *, has_return: bool) -> str:
    """``chart_note`` plus what the SPY-window returns mean; ``has_return`` is False when the
    window has fewer than ``MIN_CURVE_POINTS`` dates, so no return exists.
    """
    note = chart_note(window, partial)
    if window is None:
        return note
    if not has_return:
        return f"{note} That is fewer than {MIN_CURVE_POINTS} dates, so no SPY return is computed."
    return (
        f"{note} Returns marked (SPY window) are over that span; the full-history return "
        "is over the whole NAV."
    )
