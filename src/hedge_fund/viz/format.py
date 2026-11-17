"""Display formatting shared by the dashboard and the site. Pure: no Streamlit, no I/O.

``viz/views.py`` passes NaN and infinities through untouched; how they read is the page
layer's decision, so every number a page prints goes through ``fmt_number`` and friends.
The SPY notes take their no-SPY wording from the caller (``no_spy``): the dashboard points
at its sidebar, the site has none.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

from hedge_fund.viz.views import MIN_CURVE_POINTS, BacktestRun, OverviewView

NOT_AVAILABLE = "n/a"
MARKDOWN_SPECIAL = frozenset("\\`*_{}[]()#+-.!|<>~$&:")
Window = tuple[date, date]
WHOLE_HISTORY = "SPY covers the whole NAV history"


@dataclass(frozen=True)
class Headline:
    label: str
    value: str
    hint: str | None = None


def escape_markdown(text: str) -> str:
    """``text`` (LLM output or report text) backslash-escaped so ``st.markdown`` shows it
    literally: no emphasis, links, code, headings or ``$`` LaTeX.
    """
    return "".join(f"\\{char}" if char in MARKDOWN_SPECIAL else char for char in text)


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


def chart_note(window: Window | None, partial: bool, *, no_spy: str) -> str:
    """What the SPY line on a NAV chart covers (the strategy line is always the full NAV).

    With no SPY series (``window`` is None) the caller's ``no_spy`` text comes back
    unchanged: the dashboard and the site word that case differently.
    """
    if window is None:
        return no_spy
    if partial:
        return (
            f"SPY covers only {fmt_window(window)} (partial window): the chart rebases it to "
            f"the strategy at {fmt_date(window[0])} while the strategy line stays full length."
        )
    return f"{WHOLE_HISTORY}, {fmt_window(window)}."


def return_note(window: Window | None, partial: bool, *, has_return: bool, no_spy: str) -> str:
    """``chart_note`` plus what the SPY-window returns mean; ``has_return`` is False when the
    window has fewer than ``MIN_CURVE_POINTS`` dates, so no return exists. With no SPY series
    this is exactly ``no_spy``.
    """
    note = chart_note(window, partial, no_spy=no_spy)
    if window is None:
        return note
    if not has_return:
        return f"{note} That is fewer than {MIN_CURVE_POINTS} dates, so no SPY return is computed."
    return (
        f"{note} Returns marked (SPY window) are over that span; the full-history return "
        "is over the whole NAV."
    )


def headlines(view: OverviewView) -> tuple[Headline, ...]:
    """The Overview figures, each labelled with the span it covers."""
    outcome = NOT_AVAILABLE if view.last_cycle_outcome is None else view.last_cycle_outcome.value
    spy_window = f"SPY window: {fmt_window(view.benchmark_window)}"
    return (
        Headline("NAV", fmt_money(view.nav), f"As of {fmt_date(view.as_of)}"),
        Headline("Strategy return (full history)", fmt_percent(view.total_return)),
        Headline("Max drawdown", fmt_percent(view.max_drawdown), "Full history"),
        Headline("SPY return (SPY window)", fmt_percent(view.benchmark_return), spy_window),
        Headline(
            "Strategy return (SPY window)",
            fmt_percent(view.strategy_return_in_benchmark_window),
            spy_window,
        ),
        Headline("Last cycle", outcome, f"Cycle of {fmt_date(view.last_cycle_as_of)}"),
    )


def run_metrics(run: BacktestRun) -> tuple[Headline, ...]:
    """One backtest run's stored full-window figures."""
    window = f"Run window: {fmt_date(run.start)} to {fmt_date(run.end)}"
    return (
        Headline("Return (full window)", fmt_percent(run.total_return), window),
        Headline("SPY (full window)", fmt_percent(run.benchmark_return), window),
        Headline("Excess (full window)", fmt_percent(run.excess_return), window),
        Headline("Max drawdown (full window)", fmt_percent(run.max_drawdown), window),
        Headline("Sharpe (full window)", fmt_number(run.sharpe), window),
        Headline("Beta (full window)", fmt_number(run.beta), window),
        Headline("Breaker days", str(run.breaker_days), "Daily-loss plus drawdown breaker days"),
        Headline("Trading days", str(run.trading_days), window),
    )
