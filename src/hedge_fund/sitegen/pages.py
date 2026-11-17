"""The generated result pages of the site, as Markdown. Pure: views in, text out.

Each chart is one raw HTML block with no blank line inside: an empty ``div.hf-chart`` and two
single-line ``application/json`` scripts with the figure in the light and the dark palette
(``charts.js`` draws the one matching the page scheme). The JSON has every less-than sign
and ampersand written as a JSON unicode escape, so no string in it can close the script.
This module is the only place that rounds chart numbers (to 6 decimals, in the view data and
again in the parsed figure JSON), so two builds of one root are byte-identical. Text that
comes from a store goes through ``escape_site_markdown``, and is always placed after a fixed
prefix (``- ``, a table pipe) so it never starts a line.
"""

from __future__ import annotations

import dataclasses
import json
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import plotly.graph_objects as go

from hedge_fund.sitegen.assets import CHART_CLASS, DARK_SUFFIX, LIGHT_SUFFIX
from hedge_fund.sitegen.palette import DARK_PALETTE, LIGHT_PALETTE, Palette
from hedge_fund.viz.charts import (
    drawdown_chart,
    exposure_chart,
    funnel_chart,
    llm_cost_chart,
    nav_vs_benchmark,
    sector_exposure_chart,
)
from hedge_fund.viz.format import (
    NOT_AVAILABLE,
    Headline,
    chart_note,
    escape_site_markdown,
    fmt_date,
    fmt_money,
    headlines,
    return_note,
    run_metrics,
)
from hedge_fund.viz.views import (
    BacktestView,
    CostRow,
    CycleRow,
    CycleView,
    OverviewView,
    PortfolioView,
)

DECIMALS = 6
MAX_LISTED = 10
DEMO_BANNER = '!!! warning "Synthetic demo data, not a real result."'
SITE_NO_SPY = "No SPY series for this root."
FIGURE_HEADER = ("Figure", "Value", "Note")
PORTFOLIO_PAGE = "portfolio.md"
CYCLES_PAGE = "cycles.md"
BACKTEST_PAGE = "backtests.md"
STATS_PAGE = "stats.md"
PAGE_FILES = (PORTFOLIO_PAGE, CYCLES_PAGE, BACKTEST_PAGE, STATS_PAGE)
NO_NAV = "No account snapshots in this root, so there is no NAV to chart."
NO_EXPOSURE = "No cycle in this root has a target portfolio, so there is no exposure to chart."
NO_SECTORS = "The latest target holds no position, so there is no sector exposure to chart."
NO_CYCLES = "No cycles in this root."
NO_COSTS = "No cycle in this root records an LLM budget or spend."
NO_FUNNEL = "The newest cycle stopped before the shortlist, so there is no funnel to chart."
FUNNEL_NOTE = (
    "The funnel starts at the shortlist: a cycle report stores no universe or quality-gate "
    "counts. Target positions are the target at rebalance; orders are the planned orders."
)
CYCLE_HEADER = (
    "Date",
    "Outcome",
    "Shortlist",
    "Researched",
    "Target positions",
    "Orders",
    "Risk vetoes",
    "Approval",
    "LLM budget",
    "LLM spent",
)
NO_BIAS = "The run lists no residual bias."  # the dashboard's wording (dashboard/backtests.py)
NO_EQUITY = "This root holds no account snapshots, so there is no equity curve to draw."
EQUITY_NOTE = (
    "Equity is this root's account history, indexed to 100; the run figures above are over "
    "each run's whole window."
)


@dataclass(frozen=True)
class PageContext:
    """What every page shows above its content: the banner (DEMO roots only) and the root's
    unreadable entries."""

    demo: bool
    unreadable: tuple[str, ...]


def render_page(title: str, context: PageContext, *sections: str) -> str:
    """One H1, the demo banner, the unreadable list, then the non-empty ``sections``."""
    parts = [
        f"# {title}",
        DEMO_BANNER if context.demo else "",
        unreadable_block(context.unreadable),
        *sections,
    ]
    return "\n\n".join(part for part in parts if part) + "\n"


def cell(text: str) -> str:
    """``text`` made literal inside a Markdown table cell."""
    return escape_site_markdown(text).replace("|", "\\|")


def table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = [_row(header), "|" + "---|" * len(header), *(_row(row) for row in rows)]
    return "\n".join(lines)


def headline_table(items: Sequence[Headline]) -> str:
    return table(FIGURE_HEADER, [(item.label, item.value, item.hint or "") for item in items])


def unreadable_entry(entry: str) -> str:
    """``<where>: <ExceptionType>: <message>`` shown as ``<where> (<ExceptionType>)``.

    The message is dropped (it can hold absolute paths), so the page is the same on every
    machine; an entry with no ``": "`` is shown whole. The result must be placed after a
    fixed prefix, never at the start of a line.
    """
    where, _, rest = entry.partition(": ")
    kind = rest.partition(": ")[0]
    shown = escape_site_markdown(where)
    return f"{shown} ({escape_site_markdown(kind)})" if kind else shown


def unreadable_block(entries: Sequence[str]) -> str:
    """The dashboard's unreadable style: a count, then up to ``MAX_LISTED`` entries."""
    if not entries:
        return ""
    count = len(entries)
    noun = "file" if count == 1 else "files"
    lines = [f"{count} unreadable {noun}; this page shows what could be read.", ""]
    lines += [f"- {unreadable_entry(entry)}" for entry in entries[:MAX_LISTED]]
    if count > MAX_LISTED:
        lines.append(f"- and {count - MAX_LISTED} more")
    return "\n".join(lines)


def portfolio_page(overview: OverviewView, view: PortfolioView, context: PageContext) -> str:
    """The Overview headlines and return note, then NAV vs SPY, drawdown, exposure, sectors."""
    shown: PortfolioView = rounded(view)
    has_nav = bool(shown.nav_index)
    note = return_note(
        overview.benchmark_window,
        overview.benchmark_partial,
        has_return=overview.benchmark_return is not None,
        no_spy=SITE_NO_SPY,
    )
    return render_page(
        "Portfolio",
        context,
        headline_table(headlines(overview)),
        note,
        "## NAV vs SPY",
        chart_or("portfolio-nav", has_nav, lambda p: nav_vs_benchmark(shown, p), NO_NAV),
        chart_note(view.benchmark_window, view.benchmark_partial, no_spy=SITE_NO_SPY),
        "## Drawdown",
        chart_or("portfolio-drawdown", has_nav, lambda p: drawdown_chart(shown, p), NO_NAV),
        "## Exposure",
        chart_or(
            "portfolio-exposure",
            bool(shown.exposure),
            lambda p: exposure_chart(shown, p),
            NO_EXPOSURE,
        ),
        "## Sector exposure",
        chart_or(
            "portfolio-sectors",
            bool(shown.sectors),
            lambda p: sector_exposure_chart(shown, p),
            NO_SECTORS,
        ),
    )


def cycles_page(view: CycleView, costs: Sequence[CostRow], context: PageContext) -> str:
    """Every cycle in load order, the newest cycle's funnel, LLM budget vs spend.

    The newest cycle has the greatest ``as_of``; on a tie the later one in load order wins
    (a static page has no picker).
    """
    if not view.rows:
        return render_page("Cycles", context, NO_CYCLES)
    newest = max(reversed(view.rows), key=lambda row: row.as_of)
    funnel = newest.funnel
    has_funnel = any((funnel.shortlist, funnel.researched, funnel.portfolio, funnel.orders))
    shown_costs: tuple[CostRow, ...] = rounded(tuple(costs))
    has_costs = any(row.budget is not None or row.spent is not None for row in shown_costs)
    return render_page(
        "Cycles",
        context,
        table(CYCLE_HEADER, [_cycle_cells(row) for row in view.rows]),
        f"## Funnel of the newest cycle ({fmt_date(newest.as_of)})",
        chart_or("cycles-funnel", has_funnel, lambda p: funnel_chart(funnel, p), NO_FUNNEL),
        FUNNEL_NOTE if has_funnel else "",  # render_page drops empty sections
        "## LLM budget vs spend",
        chart_or("cycles-llm-cost", has_costs, lambda p: llm_cost_chart(shown_costs, p), NO_COSTS),
    )


def backtest_page(view: BacktestView, context: PageContext) -> str:
    """The root's run, its full-window figures and biases, then root-level equity vs SPY.

    ``classify_root`` admits exactly one backtest summary per root, so the view holds one run
    and there is no run table.
    """
    if len(view.runs) != 1:
        raise ValueError(f"backtest root must have exactly one run, got {len(view.runs)}")
    run = view.runs[0]
    shown: BacktestView = rounded(view)
    biases = "\n".join(f"- {escape_site_markdown(bias)}" for bias in run.biases)
    return render_page(
        "Backtest",
        context,
        f"This root holds one backtest run, {escape_site_markdown(run.run_id)}.",
        headline_table(run_metrics(run)),
        "## Residual biases",
        biases or NO_BIAS,
        "## Equity vs SPY",
        *_equity_sections(shown),
    )


def _equity_sections(view: BacktestView) -> tuple[str, ...]:
    """The equity chart with its two captions; only ``NO_EQUITY`` when there is no NAV
    (the dashboard's early return)."""
    if not view.nav_index:
        return (NO_EQUITY,)
    return (
        chart_block("backtest-equity", lambda p: nav_vs_benchmark(view, p)),
        chart_note(view.benchmark_window, view.benchmark_partial, no_spy=SITE_NO_SPY),
        EQUITY_NOTE,
    )


def _cycle_cells(row: CycleRow) -> tuple[str, ...]:
    approval = NOT_AVAILABLE if row.approval is None else row.approval.value
    funnel = row.funnel
    return (
        fmt_date(row.as_of),
        row.outcome.value,
        str(funnel.shortlist),
        str(funnel.researched),
        str(funnel.portfolio),
        str(funnel.orders),
        str(len(row.vetoes)),
        approval,
        fmt_money(row.budget),
        fmt_money(row.spent),
    )


def rounded(value: Any) -> Any:
    """``value`` (a view, a tuple or a float) with every finite float rounded to 6 decimals."""
    if isinstance(value, float):
        return round(value, DECIMALS) if math.isfinite(value) else value
    if isinstance(value, tuple):
        return tuple(rounded(item) for item in value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        changes = {f.name: rounded(getattr(value, f.name)) for f in dataclasses.fields(value)}
        return dataclasses.replace(value, **changes)
    return value


def chart_json(figure: go.Figure) -> str:
    """``figure`` as compact JSON with plain floats rounded, non-finite ones null, and no raw
    less-than sign or ampersand (each written as a JSON unicode escape)."""
    parsed = json.loads(figure.to_json(engine="json"))
    parsed.setdefault("layout", {})["autosize"] = True
    text = json.dumps(_round_json(parsed), separators=(",", ":"), allow_nan=False)
    return text.replace("<", "\\u003c").replace("&", "\\u0026")


def chart_block(chart_id: str, draw: Callable[[Palette], go.Figure]) -> str:
    """The div and the light and dark JSON scripts for one chart."""
    return "\n".join(
        [
            f'<div id="{chart_id}" class="{CHART_CLASS}"></div>',
            _script(chart_id + LIGHT_SUFFIX, draw(LIGHT_PALETTE)),
            _script(chart_id + DARK_SUFFIX, draw(DARK_PALETTE)),
        ]
    )


def chart_or(chart_id: str, present: bool, draw: Callable[[Palette], go.Figure], empty: str) -> str:
    """The chart when there is data to draw, else the plain ``empty`` note."""
    return chart_block(chart_id, draw) if present else empty


def _row(cells: Sequence[str]) -> str:
    return "| " + " | ".join(cell(text) for text in cells) + " |"


def _script(script_id: str, figure: go.Figure) -> str:
    return f'<script type="application/json" id="{script_id}">{chart_json(figure)}</script>'


def _round_json(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, DECIMALS) if math.isfinite(value) else None
    if isinstance(value, list):
        return [_round_json(item) for item in value]
    if isinstance(value, dict):
        return {key: _round_json(item) for key, item in value.items()}
    return value
