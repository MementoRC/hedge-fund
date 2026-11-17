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
from hedge_fund.viz.format import Headline, escape_site_markdown

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
