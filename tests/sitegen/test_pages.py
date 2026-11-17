"""The page frame, escaping, unreadable listing, rounding and chart blocks of sitegen.pages."""

from __future__ import annotations

import json
import math
from pathlib import Path

import plotly
import plotly.graph_objects as go

from hedge_fund.sitegen.assets import CHART_CLASS
from hedge_fund.sitegen.pages import (
    DEMO_BANNER,
    MAX_LISTED,
    SITE_NO_SPY,
    PageContext,
    cell,
    chart_block,
    chart_json,
    chart_or,
    headline_table,
    render_page,
    rounded,
    table,
    unreadable_block,
    unreadable_entry,
)
from hedge_fund.sitegen.palette import DARK_PALETTE, LIGHT_PALETTE
from hedge_fund.viz.charts import drawdown_chart
from hedge_fund.viz.format import Headline, chart_note, return_note
from hedge_fund.viz.views import PortfolioView, portfolio_view
from tests.dashboard.builders import NAV_VALUES
from tests.sitegen.helpers import DEMO, PLAIN, chart_divs, figures, numbers, scripts
from tests.viz.builders import DAY0, make_snapshots

GOLDEN = Path(__file__).parent / "golden" / "portfolio-drawdown.html"
TILDES = "~~~python"


def golden_block() -> str:
    """The drawdown chart of NAV 100, 120, 90, 115 (the golden test's input)."""
    view = rounded(portfolio_view(make_snapshots(*NAV_VALUES), None, ()))
    return chart_block("portfolio-drawdown", lambda palette: drawdown_chart(view, palette))


def _figure(name: str = "line") -> go.Figure:
    return go.Figure(go.Scatter(x=[1, 2, 3], y=[1 / 3, math.nan, 2.0], name=name))


def test_a_page_is_one_h1_then_the_banner_on_a_demo_root_only() -> None:
    demo = render_page("Portfolio", DEMO, "Body.")
    plain = render_page("Portfolio", PLAIN, "Body.")

    assert demo == f"# Portfolio\n\n{DEMO_BANNER}\n\nBody.\n"
    assert plain == "# Portfolio\n\nBody.\n"
    assert DEMO_BANNER == '!!! warning "Synthetic demo data, not a real result."'


def test_empty_sections_are_dropped() -> None:
    assert render_page("Stats", PLAIN, "", "A.", "") == "# Stats\n\nA.\n"


def test_a_table_cell_escapes_the_pipe_and_the_text() -> None:
    assert cell("a|b") == r"a\|b"
    assert cell("<script>alert(1)</script>") == r"&lt;script&gt;alert\(1\)&lt;/script&gt;"
    assert cell("*x* & y") == r"\*x\* &amp; y"
    assert cell("a\r\n\tb") == "a b"


def test_a_table_is_a_header_a_rule_and_one_line_per_row() -> None:
    assert table(("A", "B"), [("a|b", "*x*"), ("1", "")]) == (
        "| A | B |\n|---|---|\n" + r"| a\|b | \*x\* |" + "\n| 1 |  |"
    )


def test_a_headline_table_shows_label_value_and_note() -> None:
    assert headline_table([Headline("NAV", "n/a", None), Headline("x", "1", "As of")]) == (
        "| Figure | Value | Note |\n|---|---|---|\n| NAV | n/a |  |\n| x | 1 | As of |"
    )


def test_an_unreadable_entry_shows_the_file_and_the_exception_type_only() -> None:
    entry = (
        "runs/2024-01-02.json: JSONDecodeError: Expecting value: line 1 column 1 (char 0) "
        "in /home/someone/secret/runs/2024-01-02.json"
    )

    assert unreadable_entry(entry) == r"runs/2024\-01\-02\.json (JSONDecodeError)"
    store = "store demo.store: FileNotFoundError: [Errno 2] '/tmp/x/demo.store/manifest.json'"
    assert unreadable_entry(store) == r"store demo\.store (FileNotFoundError)"


def test_a_label_only_entry_is_handled() -> None:
    assert unreadable_entry("account") == "account"
    assert unreadable_entry("account: ValueError") == "account (ValueError)"


def test_whitespace_in_an_unreadable_entry_becomes_one_space() -> None:
    assert unreadable_entry("runs/a\n\rb.json: Value\tError: x") == r"runs/a b\.json (Value Error)"


def test_the_unreadable_block_counts_lists_ten_and_drops_messages() -> None:
    entries = tuple(f"runs/{i}.json: ValueError: /abs/path/{i} secret" for i in range(12))
    block = unreadable_block(entries)

    lines = block.splitlines()
    assert lines[0] == "12 unreadable files; this page shows what could be read."
    assert lines[1] == ""
    assert lines[2] == r"- runs/0\.json (ValueError)"
    assert len(lines) == 2 + MAX_LISTED + 1
    assert lines[-1] == "- and 2 more"
    assert "/abs/path" not in block
    assert "secret" not in block


def test_one_unreadable_file_is_singular_and_none_is_empty() -> None:
    assert unreadable_block(("account: ValueError: x",)).startswith("1 unreadable file;")
    assert unreadable_block(()) == ""


def test_every_page_lists_the_unreadable_entries() -> None:
    context = PageContext(demo=False, unreadable=("account: ValueError: x",))

    assert "- account (ValueError)" in render_page("Cycles", context)


def test_escaped_text_never_starts_a_line() -> None:
    hostile = f"{TILDES}\n{TILDES}: {TILDES}: x"
    context = PageContext(demo=False, unreadable=(hostile, f"{TILDES}: ValueError: x"))
    page = render_page("T", context, table((hostile,), [(hostile,)]))

    assert all(not line.startswith("~") for line in page.splitlines())
    assert f"- {unreadable_entry(TILDES)}" in page
    assert f"| {cell(hostile)} |" in page


def test_rounded_rounds_finite_floats_in_views_and_tuples_only() -> None:
    assert rounded((1 / 3, math.inf, 2, DAY0, "x")) == (0.333333, math.inf, 2, DAY0, "x")
    view = rounded(portfolio_view(make_snapshots(*NAV_VALUES), None, ()))
    assert view.drawdown[3] == (DAY0.replace(day=5), -0.041667)
    assert math.isnan(rounded(math.nan))


def test_chart_json_is_compact_rounded_and_has_no_raw_lt_or_amp() -> None:
    text = chart_json(_figure("a</script>&b"))

    assert "<" not in text
    assert "&" not in text
    assert "\\u003c/script>" in text
    assert "\\u0026b" in text
    parsed = json.loads(text)
    assert parsed["data"][0]["name"] == "a</script>&b"
    assert numbers(parsed["data"][0]["y"]) == [0.333333, None, 2.0]
    assert parsed["layout"]["autosize"] is True
    assert "\n" not in text


def test_a_chart_block_is_one_div_and_a_light_and_a_dark_script() -> None:
    block = chart_block("portfolio-x", lambda palette: drawdown_chart(_view(), palette))
    lines = block.split("\n")

    assert lines[0] == f'<div id="portfolio-x" class="{CHART_CLASS}"></div>'
    assert len(lines) == 3
    assert "" not in lines
    assert chart_divs(block) == [("portfolio-x", CHART_CLASS)]
    shown = figures(block)
    assert list(shown) == ["portfolio-x-light", "portfolio-x-dark"]
    assert shown["portfolio-x-light"]["layout"]["paper_bgcolor"] == LIGHT_PALETTE.background
    assert shown["portfolio-x-dark"]["layout"]["paper_bgcolor"] == DARK_PALETTE.background
    assert all("<" not in text and "&" not in text for text in scripts(block).values())


def test_chart_or_draws_only_when_there_is_data() -> None:
    assert chart_or("x", False, lambda palette: _figure(), "Nothing.") == "Nothing."
    assert chart_or("x", True, lambda palette: _figure(), "Nothing.").startswith('<div id="x"')


def test_the_site_no_spy_text_comes_back_exactly_from_both_notes() -> None:
    assert SITE_NO_SPY == "No SPY series for this root."
    assert chart_note(None, partial=False, no_spy=SITE_NO_SPY) == SITE_NO_SPY
    assert return_note(None, partial=False, has_return=False, no_spy=SITE_NO_SPY) == SITE_NO_SPY


def test_a_chart_block_is_byte_identical_to_the_golden_file() -> None:
    assert plotly.__version__ == "7.1.0"  # the golden pins this plotly's JSON (incl. bdata)
    assert golden_block() + "\n" == GOLDEN.read_text(encoding="utf-8")
    light = figures(golden_block())["portfolio-drawdown-light"]
    assert numbers(light["data"][0]["y"]) == [0.0, 0.0, -0.25, -0.041667]


def _view() -> PortfolioView:
    return portfolio_view(make_snapshots(*NAV_VALUES), None, ())
