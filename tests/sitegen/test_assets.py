"""palette.css frames charts in the palette colours; charts.js draws and re-draws them."""

from __future__ import annotations

import json
import re

from hedge_fund.sitegen.assets import (
    CHART_CLASS,
    CHART_MIN_HEIGHT,
    DARK_SUFFIX,
    LIGHT_SUFFIX,
    charts_js,
    palette_css,
)
from hedge_fund.sitegen.palette import DARK_PALETTE, LIGHT_PALETTE

HEX = re.compile(r"#[0-9A-Fa-f]{6}\b")


def _classes(css: str) -> set[str]:
    selectors = re.findall(r"([^{}]+)\{", css)
    return {name for selector in selectors for name in re.findall(r"\.([A-Za-z][\w-]*)", selector)}


def test_the_css_colours_come_straight_from_the_palettes() -> None:
    assert set(HEX.findall(palette_css())) == {
        LIGHT_PALETTE.background,
        LIGHT_PALETTE.grid,
        DARK_PALETTE.background,
        DARK_PALETTE.grid,
    }


def test_the_css_styles_only_the_chart_class_and_gives_it_a_min_height() -> None:
    css = palette_css()

    assert _classes(css) == {CHART_CLASS}
    assert f"min-height: {CHART_MIN_HEIGHT}px" in css
    assert CHART_MIN_HEIGHT == 420


def test_the_dark_frame_applies_under_material_slate() -> None:
    assert (
        f'[data-md-color-scheme="slate"] .{CHART_CLASS} {{ '
        f"background: {DARK_PALETTE.background}; border-color: {DARK_PALETTE.grid}; }}"
    ) in palette_css()


def test_charts_js_reads_the_scheme_and_draws_with_newplot_then_react() -> None:
    js = charts_js()

    assert 'document.body.getAttribute("data-md-color-scheme") === "slate"' in js
    assert 'document.addEventListener("DOMContentLoaded"' in js
    assert "Plotly.newPlot(div, figure.data, figure.layout, CONFIG)" in js
    assert "Plotly.react(div, figure.data, figure.layout, CONFIG)" in js
    assert "var CONFIG = {responsive: true, displaylogo: false};" in js
    assert "JSON.parse(document.getElementById(div.id + suffix).textContent)" in js


def test_charts_js_observes_only_the_scheme_attribute() -> None:
    js = charts_js()

    assert js.count(".observe(") == 1
    assert "observe(document.body, {attributes: true, attributeFilter: [" in js
    assert re.findall(r"attributeFilter: \[(.*?)\]", js) == ['"data-md-color-scheme"']


def test_charts_js_uses_the_shared_class_and_suffix_constants() -> None:
    js = charts_js()

    assert f"var SELECTOR = {json.dumps('.' + CHART_CLASS)};" in js
    assert "document.querySelectorAll(SELECTOR)" in js
    assert f"light: {json.dumps(LIGHT_SUFFIX)}" in js
    assert f"dark: {json.dumps(DARK_SUFFIX)}" in js
    assert "__" not in js  # every template token was filled


def test_charts_js_has_no_resize_or_hidden_div_workaround() -> None:
    assert "resize" not in charts_js()
