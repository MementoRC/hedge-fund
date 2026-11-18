"""The site's chart assets as strings: ``palette.css`` and ``charts.js``. Pure: no I/O.

Discovery convention, shared with ``pages.py``: every chart is ``<div id="<id>"
class="hf-chart">`` and its figure JSON sits in the scripts ``<id>-light`` and
``<id>-dark``. ``charts.js`` draws each chart in the palette matching Material's scheme
(``slate`` is dark, anything else light) and re-draws when the scheme attribute changes.
A chart whose JSON is missing or invalid is logged and skipped, so it cannot blank the others.

Rendering relies on ``DOMContentLoaded``, which holds because ``website/mkdocs.yml`` does not
enable ``navigation.instant``. If that is ever enabled, render from Material's ``document$``.
"""

from __future__ import annotations

import json

from hedge_fund.sitegen.palette import DARK_PALETTE, LIGHT_PALETTE

CHART_CLASS = "hf-chart"
LIGHT_SUFFIX = "-light"
DARK_SUFFIX = "-dark"
CHART_MIN_HEIGHT = 420  # px, so a chart's space is kept before plotly draws it
ASSETS_DIR = "assets"
PALETTE_CSS = "palette.css"
CHARTS_JS = "charts.js"
PLOTLY_JS = "plotly.min.js"

_CHARTS_JS = """\
(function () {
  "use strict";
  var SELECTOR = __SELECTOR__;
  var SUFFIX = {light: __LIGHT__, dark: __DARK__};
  var CONFIG = {responsive: true, displaylogo: false};

  function scheme() {
    return document.body.getAttribute("data-md-color-scheme") === "slate" ? "dark" : "light";
  }

  function each(plot) {
    var suffix = SUFFIX[scheme()];
    document.querySelectorAll(SELECTOR).forEach(function (div) {
      try {
        plot(div, JSON.parse(document.getElementById(div.id + suffix).textContent));
      } catch (error) {
        console.error("chart " + div.id + " not drawn:", error);
      }
    });
  }

  // DOMContentLoaded is enough while website/mkdocs.yml leaves navigation.instant off.
  // With instant navigation, render from Material's document$ observable instead.
  document.addEventListener("DOMContentLoaded", function () {
    each(function (div, figure) {
      Plotly.newPlot(div, figure.data, figure.layout, CONFIG);
    });
    new MutationObserver(function () {
      each(function (div, figure) {
        Plotly.react(div, figure.data, figure.layout, CONFIG);
      });
    }).observe(document.body, {attributes: true, attributeFilter: ["data-md-color-scheme"]});
  });
})();
"""


def palette_css() -> str:
    """Chart panel framing per scheme; the only selector it styles is ``.hf-chart``."""
    light, dark = LIGHT_PALETTE, DARK_PALETTE
    return (
        f".{CHART_CLASS} {{ min-height: {CHART_MIN_HEIGHT}px; background: {light.background}; "
        f"border: 1px solid {light.grid}; }}\n"
        f'[data-md-color-scheme="slate"] .{CHART_CLASS} {{ background: {dark.background}; '
        f"border-color: {dark.grid}; }}\n"
    )


def charts_js() -> str:
    """The script that draws every ``.hf-chart`` from its light or dark JSON."""
    return (
        _CHARTS_JS.replace("__SELECTOR__", json.dumps(f".{CHART_CLASS}"))
        .replace("__LIGHT__", json.dumps(LIGHT_SUFFIX))
        .replace("__DARK__", json.dumps(DARK_SUFFIX))
    )
