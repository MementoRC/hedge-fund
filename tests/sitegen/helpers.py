"""Pull chart ids and figure JSON out of a generated page."""

from __future__ import annotations

import base64
import json
import re
from typing import Any

import numpy as np

from hedge_fund.sitegen.pages import PageContext

SCRIPT = re.compile(r'<script type="application/json" id="([^"]+)">(.*?)</script>')
DIV = re.compile(r'<div id="([^"]+)" class="([^"]+)"></div>')
PLAIN = PageContext(demo=False, unreadable=())
DEMO = PageContext(demo=True, unreadable=())


def scripts(page: str) -> dict[str, str]:
    """Script id to its raw JSON text."""
    return dict(SCRIPT.findall(page))


def figures(page: str) -> dict[str, Any]:
    return {key: json.loads(text) for key, text in scripts(page).items()}


def chart_divs(page: str) -> list[tuple[str, str]]:
    """``(id, class)`` of every chart div, in page order."""
    return DIV.findall(page)


def chart_ids(page: str) -> list[str]:
    return [chart_id for chart_id, _ in chart_divs(page)]


def numbers(array: Any) -> list[float]:
    """A plotly data array as numbers, whether a plain list or a typed ``bdata`` array."""
    if isinstance(array, dict):
        raw = base64.b64decode(array["bdata"])
        return [float(x) for x in np.frombuffer(raw, dtype=np.dtype(array["dtype"]))]
    return list(array)
