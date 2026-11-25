"""One palette and one plotly template, in a dark and a light variant. Pure.

The series colours follow the Okabe-Ito idea (hues that stay apart for colour-vision
deficiency), darkened in the light variant so every chart mark keeps 3:1 against white.
``tests/viz/test_theme.py`` pins text contrast (>= 4.5:1), mark contrast (>= 3:1) and the
colour distance under deuteranopia. Charts that need to tell series apart also vary the line
style, so colour is never the only cue.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import plotly.graph_objects as go


@dataclass(frozen=True)
class Palette:
    """Page colours and four categorical series colours, all ``#RRGGBB``."""

    name: str
    background: str
    text: str
    muted: str
    grid: str
    strategy: str
    secondary: str
    benchmark: str
    emphasis: str

    @property
    def series(self) -> tuple[str, ...]:
        """The categorical colours in plot order."""
        return (self.strategy, self.secondary, self.benchmark, self.emphasis)


LIGHT = Palette(
    name="light",
    background="#FFFFFF",
    text="#1A1A1A",
    muted="#595959",
    grid="#E0E0E0",
    strategy="#0072B2",
    secondary="#B35A00",
    benchmark="#6B6B6B",
    emphasis="#222222",
)
DARK = Palette(
    name="dark",
    background="#0E1117",
    text="#FAFAFA",
    muted="#A3A8B8",
    grid="#2B3040",
    strategy="#56B4E9",
    secondary="#E69F00",
    benchmark="#9AA0A6",
    emphasis="#F2F2F2",
)
PALETTES: Mapping[str, Palette] = {"light": LIGHT, "dark": DARK}


def plotly_template(palette: Palette) -> go.layout.Template:
    """A plotly template that paints a figure with ``palette``."""
    axis = {
        "gridcolor": palette.grid,
        "zerolinecolor": palette.grid,
        "linecolor": palette.muted,
        "tickfont": {"color": palette.muted},
        "title": {"font": {"color": palette.muted}},
    }
    return go.layout.Template(
        layout=go.Layout(
            paper_bgcolor=palette.background,
            plot_bgcolor=palette.background,
            font={"color": palette.text},
            colorway=list(palette.series),
            xaxis=axis,
            yaxis=axis,
            legend={"font": {"color": palette.text}},
            hoverlabel={"bgcolor": palette.background, "font": {"color": palette.text}},
        )
    )
