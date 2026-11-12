"""The palette: WCAG contrast for text and chart marks, and series that stay apart under
deuteranopia. The helpers here are the specification of those two measures.
"""

from __future__ import annotations

import itertools
import math

import pytest

from hedge_fund.viz.theme import DARK, LIGHT, PALETTES, Palette, plotly_template

TEXT_CONTRAST = 4.5  # WCAG AA, normal text
MARK_CONTRAST = 3.0  # WCAG AA, graphical objects
MIN_DISTANCE = 25.0  # CIE76 delta E between any two series colours, as a deuteranope sees them

# Machado et al. 2009, deuteranopia (severity 1.0), applied to linear RGB.
DEUTERANOPIA = (
    (0.367322, 0.860646, -0.227968),
    (0.280085, 0.672501, 0.047413),
    (-0.011820, 0.042940, 0.968881),
)
D65 = (0.95047, 1.0, 1.08883)


def _linear(hex_colour: str) -> tuple[float, float, float]:
    def channel(index: int) -> float:
        value = int(hex_colour[1 + 2 * index : 3 + 2 * index], 16) / 255
        return value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4

    return channel(0), channel(1), channel(2)


def _luminance(linear: tuple[float, float, float]) -> float:
    red, green, blue = linear
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def contrast_ratio(first: str, second: str) -> float:
    high, low = sorted((_luminance(_linear(first)), _luminance(_linear(second))), reverse=True)
    return (high + 0.05) / (low + 0.05)


def _deuteranopia(hex_colour: str) -> tuple[float, float, float]:
    linear = _linear(hex_colour)
    red, green, blue = (
        min(1.0, max(0.0, sum(m * c for m, c in zip(row, linear, strict=True))))
        for row in DEUTERANOPIA
    )
    return red, green, blue


def _lab(linear: tuple[float, float, float]) -> tuple[float, float, float]:
    red, green, blue = linear
    xyz = (
        0.4124564 * red + 0.3575761 * green + 0.1804375 * blue,
        0.2126729 * red + 0.7151522 * green + 0.0721750 * blue,
        0.0193339 * red + 0.1191920 * green + 0.9503041 * blue,
    )

    def f(t: float) -> float:
        return t ** (1 / 3) if t > 216 / 24389 else (24389 / 27 * t + 16) / 116

    fx, fy, fz = (f(value / white) for value, white in zip(xyz, D65, strict=True))
    return 116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)


def distance(first: str, second: str, *, simulate: bool) -> float:
    colours = (
        (_deuteranopia(first), _deuteranopia(second))
        if simulate
        else (
            _linear(first),
            _linear(second),
        )
    )
    return math.dist(_lab(colours[0]), _lab(colours[1]))


def test_the_contrast_helper_matches_the_wcag_extremes() -> None:
    assert contrast_ratio("#000000", "#FFFFFF") == pytest.approx(21.0)
    assert contrast_ratio("#777777", "#777777") == pytest.approx(1.0)


def test_the_distance_helper_spans_lightness_and_is_zero_for_equal_colours() -> None:
    assert distance("#000000", "#FFFFFF", simulate=False) == pytest.approx(100.0, abs=0.1)
    assert distance("#56B4E9", "#56B4E9", simulate=True) == pytest.approx(0.0)


@pytest.mark.parametrize("palette", [LIGHT, DARK], ids=lambda palette: palette.name)
def test_text_meets_the_text_contrast_on_the_background(palette: Palette) -> None:
    assert contrast_ratio(palette.text, palette.background) >= TEXT_CONTRAST
    assert contrast_ratio(palette.muted, palette.background) >= TEXT_CONTRAST


@pytest.mark.parametrize("palette", [LIGHT, DARK], ids=lambda palette: palette.name)
def test_every_series_colour_meets_the_mark_contrast_on_the_background(palette: Palette) -> None:
    for colour in palette.series:
        assert contrast_ratio(colour, palette.background) >= MARK_CONTRAST, colour


@pytest.mark.parametrize("palette", [LIGHT, DARK], ids=lambda palette: palette.name)
def test_series_colours_stay_apart_under_deuteranopia(palette: Palette) -> None:
    for first, second in itertools.combinations(palette.series, 2):
        assert distance(first, second, simulate=True) >= MIN_DISTANCE, (first, second)


def test_the_series_colours_are_four_distinct_names() -> None:
    for palette in (LIGHT, DARK):
        assert len(set(palette.series)) == len(palette.series) == 4
    assert PALETTES == {"light": LIGHT, "dark": DARK}


@pytest.mark.parametrize("palette", [LIGHT, DARK], ids=lambda palette: palette.name)
def test_the_plotly_template_carries_the_palette(palette: Palette) -> None:
    layout = plotly_template(palette).layout

    assert layout.paper_bgcolor == palette.background
    assert layout.plot_bgcolor == palette.background
    assert layout.font.color == palette.text
    assert tuple(layout.colorway) == palette.series
    assert layout.xaxis.gridcolor == palette.grid
