"""The site draws every chart in both theme palettes."""

from __future__ import annotations

from hedge_fund.sitegen.palette import DARK_PALETTE, LIGHT_PALETTE
from hedge_fund.viz.theme import DARK, LIGHT


def test_the_site_palettes_are_the_theme_light_and_dark() -> None:
    assert LIGHT_PALETTE is LIGHT
    assert DARK_PALETTE is DARK
