"""The site's two chart palettes. Pure: no I/O.

Every chart is built in both, and ``charts.js`` draws the one matching Material's scheme
(``slate`` is dark). This is the only sitegen module that imports ``viz.theme``; the others
take ``Palette`` from here.
"""

from __future__ import annotations

from hedge_fund.viz.theme import DARK, LIGHT, Palette

__all__ = ["DARK_PALETTE", "LIGHT_PALETTE", "Palette"]

LIGHT_PALETTE: Palette = LIGHT
DARK_PALETTE: Palette = DARK
