"""The dashboard's one chart palette. Pure: no Streamlit, no I/O.

One fixed palette (user decision in D1: no theme selector). Pages draw charts with
``st.plotly_chart(figure, theme=None)``, so this palette's plotly template paints them, not
Streamlit's theme. This is the only dashboard module that imports ``viz.theme``.
"""

from __future__ import annotations

from hedge_fund.viz.theme import DARK, Palette

PALETTE: Palette = DARK
