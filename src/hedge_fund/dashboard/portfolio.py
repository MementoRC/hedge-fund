"""Portfolio & NAV page: NAV vs SPY, drawdown, target exposure, sectors and positions.

Exposure here is each rebalance's target weights, not the book between rebalances.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from hedge_fund.dashboard.data import RootState
from hedge_fund.dashboard.format import chart_note, fmt_date, fmt_percent
from hedge_fund.dashboard.states import DEMO_COMMAND, usable_workspace
from hedge_fund.viz.charts import (
    drawdown_chart,
    exposure_chart,
    nav_vs_benchmark,
    sector_exposure_chart,
)
from hedge_fund.viz.theme import DARK
from hedge_fund.viz.views import PortfolioView, portfolio_view

PALETTE = DARK  # one fixed palette (user decision: no theme selector)
POSITION_COLUMNS = ["Ticker", "Sector", "Side", "Target weight"]
EXPOSURE_NOTE = (
    "Exposure is the target weights at each rebalance (gross = long plus short, "
    "net = long minus short), not the holdings between rebalances."
)
NO_TARGET_MESSAGE = (
    "No cycle in this root has a target portfolio, so there is no exposure to show. "
    f"A cycle or backtest produces one; `{DEMO_COMMAND}` builds a synthetic world."
)


def positions_table(view: PortfolioView) -> pd.DataFrame:
    """The latest target book, one formatted row per position."""
    return pd.DataFrame(
        [
            {
                "Ticker": position.ticker,
                "Sector": position.sector,
                "Side": position.side.value,
                "Target weight": fmt_percent(position.weight),
            }
            for position in view.positions
        ],
        columns=POSITION_COLUMNS,
    )


def _chart(figure: go.Figure) -> None:
    st.plotly_chart(figure, theme=None)  # the plotly template, not Streamlit's, paints it


def render(state: RootState) -> None:
    st.title("Portfolio & NAV")
    workspace = usable_workspace(state)
    if workspace is None:
        return
    view = portfolio_view(workspace.snapshots, workspace.benchmark, workspace.cycles)
    _chart(nav_vs_benchmark(view, PALETTE))
    st.caption(chart_note(view.benchmark_window, view.benchmark_partial))
    _chart(drawdown_chart(view, PALETTE))
    st.subheader("Exposure")
    if not view.exposure:
        st.info(NO_TARGET_MESSAGE)
        return
    st.caption(EXPOSURE_NOTE)
    _chart(exposure_chart(view, PALETTE))
    if view.sectors:
        _chart(sector_exposure_chart(view, PALETTE))
    if view.positions:
        st.subheader("Positions")
        st.caption(f"Target weights of the cycle of {fmt_date(view.target_as_of)}.")
        st.dataframe(positions_table(view), hide_index=True)
