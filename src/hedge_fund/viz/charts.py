"""Plotly figures from views, for the dashboard and the site. Pure: no Streamlit.

Only the charts the first dashboard PR needs live here (NAV vs SPY, drawdown, exposure,
sector exposure). A new chart lands in the PR that first needs it; the other track consumes
it after that merges. Each builder returns a figure whose traces are empty for an empty
view: showing an empty state is the page's job.
"""

from __future__ import annotations

import plotly.graph_objects as go

from hedge_fund.viz.theme import Palette, plotly_template
from hedge_fund.viz.views import Curve, PortfolioView

PERCENT_TICKS = ".0%"
PERCENT_HOVER = ".1%"


def nav_vs_benchmark(view: PortfolioView, palette: Palette) -> go.Figure:
    """The strategy and SPY, both indexed to 100 at the first day; SPY is dashed."""
    figure = _figure("NAV vs SPY (indexed to 100)", palette, "Index")
    figure.add_trace(_line(view.nav_index, "Strategy", palette.strategy))
    if view.benchmark_index is not None:
        figure.add_trace(_line(view.benchmark_index, "SPY", palette.benchmark, dash="dash"))
    return figure


def drawdown_chart(view: PortfolioView, palette: Palette) -> go.Figure:
    """Drawdown from the running peak, as a filled area below zero."""
    figure = _figure("Drawdown", palette, "Drawdown")
    trace = _line(view.drawdown, "Drawdown", palette.secondary)
    trace.update(fill="tozeroy")
    figure.add_trace(trace)
    figure.update_yaxes(tickformat=PERCENT_TICKS, hoverformat=PERCENT_HOVER)
    return figure


def exposure_chart(view: PortfolioView, palette: Palette) -> go.Figure:
    """Gross and net exposure of each cycle's target (net is dashed)."""
    figure = _figure("Target exposure at each rebalance", palette, "Share of capital")
    points = view.exposure
    figure.add_trace(
        _line(tuple((p.as_of, p.gross) for p in points), "Gross", palette.strategy, markers=True)
    )
    figure.add_trace(
        _line(
            tuple((p.as_of, p.net) for p in points),
            "Net",
            palette.secondary,
            dash="dash",
            markers=True,
        )
    )
    figure.update_yaxes(tickformat=PERCENT_TICKS, hoverformat=PERCENT_HOVER)
    return figure


def sector_exposure_chart(view: PortfolioView, palette: Palette) -> go.Figure:
    """Long (up) and short (down) weight by sector for the latest book."""
    figure = _figure("Sector exposure (latest book)", palette, "Share of capital")
    sectors = [row.sector for row in view.sectors]
    figure.add_trace(
        go.Bar(
            x=sectors,
            y=[row.long for row in view.sectors],
            name="Long",
            marker_color=palette.strategy,
        )
    )
    figure.add_trace(
        go.Bar(
            x=sectors,
            y=[row.short for row in view.sectors],
            name="Short",
            marker_color=palette.secondary,
        )
    )
    figure.update_layout(barmode="relative")
    figure.update_yaxes(tickformat=PERCENT_TICKS, hoverformat=PERCENT_HOVER)
    return figure


def _figure(title: str, palette: Palette, y_title: str) -> go.Figure:
    figure = go.Figure()
    figure.update_layout(
        template=plotly_template(palette),
        title={"text": title},
        yaxis={"title": {"text": y_title}},
        margin={"l": 60, "r": 20, "t": 50, "b": 40},
    )
    return figure


def _line(
    curve: Curve, name: str, colour: str, *, dash: str = "solid", markers: bool = False
) -> go.Scatter:
    return go.Scatter(
        x=[day for day, _ in curve],
        y=[value for _, value in curve],
        name=name,
        mode="lines+markers" if markers else "lines",
        line={"color": colour, "dash": dash},
    )
