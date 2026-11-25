"""Research & attribution page: the latest research per name, thesis verification per
horizon, attribution by name and sector, and LLM cost across cycles.

On a demo or backtest root (``RootState.kind`` from ``classify_root``) attribution and
verification cannot exist, and the page says why.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pandas as pd
import streamlit as st

from hedge_fund.dashboard.data import RootState
from hedge_fund.dashboard.palette import PALETTE
from hedge_fund.dashboard.states import DEMO_COMMAND, nothing_message, readable_workspace
from hedge_fund.domain.attribution import AttributionTotals
from hedge_fund.viz.charts import llm_cost_chart
from hedge_fund.viz.format import escape_markdown, fmt_date, fmt_money, fmt_number, fmt_percent
from hedge_fund.viz.load import RootKind
from hedge_fund.viz.views import ResearchView, research_view

NAME_COLUMNS = [
    "Ticker",
    "Sector",
    "Side",
    "Verdict",
    "Mean rating",
    "Dispersion",
    "Agreement",
    "Valid runs",
]
HORIZON_COLUMNS = ["Horizon", "Scored", "Hit rate", "IC (rank correlation)"]
ATTRIBUTION_COLUMNS = ["Market", "Sector", "Residual", "Trading", "Total"]
COST_COLUMNS = ["Cycle", "LLM budget", "LLM spent"]
NO_THESIS = "No valid research run, so there is no thesis text."
VERIFY_NOTE = (
    "From the latest completed verification. IC is the rank correlation of the mean rating "
    "with the aligned excess return; hit rate and IC read n/a when they cannot be computed."
)
ATTRIBUTION_NOTE = (
    "From the latest completed attribution: P&L in dollars split into market, sector, "
    "residual and trading parts (total = their sum)."
)
NO_COSTS = (
    "No cycle reports, so there is no LLM cost to show. "
    f"`{DEMO_COMMAND}` builds a synthetic demo world that has them."
)


def names_table(view: ResearchView) -> pd.DataFrame:
    # Pre-formatted strings (NaN reads "n/a"), so sorting the number columns is lexical.
    return pd.DataFrame(
        [
            {
                "Ticker": row.ticker,
                "Sector": row.sector,
                "Side": row.side.value,
                "Verdict": row.verdict.value,
                "Mean rating": fmt_number(row.mean),
                "Dispersion": fmt_number(row.dispersion),
                "Agreement": fmt_number(row.agreement),
                "Valid runs": row.valid,
            }
            for row in view.names
        ],
        columns=NAME_COLUMNS,
    )


def horizons_table(view: ResearchView) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Horizon": row.horizon,
                "Scored": row.scored,
                "Hit rate": fmt_percent(row.hit_rate),
                "IC (rank correlation)": fmt_number(row.ic),
            }
            for row in view.horizons
        ],
        columns=HORIZON_COLUMNS,
    )


def attribution_table(totals: Mapping[str, AttributionTotals], label: str) -> pd.DataFrame:
    """One row per key of ``totals`` (``label`` names the key column and must not be one of
    ``ATTRIBUTION_COLUMNS``), P&L in dollars.
    """
    return pd.DataFrame(
        [
            {
                label: key,
                "Market": fmt_money(part.market),
                "Sector": fmt_money(part.sector),
                "Residual": fmt_money(part.residual),
                "Trading": fmt_money(part.trading),
                "Total": fmt_money(part.total),
            }
            for key, part in totals.items()
        ],
        columns=[label, *ATTRIBUTION_COLUMNS],
    )


def cost_table(view: ResearchView) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Cycle": fmt_date(row.as_of),
                "LLM budget": fmt_money(row.budget),
                "LLM spent": fmt_money(row.spent),
            }
            for row in view.costs
        ],
        columns=COST_COLUMNS,
    )


def gap_message(kind: RootKind | None, what: str, command: str) -> str:
    """Why ``what`` (verification or attribution) is absent, by ``classify_root``'s kind."""
    if kind is None:
        return f"No completed {what} in this root yet; `{command}` writes one."
    return (
        f"No {what}: this is a {kind.value} root. A backtest keeps its fills in memory, so "
        "`hedge-fund settle`, `hedge-fund attribute` and `hedge-fund verify` cannot run on it."
    )


def render(state: RootState) -> None:
    st.title("Research & attribution")
    workspace = readable_workspace(state)
    if workspace is None:
        return
    view = research_view(
        workspace.research, workspace.cycles, workspace.verifications, workspace.attributions
    )
    _names(view, state.root)
    _verification(view, state.kind)
    _attribution(view, state.kind)
    _costs(view)


def _names(view: ResearchView, root: Path) -> None:
    st.subheader("Latest research")
    if not view.names:
        st.info(nothing_message(root, "research", "hedge-fund cycle"))
        return
    st.caption(f"Research of {fmt_date(view.names[0].as_of)}.")
    st.dataframe(names_table(view), hide_index=True)
    for row in view.names:
        with st.expander(f"Thesis: {escape_markdown(row.ticker)}"):  # stored text, literal
            # LLM output: escaped so markdown and "$" LaTeX render literally (Decision 10).
            st.markdown(escape_markdown(row.thesis) if row.thesis else NO_THESIS)


def _verification(view: ResearchView, kind: RootKind | None) -> None:
    st.subheader("Thesis verification")
    if not view.horizons:
        st.info(gap_message(kind, "thesis verification", "hedge-fund verify"))
        return
    st.dataframe(horizons_table(view), hide_index=True)
    st.caption(VERIFY_NOTE)


def _attribution(view: ResearchView, kind: RootKind | None) -> None:
    st.subheader("Attribution")
    if view.attribution is None:
        st.info(gap_message(kind, "attribution", "hedge-fund attribute"))
        return
    st.dataframe(attribution_table(view.attribution.by_name, "Ticker"), hide_index=True)
    st.dataframe(attribution_table(view.attribution.by_sector, "Sector name"), hide_index=True)
    st.dataframe(attribution_table({"All names": view.attribution.total}, "Book"), hide_index=True)
    st.caption(ATTRIBUTION_NOTE)


def _costs(view: ResearchView) -> None:
    st.subheader("LLM cost")
    if not view.costs:
        st.caption(NO_COSTS)
        return
    st.plotly_chart(llm_cost_chart(view.costs, PALETTE), theme=None)
    st.dataframe(cost_table(view), hide_index=True)
