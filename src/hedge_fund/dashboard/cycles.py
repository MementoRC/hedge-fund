"""Cycles page: every cycle with its outcome, and the selected cycle's funnel, risk vetoes,
approval and LLM budget vs spend.

The funnel starts at the shortlist: a cycle report stores no universe or gate counts.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from hedge_fund.dashboard.data import RootState
from hedge_fund.dashboard.format import NOT_AVAILABLE, escape_markdown, fmt_date, fmt_money
from hedge_fund.dashboard.palette import PALETTE
from hedge_fund.dashboard.picker import pick_index
from hedge_fund.dashboard.states import nothing_message, readable_workspace
from hedge_fund.viz.charts import funnel_chart
from hedge_fund.viz.views import CycleRow, CycleView, cycle_view

CYCLE_COLUMNS = [
    "Date",
    "Outcome",
    "Dry run",
    "Shortlist",
    "Researched",
    "Target positions",
    "Orders",
    "Risk vetoes",
    "Approval",
    "LLM budget",
    "LLM spent",
]


def _approval(row: CycleRow) -> str:
    return NOT_AVAILABLE if row.approval is None else row.approval.value


def cycle_table(view: CycleView) -> pd.DataFrame:
    """One formatted row per cycle, in load order (oldest first)."""
    # Pre-formatted strings (NaN reads "n/a"), so sorting the money columns is lexical.
    return pd.DataFrame(
        [
            {
                "Date": fmt_date(row.as_of),
                "Outcome": row.outcome.value,
                "Dry run": "yes" if row.dry_run else "no",
                "Shortlist": row.funnel.shortlist,
                "Researched": row.funnel.researched,
                "Target positions": row.funnel.portfolio,
                "Orders": row.funnel.orders,
                "Risk vetoes": len(row.vetoes),
                "Approval": _approval(row),
                "LLM budget": fmt_money(row.budget),
                "LLM spent": fmt_money(row.spent),
            }
            for row in view.rows
        ],
        columns=CYCLE_COLUMNS,
    )


def cycle_label(row: CycleRow) -> str:
    return f"{fmt_date(row.as_of)} ({row.outcome.value})"


FUNNEL_NOTE = (
    "The funnel starts at the shortlist: a cycle report stores no universe or quality-gate "
    "counts. Target positions are the target at rebalance; orders are the planned orders."
)


def render(state: RootState) -> None:
    st.title("Cycles")
    workspace = readable_workspace(state)
    if workspace is None:
        return
    view = cycle_view(workspace.cycles)
    if not view.rows:
        st.info(nothing_message(state.root, "cycle reports", "hedge-fund cycle"))
        return
    st.dataframe(cycle_table(view), hide_index=True)
    _detail(_selected(view, state.root))


def _selected(view: CycleView, root: Path) -> CycleRow:
    """The cycle picked in the selector; the newest is listed (and selected) first.

    The key is scoped to the root, so a row index saved for another root is never read; an
    index out of range (the root's own files changed) falls back to the newest cycle.
    """
    newest = len(view.rows) - 1
    index = pick_index(
        "Cycle",
        list(reversed(range(len(view.rows)))),
        lambda i: cycle_label(view.rows[i]),
        key=f"cycle:{root}",
        fallback=newest,
    )
    return view.rows[index]


def _detail(row: CycleRow) -> None:
    st.subheader(f"Cycle of {fmt_date(row.as_of)}")
    st.plotly_chart(funnel_chart(row.funnel, PALETTE), theme=None)
    st.caption(FUNNEL_NOTE)
    outcome, approval, budget, spent = st.columns(4)
    outcome.metric("Outcome", row.outcome.value)
    approval.metric("Approval", _approval(row))
    budget.metric("LLM budget", fmt_money(row.budget))
    spent.metric("LLM spent", fmt_money(row.spent))
    st.markdown("**Risk vetoes**")
    if row.vetoes:
        st.markdown("\n".join(f"- {escape_markdown(veto)}" for veto in row.vetoes))
    else:
        st.caption("No risk veto.")
    if row.halted:
        st.markdown("**Halted** by a risk breaker.")
    if row.unresolved:
        st.caption(f"Unresolved tickers (no price or fundamentals): {row.unresolved}.")
    if row.error is not None:
        st.markdown(f"**Cycle error:** {escape_markdown(row.error)}")
