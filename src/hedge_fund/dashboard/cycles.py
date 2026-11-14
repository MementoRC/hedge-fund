"""Cycles page: every cycle with its outcome, and the selected cycle's funnel, risk vetoes,
approval and LLM budget vs spend.

The funnel starts at the shortlist: a cycle report stores no universe or gate counts.
"""

from __future__ import annotations

import pandas as pd

from hedge_fund.dashboard.format import NOT_AVAILABLE, fmt_date, fmt_money
from hedge_fund.viz.views import CycleRow, CycleView

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
