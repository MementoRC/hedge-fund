"""Overview page: the headline figures of a root and its backtest runs.

Every SPY figure carries the window it covers: the strategy return is labelled "full
history", the SPY pair "SPY window", and the run table's returns "full window".
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd
import streamlit as st

from hedge_fund.dashboard.data import RootState
from hedge_fund.dashboard.format import (
    NOT_AVAILABLE,
    fmt_date,
    fmt_money,
    fmt_number,
    fmt_percent,
    fmt_window,
    return_note,
)
from hedge_fund.dashboard.states import usable_workspace
from hedge_fund.viz.views import BacktestRun, OverviewView, backtest_view, overview_view

COLUMNS_PER_ROW = 3
RUN_NOTE = (
    "Run figures (return, SPY, excess) are over each run's whole window, which can be longer "
    "than the SPY span of the NAV chart."
)


@dataclass(frozen=True)
class Headline:
    label: str
    value: str
    hint: str | None = None


def headlines(view: OverviewView) -> tuple[Headline, ...]:
    outcome = NOT_AVAILABLE if view.last_cycle_outcome is None else view.last_cycle_outcome.value
    spy_window = f"SPY window: {fmt_window(view.benchmark_window)}"
    return (
        Headline("NAV", fmt_money(view.nav), f"As of {fmt_date(view.as_of)}"),
        Headline("Strategy return (full history)", fmt_percent(view.total_return)),
        Headline("Max drawdown", fmt_percent(view.max_drawdown), "Full history"),
        Headline("SPY return (SPY window)", fmt_percent(view.benchmark_return), spy_window),
        Headline(
            "Strategy return (SPY window)",
            fmt_percent(view.strategy_return_in_benchmark_window),
            spy_window,
        ),
        Headline("Last cycle", outcome, f"Cycle of {fmt_date(view.last_cycle_as_of)}"),
    )


def run_table(runs: Sequence[BacktestRun]) -> pd.DataFrame:
    """One formatted row per run; the three return columns are full-window figures."""
    # Pre-formatted strings: raw floats would show NaN as blank, not "n/a", so sorting is lexical.
    return pd.DataFrame(
        [
            {
                "Run": run.run_id,
                "Window": f"{fmt_date(run.start)} to {fmt_date(run.end)}",
                "Trading days": run.trading_days,
                "Rebalances": run.rebalances,
                "Return (full window)": fmt_percent(run.total_return),
                "SPY (full window)": fmt_percent(run.benchmark_return),
                "Excess (full window)": fmt_percent(run.excess_return),
                "Max drawdown": fmt_percent(run.max_drawdown),
                "Sharpe": fmt_number(run.sharpe),
                "Beta": fmt_number(run.beta),
                "Breaker days": run.breaker_days,
            }
            for run in runs
        ]
    )


def render(state: RootState) -> None:
    st.title("Overview")
    workspace = usable_workspace(state)
    if workspace is None:
        return
    view = overview_view(workspace.snapshots, workspace.benchmark, workspace.cycles)
    shown = headlines(view)
    for start in range(0, len(shown), COLUMNS_PER_ROW):
        row = shown[start : start + COLUMNS_PER_ROW]
        for column, item in zip(st.columns(COLUMNS_PER_ROW), row, strict=True):
            column.metric(item.label, item.value, help=item.hint)
    st.caption(
        return_note(
            view.benchmark_window,
            view.benchmark_partial,
            has_return=view.benchmark_return is not None,
        )
    )
    runs = backtest_view(workspace.backtests, workspace.snapshots, workspace.benchmark).runs
    if runs:
        st.subheader("Backtest runs")
        st.dataframe(run_table(runs), hide_index=True)
        st.caption(RUN_NOTE)
