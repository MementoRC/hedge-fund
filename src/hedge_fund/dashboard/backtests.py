"""Backtests page: the run table, the selected run's full-window figures, breaker days and
residual biases, and equity vs SPY.

Run figures (``BacktestRun``) are stored full-window metrics; the SPY line on the equity
chart covers only the NAV dates it has (``benchmark_window`` / ``benchmark_partial``).
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from hedge_fund.dashboard.data import RootState
from hedge_fund.dashboard.format import NO_SPY
from hedge_fund.dashboard.overview import RUN_NOTE, run_table, show_headlines
from hedge_fund.dashboard.palette import PALETTE
from hedge_fund.dashboard.picker import pick_index
from hedge_fund.dashboard.states import DEMO_COMMAND, nothing_message, readable_workspace
from hedge_fund.viz.charts import nav_vs_benchmark
from hedge_fund.viz.format import chart_note, escape_markdown, run_metrics
from hedge_fund.viz.views import BacktestRun, BacktestView, backtest_view

METRICS_PER_ROW = 4
EQUITY_NOTE = (
    "Equity is this root's account history, indexed to 100; the run figures above are over "
    "each run's whole window."
)
NO_EQUITY = (
    "This root holds no account snapshots, so there is no equity curve to draw; the run "
    f"figures above come from the stored summaries. `{DEMO_COMMAND}` builds a synthetic "
    "demo world with both."
)
NO_BIAS = "The run lists no residual bias."


def bias_lines(run: BacktestRun) -> str:
    """The run's residual biases as a markdown list ("" when it lists none); each bias is
    escaped so it renders literally.
    """
    return "\n".join(f"- {escape_markdown(bias)}" for bias in run.biases)


def render(state: RootState) -> None:
    st.title("Backtests")
    workspace = readable_workspace(state)
    if workspace is None:
        return
    view = backtest_view(workspace.backtests, workspace.snapshots, workspace.benchmark)
    if not view.runs:
        st.info(nothing_message(state.root, "backtest summaries", "hedge-fund backtest run"))
        return
    st.dataframe(run_table(view.runs), hide_index=True)
    st.caption(RUN_NOTE)
    _show_run(_selected(view, state.root))
    _equity(view)


def _selected(view: BacktestView, root: Path) -> BacktestRun:
    """The run picked in the selector (load order). The key is scoped to the root, and an
    index out of range (or none) falls back to the first run.
    """
    index = pick_index(
        "Run",
        list(range(len(view.runs))),
        lambda i: view.runs[i].run_id,
        key=f"run:{root}",
    )
    return view.runs[index]


def _show_run(run: BacktestRun) -> None:
    st.subheader(f"Run {escape_markdown(run.run_id)}")  # a file-derived id, shown literally
    show_headlines(run_metrics(run), per_row=METRICS_PER_ROW)
    st.markdown("**Residual biases**")
    lines = bias_lines(run)
    if lines:
        st.markdown(lines)
    else:
        st.caption(NO_BIAS)


def _equity(view: BacktestView) -> None:
    st.subheader("Equity vs SPY")
    if not view.nav_index:
        st.info(NO_EQUITY)
        return
    st.plotly_chart(nav_vs_benchmark(view, PALETTE), theme=None)
    st.caption(chart_note(view.benchmark_window, view.benchmark_partial, no_spy=NO_SPY))
    st.caption(EQUITY_NOTE)
