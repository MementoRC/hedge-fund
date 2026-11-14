"""Empty and error states shared by the pages: say what is missing and which command fills it.

The dashboard is read-only and never builds the demo itself; it only names the command.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from hedge_fund.dashboard.data import RootState
from hedge_fund.viz.load import Workspace

DEMO_COMMAND = "pixi run -e dev demo"


def missing_message(root: Path) -> str:
    return (
        f"No store at `{root}`. Build the synthetic demo world with `{DEMO_COMMAND}` "
        "and reload, or type another store root in the sidebar."
    )


def empty_message(root: Path) -> str:
    return (
        f"`{root}` holds no account snapshots yet, so there is nothing to chart. Fill it with "
        f"`{DEMO_COMMAND}` (synthetic demo world), `hedge-fund backtest run` or "
        "`hedge-fund cycle`; if the sidebar lists unreadable files, fix those first."
    )


def usable_workspace(state: RootState) -> Workspace | None:
    """The workspace when it has an account history to show; otherwise the explained empty
    state is drawn and None is returned.
    """
    if state.workspace is None:
        st.info(missing_message(state.root))
        return None
    if not state.workspace.snapshots:
        st.info(empty_message(state.root))
        return None
    return state.workspace
