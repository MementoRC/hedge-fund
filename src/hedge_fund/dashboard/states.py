"""Empty and error states shared by the pages: say what is missing and which command fills it.

The dashboard is read-only and never builds the demo itself; it only names the command.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from hedge_fund.dashboard.data import RootState
from hedge_fund.dashboard.format import escape_markdown
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


def nothing_message(root: Path, what: str, command: str) -> str:
    """The empty state of a page whose input (``what``) the root does not hold."""
    return (
        f"`{root}` holds no {what} yet. `{command}` writes them, or `{DEMO_COMMAND}` builds "
        "a synthetic demo world that has them."
    )


def readable_workspace(state: RootState) -> Workspace | None:
    """The workspace of a root that exists and could be read, whatever it holds.

    A read failure draws an error and shows no data; a missing root draws the demo empty
    state. ``kind_error`` is not a reason to refuse: ``classify_root`` refuses every ordinary
    live or paper root.
    """
    if state.read_error is not None:
        reason = escape_markdown(state.read_error)  # exception text: shown literally
        st.error(f"Could not read `{state.root}`: {reason}. No data is shown from it.")
        return None
    if state.workspace is None:
        st.info(missing_message(state.root))
        return None
    return state.workspace


def usable_workspace(state: RootState) -> Workspace | None:
    """The workspace when it has an account history to show; otherwise the explained empty
    state is drawn and None is returned. A root that could not be read is refused outright:
    an error is drawn and no data is shown from it.
    """
    workspace = readable_workspace(state)
    if workspace is None:
        return None
    if not workspace.snapshots:
        st.info(empty_message(state.root))
        return None
    return workspace
