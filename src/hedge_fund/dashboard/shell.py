"""The dashboard shell: the sidebar every page shows, the current root, the page list
and the entry point."""

from __future__ import annotations

import streamlit as st

from hedge_fund.dashboard import overview, portfolio
from hedge_fund.dashboard.data import DEFAULT_ROOT, RootState, load_state
from hedge_fund.viz.load import RootKind

MAX_LISTED = 10


def current_state() -> RootState:
    return load_state(st.session_state.get("root", DEFAULT_ROOT))


def render_sidebar() -> RootState:
    """The root picker, what kind of root this is, and what could not be read."""
    st.session_state.setdefault("root", str(DEFAULT_ROOT))
    st.sidebar.text_input("Store root", key="root")
    state = current_state()
    _show_kind(state)
    _show_read_error(state)
    _show_unreadable(state)
    return state


def _show_kind(state: RootState) -> None:
    if state.workspace is None and state.read_error is None:
        st.sidebar.caption("No such directory.")
    elif state.kind is RootKind.DEMO:
        st.sidebar.markdown(":orange-badge[DEMO] Synthetic data, not a real result.")
    elif state.kind is RootKind.BACKTEST:
        st.sidebar.caption("Backtest root.")
    elif state.is_live:
        st.sidebar.caption("Live root", help=state.kind_error)
    elif state.kind_error is not None:
        st.sidebar.caption(f"Not a demo or backtest root: {state.kind_error}")


def _show_read_error(state: RootState) -> None:
    if state.read_error is not None:
        st.sidebar.warning(f"The root could not be read: {state.read_error}")


def _show_unreadable(state: RootState) -> None:
    if state.workspace is None or not state.workspace.unreadable:
        return
    problems = state.workspace.unreadable
    count = len(problems)
    lines = [f"- {problem}" for problem in problems[:MAX_LISTED]]
    if count > MAX_LISTED:
        lines.append(f"- and {count - MAX_LISTED} more")
    noun = "file" if count == 1 else "files"
    st.sidebar.warning(
        f"{count} unreadable {noun}; pages show what could be read.\n\n" + "\n".join(lines)
    )


def overview_page() -> None:
    overview.render(current_state())


def portfolio_page() -> None:
    portfolio.render(current_state())


def main() -> None:
    """Entry point of ``app.py``: navigation over the pages, the sidebar, then the page."""
    st.set_page_config(page_title="Hedge Fund", layout="wide")
    page = st.navigation(
        [
            st.Page(overview_page, title="Overview", url_path="overview", default=True),
            st.Page(portfolio_page, title="Portfolio & NAV", url_path="portfolio"),
        ]
    )
    render_sidebar()
    page.run()
