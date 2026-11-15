"""The dashboard shell: the sidebar every page shows, the current root, the page list
and the entry point."""

from __future__ import annotations

import streamlit as st

from hedge_fund.dashboard import backtests, cycles, overview, portfolio, research
from hedge_fund.dashboard.data import DEFAULT_ROOT, RootState, load_state
from hedge_fund.dashboard.format import escape_markdown
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
        # is_live implies kind_error is set (data.py:82); the guard is for the type checker.
        tip = escape_markdown(state.kind_error) if state.kind_error is not None else None
        st.sidebar.caption("Live root", help=tip)
    elif state.kind_error is not None:
        st.sidebar.caption(f"Not a demo or backtest root: {escape_markdown(state.kind_error)}")


def _show_read_error(state: RootState) -> None:
    if state.read_error is not None:
        st.sidebar.warning(f"The root could not be read: {escape_markdown(state.read_error)}")


def _show_unreadable(state: RootState) -> None:
    if state.workspace is None or not state.workspace.unreadable:
        return
    problems = state.workspace.unreadable
    count = len(problems)
    # Each entry is "<where>: <reason>" from files on disk; the "- " prefix stays markdown.
    lines = [f"- {escape_markdown(problem)}" for problem in problems[:MAX_LISTED]]
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


def cycles_page() -> None:
    cycles.render(current_state())


def research_page() -> None:
    research.render(current_state())


def backtests_page() -> None:
    backtests.render(current_state())


def main() -> None:
    """Entry point of ``app.py``: navigation over the pages, the sidebar, then the page."""
    st.set_page_config(page_title="Hedge Fund", layout="wide")
    page = st.navigation(
        [
            st.Page(overview_page, title="Overview", url_path="overview", default=True),
            st.Page(portfolio_page, title="Portfolio & NAV", url_path="portfolio"),
            st.Page(cycles_page, title="Cycles", url_path="cycles"),
            st.Page(research_page, title="Research & attribution", url_path="research"),
            st.Page(backtests_page, title="Backtests", url_path="backtests"),
        ]
    )
    render_sidebar()
    page.run()
