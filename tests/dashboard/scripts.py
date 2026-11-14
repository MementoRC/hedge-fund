"""Scripts run by ``AppTest.from_function``: each must carry its own imports."""

from __future__ import annotations


def usable_probe(case: str) -> None:
    import dataclasses

    import streamlit as st

    from hedge_fund.dashboard.states import usable_workspace
    from tests.dashboard.builders import benchmark_state, make_state, missing_state

    states = {
        "missing": missing_state(),
        "empty": make_state(),
        "usable": benchmark_state(),
        "refused": dataclasses.replace(benchmark_state(), kind_error="not a demo"),
        "read_error": dataclasses.replace(benchmark_state(), read_error="denied"),
    }
    workspace = usable_workspace(states[case])
    st.caption("usable" if workspace is not None else "unusable")


def sidebar_only() -> None:
    from hedge_fund.dashboard.shell import render_sidebar

    render_sidebar()


def overview_of_root() -> None:
    from hedge_fund.dashboard.overview import render
    from hedge_fund.dashboard.shell import current_state, render_sidebar

    render_sidebar()
    render(current_state())


def overview_with_benchmark(*closes: float) -> None:
    from hedge_fund.dashboard.overview import render
    from tests.dashboard.builders import benchmark_state

    render(benchmark_state(*closes))


def overview_read_error() -> None:
    import dataclasses

    from hedge_fund.dashboard.overview import render
    from tests.dashboard.builders import benchmark_state

    render(dataclasses.replace(benchmark_state(60.0, 54.0, 66.0), read_error="denied"))


def portfolio_of_root() -> None:
    from hedge_fund.dashboard.portfolio import render
    from hedge_fund.dashboard.shell import current_state, render_sidebar

    render_sidebar()
    render(current_state())


def portfolio_page_with_sidebar() -> None:
    from hedge_fund.dashboard.shell import portfolio_page, render_sidebar

    render_sidebar()
    portfolio_page()


def portfolio_with_targets(*closes: float) -> None:
    from hedge_fund.dashboard.portfolio import render
    from tests.dashboard.builders import targeted_state

    render(targeted_state(*closes))


def portfolio_without_target() -> None:
    from hedge_fund.dashboard.portfolio import render
    from tests.dashboard.builders import benchmark_state

    render(benchmark_state())


def portfolio_with_flat_target() -> None:
    from hedge_fund.dashboard.portfolio import render
    from tests.dashboard.builders import flat_state

    render(flat_state())


def portfolio_read_error() -> None:
    import dataclasses

    from hedge_fund.dashboard.portfolio import render
    from tests.dashboard.builders import targeted_state

    render(dataclasses.replace(targeted_state(60.0, 54.0, 66.0), read_error="denied"))


def sidebar_with_state(case: str) -> None:
    import dataclasses
    from unittest.mock import patch

    from hedge_fund.dashboard import shell
    from tests.dashboard.builders import make_state

    state = make_state()
    if case == "read_error":
        state = dataclasses.replace(state, read_error="denied")
    with patch.object(shell, "current_state", lambda: state):
        shell.render_sidebar()


def page_of_root(page: str) -> None:
    import importlib

    from hedge_fund.dashboard.shell import current_state, render_sidebar

    render_sidebar()
    importlib.import_module(f"hedge_fund.dashboard.{page}").render(current_state())


def page_of_builder(page: str, builder: str, *closes: float) -> None:
    import importlib

    from tests.dashboard import builders

    state = getattr(builders, builder)(*closes)
    importlib.import_module(f"hedge_fund.dashboard.{page}").render(state)


def page_in_state(page: str, case: str) -> None:
    """``rich_state`` with SPY on three dates, flagged by hand as ``case``, shown with the
    sidebar: "read_error", "live" (``kind_error`` and ``is_live``) or "refused" (``kind_error``).
    """
    import dataclasses
    import importlib
    from unittest.mock import patch

    from hedge_fund.dashboard import shell
    from tests.dashboard.builders import LIVE_REFUSAL, rich_state

    state = rich_state(60.0, 54.0, 66.0)
    if case == "read_error":
        state = dataclasses.replace(state, read_error="denied")
    elif case == "live":
        state = dataclasses.replace(state, kind_error=LIVE_REFUSAL, is_live=True)
    elif case == "refused":
        state = dataclasses.replace(state, kind_error=LIVE_REFUSAL)
    with patch.object(shell, "current_state", lambda: state):
        shell.render_sidebar()
        importlib.import_module(f"hedge_fund.dashboard.{page}").render(state)


def readable_probe(case: str) -> None:
    import dataclasses

    import streamlit as st

    from hedge_fund.dashboard.states import readable_workspace
    from tests.dashboard.builders import benchmark_state, make_state, missing_state

    states = {
        "missing": missing_state(),
        "empty": make_state(),
        "refused": dataclasses.replace(make_state(), kind_error="not a demo"),
        "live": dataclasses.replace(benchmark_state(), kind_error="not a demo", is_live=True),
        "read_error": dataclasses.replace(benchmark_state(), read_error="denied"),
    }
    workspace = readable_workspace(states[case])
    st.caption("readable" if workspace is not None else "unreadable")
