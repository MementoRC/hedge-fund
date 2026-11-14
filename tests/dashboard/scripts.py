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
