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
