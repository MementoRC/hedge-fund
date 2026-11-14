"""The Backtests page: full-window run figures, breaker days, biases, equity vs SPY."""

from __future__ import annotations

import math

from hedge_fund.dashboard.backtests import bias_lines, run_metrics
from tests.dashboard.builders import make_run


def test_run_metrics_label_every_stored_figure_full_window_and_carry_the_run_window() -> None:
    shown = run_metrics(make_run())

    assert [(item.label, item.value) for item in shown] == [
        ("Return (full window)", "10.0%"),
        ("SPY (full window)", "5.0%"),
        ("Excess (full window)", "5.0%"),
        ("Max drawdown (full window)", "20.0%"),
        ("Sharpe (full window)", "1.50"),
        ("Beta (full window)", "0.25"),
        ("Breaker days", "2"),
        ("Trading days", "40"),
    ]
    assert shown[0].hint == "Run window: 2024-01-10 to 2024-03-08"


def test_non_finite_and_missing_run_figures_read_n_a() -> None:
    run = make_run(total_return=math.nan, excess_return=-math.inf, sharpe=None, beta=math.inf)
    assert math.isnan(run.total_return)

    values = {item.label: item.value for item in run_metrics(run)}

    assert values["Return (full window)"] == "n/a"
    assert values["Excess (full window)"] == "n/a"
    assert values["Sharpe (full window)"] == "n/a"
    assert values["Beta (full window)"] == "n/a"
    assert values["SPY (full window)"] == "5.0%"


def test_bias_lines_are_a_markdown_list() -> None:
    assert bias_lines(make_run(biases=("survivorship", "no borrow fees"))) == (
        "- survivorship\n- no borrow fees"
    )
    assert bias_lines(make_run()) == ""
