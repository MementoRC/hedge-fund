"""The Backtests page: full-window run figures, breaker days, biases, equity vs SPY."""

from __future__ import annotations

import math

import pytest
from streamlit.testing.v1 import AppTest

from hedge_fund.dashboard.backtests import bias_lines, run_metrics
from tests.dashboard.builders import make_run
from tests.dashboard.scripts import headline_cards


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


def test_run_metrics_read_n_a_when_no_return_or_drawdown_exists() -> None:
    run = make_run(
        total_return=None,
        benchmark_return=None,
        excess_return=None,
        max_drawdown=math.nan,
    )

    values = {item.label: item.value for item in run_metrics(run)}

    for label in (
        "Return (full window)",
        "SPY (full window)",
        "Excess (full window)",
        "Max drawdown (full window)",
    ):
        assert values[label] == "n/a"


def test_every_run_metric_carries_its_hint() -> None:
    window = "Run window: 2024-01-10 to 2024-03-08"

    assert [item.hint for item in run_metrics(make_run())] == [
        window,
        window,
        window,
        window,
        window,
        window,
        "Daily-loss plus drawdown breaker days",
        window,
    ]


@pytest.mark.parametrize(("count", "per_row", "rows"), [(8, 4, 2), (7, 3, 3)])
def test_show_headlines_draws_every_card_in_order_with_a_short_last_row(
    count: int, per_row: int, rows: int
) -> None:
    at = AppTest.from_function(headline_cards, default_timeout=60, args=(count, per_row)).run()

    assert not at.exception
    assert [(m.label, m.value) for m in at.metric] == [(f"card {i}", str(i)) for i in range(count)]
    assert len(at.columns) == rows * per_row


def test_bias_lines_are_a_markdown_list() -> None:
    assert bias_lines(make_run(biases=("survivorship", "no borrow fees"))) == (
        "- survivorship\n- no borrow fees"
    )
    assert bias_lines(make_run()) == ""
