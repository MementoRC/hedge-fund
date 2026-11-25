"""The Backtests page: full-window run figures, breaker days, biases, equity vs SPY."""

from __future__ import annotations

import math
from datetime import timedelta
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from hedge_fund.dashboard import backtests
from hedge_fund.dashboard.backtests import (
    EQUITY_NOTE,
    NO_BIAS,
    NO_EQUITY,
    bias_lines,
)
from hedge_fund.dashboard.overview import RUN_NOTE
from hedge_fund.viz.format import WHOLE_HISTORY, escape_markdown, fmt_date, fmt_percent, run_metrics
from hedge_fund.viz.load import Workspace
from hedge_fund.viz.views import backtest_view
from tests.dashboard.builders import (
    MARKDOWN_TEXT,
    RUN_ID,
    live_root,
    make_backtest_report,
    make_run,
    open_page,
    two_run_root,
    two_runs_state,
)
from tests.dashboard.scripts import (
    headline_cards,
    page_in_state,
    page_of_builder,
    page_of_root,
)
from tests.viz.builders import DAY0


def _charts(at: AppTest) -> int:
    return len(at.get("plotly_chart"))


def _captions(at: AppTest) -> list[str]:
    return [caption.value for caption in at.main.caption]


def _metrics(at: AppTest) -> dict[str, str]:
    return {metric.label: metric.value for metric in at.metric}


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


def test_bias_lines_escape_each_bias_but_not_the_list_prefix() -> None:
    run = make_run(biases=("*survivorship* `x`", "costs $5"))

    assert bias_lines(run) == "- \\*survivorship\\* \\`x\\`\n- costs \\$5"


def test_the_equity_note_says_whose_history_the_chart_is() -> None:
    assert "account history" in EQUITY_NOTE
    assert "whole window" in EQUITY_NOTE


def test_the_demo_backtest_shows_the_run_its_figures_biases_and_equity(
    demo_root: Path, workspace: Workspace
) -> None:
    view = backtest_view(workspace.backtests, workspace.snapshots, workspace.benchmark)

    at = open_page(page_of_root, demo_root, "backtests")

    assert len(at.exception) == 0
    assert len(at.error) == 0
    assert [title.value for title in at.title] == ["Backtests"]
    [table] = at.dataframe
    assert table.value.iloc[0]["Run"] == workspace.backtests[0].run_id
    assert "SPY (full window)" in table.value.columns
    expected = {item.label: item.value for item in run_metrics(view.runs[0])}
    assert _metrics(at) == expected
    assert _charts(at) == 1
    captions = _captions(at)
    assert RUN_NOTE in captions
    assert EQUITY_NOTE in captions
    assert any(WHOLE_HISTORY in caption for caption in captions)
    markdown = "\n".join(m.value for m in at.main.markdown)
    assert escape_markdown(workspace.backtests[0].biases[0]) in markdown


def test_a_missing_root_names_the_demo_command() -> None:
    at = open_page(page_of_root, None, "backtests")

    assert len(at.exception) == 0
    assert "pixi run -e dev demo" in at.info[0].value
    assert len(at.dataframe) == 0


def test_a_root_without_a_summary_names_the_backtest_command(tmp_path: Path) -> None:
    at = open_page(page_of_root, tmp_path, "backtests")

    assert len(at.exception) == 0
    assert len(at.error) == 0
    [info] = at.info
    assert "holds no backtest summaries" in info.value
    assert "`hedge-fund backtest run`" in info.value
    assert "`pixi run -e dev demo`" in info.value


def test_a_real_live_root_has_no_runs_and_a_neutral_caption(tmp_path: Path) -> None:
    at = open_page(page_of_root, live_root(tmp_path), "backtests")

    assert len(at.exception) == 0
    assert len(at.error) == 0
    assert [caption.value for caption in at.sidebar.caption] == ["Live root"]
    assert "`hedge-fund backtest run`" in at.info[0].value


@pytest.mark.parametrize(
    ("case", "caption"),
    [("live", "Live root"), ("refused", "Not a demo or backtest root: ")],
)
def test_a_classify_root_refusal_still_shows_the_runs(case: str, caption: str) -> None:
    at = open_page(page_in_state, None, "backtests", case)

    assert len(at.exception) == 0
    assert len(at.error) == 0
    [shown] = [c.value for c in at.sidebar.caption]
    assert shown.startswith(caption)
    [table] = at.dataframe
    assert table.value.iloc[0]["Run"] == RUN_ID
    assert len(at.metric) == 8
    assert _charts(at) == 1


def test_a_root_that_could_not_be_read_fails_closed_on_the_backtests() -> None:
    at = open_page(page_in_state, None, "backtests", "read_error")

    assert len(at.exception) == 0
    assert "denied" in at.error[0].value
    assert len(at.dataframe) == 0
    assert len(at.metric) == 0
    assert _charts(at) == 0


def test_a_partial_benchmark_is_named_under_the_equity_chart() -> None:
    at = open_page(page_of_builder, None, "backtests", "rich_state", 60.0, 54.0, 66.0)

    assert len(at.exception) == 0
    assert _charts(at) == 1
    window = f"{fmt_date(DAY0 + timedelta(days=1))} to {fmt_date(DAY0 + timedelta(days=3))}"
    assert any("partial window" in c and window in c for c in _captions(at))


def test_a_benchmark_window_without_a_return_is_drawn_and_the_run_figures_stay_stored() -> None:
    at = open_page(page_of_builder, None, "backtests", "rich_state", 60.0, 54.0)

    assert len(at.exception) == 0
    assert _charts(at) == 1  # two covered dates still draw a line
    window = f"{fmt_date(DAY0 + timedelta(days=1))} to {fmt_date(DAY0 + timedelta(days=2))}"
    assert any("partial window" in c and window in c for c in _captions(at))
    stored = make_backtest_report().metrics.benchmark.total_return
    assert _metrics(at)["SPY (full window)"] == fmt_percent(stored)
    assert not any("(SPY window)" in c for c in _captions(at))  # no window return is claimed


def test_runs_without_snapshots_show_the_figures_and_explain_the_missing_chart() -> None:
    at = open_page(page_of_builder, None, "backtests", "runs_only_state")

    assert len(at.exception) == 0
    assert len(at.dataframe) == 1
    assert len(at.metric) == 8
    assert _charts(at) == 0
    assert NO_EQUITY in [info.value for info in at.info]
    assert "`pixi run -e dev demo`" in NO_EQUITY


def test_another_run_can_be_picked_and_a_run_without_biases_says_so() -> None:
    at = open_page(page_of_builder, None, "backtests", "two_runs_state")

    assert at.selectbox[0].options == [RUN_ID, "run-b"]
    [table] = at.dataframe
    assert len(table.value) == 2
    assert NO_BIAS not in _captions(at)

    at.selectbox[0].select_index(1).run()

    assert len(at.exception) == 0
    assert NO_BIAS in _captions(at)


def test_switching_to_a_root_with_fewer_runs_opens_its_first_run(
    demo_root: Path, tmp_path: Path, workspace: Workspace
) -> None:
    at = open_page(page_of_root, two_run_root(demo_root, tmp_path), "backtests")
    assert len(at.selectbox[0].options) == 2
    at.selectbox[0].select_index(1).run()  # the second run: row index 1 of two
    picked = escape_markdown(at.selectbox[0].options[1])
    assert [s.value for s in at.subheader][0] == f"Run {picked}"

    at.sidebar.text_input(key="root").set_value(str(demo_root)).run()  # one run only

    assert len(at.exception) == 0
    [table] = at.dataframe
    assert len(table.value) == 1
    assert at.selectbox[0].options == [workspace.backtests[0].run_id]
    only = escape_markdown(workspace.backtests[0].run_id)
    assert [s.value for s in at.subheader][0] == f"Run {only}"


def test_the_run_heading_shows_the_run_id_literally() -> None:
    at = open_page(page_of_builder, None, "backtests", "markdown_run_state")

    assert len(at.exception) == 0
    assert at.selectbox[0].options == [f"run {MARKDOWN_TEXT}"]
    assert [s.value for s in at.subheader][0] == r"Run run \`boom\` costs \$5"


@pytest.mark.parametrize("picked", [5, -1, None])
def test_an_out_of_range_or_missing_selection_falls_back_to_the_first_run(
    monkeypatch: pytest.MonkeyPatch, picked: int | None
) -> None:
    workspace = two_runs_state().workspace
    assert workspace is not None
    view = backtest_view(workspace.backtests, workspace.snapshots, workspace.benchmark)
    monkeypatch.setattr(backtests.st, "selectbox", lambda *args, **kwargs: picked)

    assert backtests._selected(view, Path("r")).run_id == view.runs[0].run_id
