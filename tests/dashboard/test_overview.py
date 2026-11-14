"""The Overview page: headline figures, benchmark windows and the backtest-run table."""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Iterable
from datetime import date
from pathlib import Path

from streamlit.testing.v1 import AppTest

from hedge_fund.dashboard.format import WHOLE_HISTORY, fmt_money, fmt_percent
from hedge_fund.dashboard.overview import RUN_NOTE, Headline, headlines, run_table
from hedge_fund.domain.cycle import CycleOutcome
from hedge_fund.viz.load import Workspace
from hedge_fund.viz.views import overview_view
from tests.dashboard.builders import make_run, open_page
from tests.dashboard.scripts import (
    overview_of_root,
    overview_read_error,
    overview_with_benchmark,
)
from tests.viz.builders import make_cycle, make_nav, make_snapshots

SNAPSHOTS = make_snapshots(100.0, 120.0, 90.0, 115.0)  # full history +15%, max drawdown 25%
FROM_DAY1 = date(2024, 1, 3)


def _values(view_headlines: Iterable[Headline]) -> dict[str, str]:
    return {item.label: item.value for item in view_headlines}


def _metrics(at: AppTest) -> dict[str, str]:
    return {metric.label: metric.value for metric in at.metric}


def test_headlines_for_a_benchmark_covering_part_of_the_nav() -> None:
    benchmark = make_nav(60.0, 54.0, 66.0, start=FROM_DAY1)
    cycles = (make_cycle(date(2024, 1, 5), outcome=CycleOutcome.BLOCKED),)

    shown = headlines(overview_view(SNAPSHOTS, benchmark, cycles))

    assert _values(shown) == {
        "NAV": "$115.00",
        "Strategy return (full history)": "15.0%",
        "Max drawdown": "25.0%",
        "SPY return (SPY window)": "10.0%",
        "Strategy return (SPY window)": "-4.2%",
        "Last cycle": "blocked",
    }
    help_by_label = {item.label: item.hint for item in shown}
    assert help_by_label["SPY return (SPY window)"] == "SPY window: 2024-01-03 to 2024-01-05"
    assert help_by_label["Last cycle"] == "Cycle of 2024-01-05"


def test_a_benchmark_window_with_too_few_dates_has_no_return_and_reads_n_a() -> None:
    benchmark = make_nav(60.0, 54.0, start=FROM_DAY1)  # two covered dates

    view = overview_view(SNAPSHOTS, benchmark, ())

    assert view.benchmark_window == (FROM_DAY1, date(2024, 1, 4))  # a window exists ...
    assert view.benchmark_return is None  # ... but no return does
    assert view.strategy_return_in_benchmark_window is None
    shown = _values(headlines(view))
    assert shown["SPY return (SPY window)"] == "n/a"
    assert shown["Strategy return (SPY window)"] == "n/a"
    assert shown["Strategy return (full history)"] == "15.0%"


def test_without_a_benchmark_the_spy_figures_read_n_a() -> None:
    view = overview_view(SNAPSHOTS, None, ())

    shown = _values(headlines(view))

    assert view.benchmark_window is None
    assert shown["SPY return (SPY window)"] == "n/a"
    assert shown["Strategy return (SPY window)"] == "n/a"
    assert shown["Last cycle"] == "n/a"


def test_headlines_of_nothing_are_all_n_a() -> None:
    shown = _values(headlines(overview_view((), None, ())))

    assert set(shown.values()) == {"n/a"}


def test_non_finite_figures_read_n_a() -> None:
    view = overview_view(SNAPSHOTS, make_nav(60.0, 54.0, 66.0, start=FROM_DAY1), ())
    bad = dataclasses.replace(
        view, total_return=math.nan, max_drawdown=math.inf, benchmark_return=-math.inf
    )
    assert math.isnan(bad.total_return)  # type: ignore[arg-type]

    shown = _values(headlines(bad))

    assert shown["Strategy return (full history)"] == "n/a"
    assert shown["Max drawdown"] == "n/a"
    assert shown["SPY return (SPY window)"] == "n/a"
    assert shown["NAV"] == "$115.00"


def test_the_run_table_labels_the_returns_as_full_window() -> None:
    table = run_table([make_run()])

    assert list(table.columns) == [
        "Run",
        "Window",
        "Trading days",
        "Rebalances",
        "Return (full window)",
        "SPY (full window)",
        "Excess (full window)",
        "Max drawdown",
        "Sharpe",
        "Beta",
        "Breaker days",
    ]
    row = table.iloc[0]
    assert row["Run"] == "run-1"
    assert row["Window"] == "2024-01-10 to 2024-03-08"
    assert row["Return (full window)"] == "10.0%"
    assert row["SPY (full window)"] == "5.0%"
    assert row["Excess (full window)"] == "5.0%"
    assert row["Max drawdown"] == "20.0%"
    assert row["Sharpe"] == "1.50"
    assert row["Beta"] == "0.25"
    assert row["Breaker days"] == 2


def test_the_run_table_shows_n_a_for_nan_infinite_and_missing_ratios() -> None:
    run = make_run(sharpe=math.nan, beta=math.inf)
    assert math.isnan(run.sharpe)  # type: ignore[arg-type]
    unknown = dataclasses.replace(run, sharpe=None, beta=None)  # the NaN-free copy

    nan_row = run_table([run]).iloc[0]
    none_row = run_table([unknown]).iloc[0]

    assert (nan_row["Sharpe"], nan_row["Beta"]) == ("n/a", "n/a")
    assert (none_row["Sharpe"], none_row["Beta"]) == ("n/a", "n/a")
    assert nan_row["Return (full window)"] == none_row["Return (full window)"] == "10.0%"


def test_the_run_note_says_the_run_figures_are_full_window(demo_root: Path) -> None:
    assert "whole window" in RUN_NOTE
    assert "SPY" in RUN_NOTE

    at = open_page(overview_of_root, demo_root)

    assert [caption.value for caption in at.caption].count(RUN_NOTE) == 1


def test_an_unreadable_root_fails_closed_on_the_overview() -> None:
    at = open_page(overview_read_error)

    assert len(at.exception) == 0
    assert len(at.metric) == 0
    assert len(at.dataframe) == 0
    assert "denied" in at.error[0].value


def test_the_demo_overview_shows_the_headlines_and_the_run(
    demo_root: Path, workspace: Workspace
) -> None:
    view = overview_view(workspace.snapshots, workspace.benchmark, workspace.cycles)

    at = open_page(overview_of_root, demo_root)

    assert len(at.exception) == 0
    assert [title.value for title in at.title] == ["Overview"]
    assert _metrics(at) == _values(headlines(view))
    assert _metrics(at)["NAV"] == fmt_money(view.nav)
    assert _metrics(at)["Max drawdown"] == fmt_percent(view.max_drawdown)
    [table] = at.dataframe
    assert table.value.iloc[0]["Run"] == workspace.backtests[0].run_id
    assert "Return (full window)" in table.value.columns
    assert RUN_NOTE in [caption.value for caption in at.caption]
    assert any(WHOLE_HISTORY in caption.value for caption in at.caption)


def test_an_empty_root_explains_how_to_fill_it() -> None:
    at = open_page(overview_of_root)

    assert len(at.exception) == 0
    assert len(at.metric) == 0
    assert len(at.dataframe) == 0
    assert "pixi run -e dev demo" in at.info[0].value


def test_a_partial_benchmark_is_labelled_on_the_page() -> None:
    at = open_page(overview_with_benchmark, None, 60.0, 54.0, 66.0)

    assert len(at.exception) == 0
    assert _metrics(at)["SPY return (SPY window)"] == "10.0%"
    assert _metrics(at)["Strategy return (full history)"] == "15.0%"
    notes = [caption.value for caption in at.caption]
    assert any("partial window" in note and "2024-01-03 to 2024-01-05" in note for note in notes)
    assert len(at.dataframe) == 0  # no backtest summary in a hand-built root


def test_a_window_without_a_return_shows_n_a_and_says_why() -> None:
    at = open_page(overview_with_benchmark, None, 60.0, 54.0)

    assert len(at.exception) == 0
    assert _metrics(at)["SPY return (SPY window)"] == "n/a"
    assert _metrics(at)["Strategy return (SPY window)"] == "n/a"
    assert any("fewer than 3 dates" in caption.value for caption in at.caption)


def test_no_benchmark_says_so() -> None:
    at = open_page(overview_with_benchmark)

    assert _metrics(at)["SPY return (SPY window)"] == "n/a"
    assert any("No SPY series" in caption.value for caption in at.caption)
