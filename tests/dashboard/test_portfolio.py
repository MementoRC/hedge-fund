"""The Portfolio & NAV page: charts, exposure caveat, positions table, empty states."""

from __future__ import annotations

import math
from pathlib import Path

from streamlit.testing.v1 import AppTest

from hedge_fund.dashboard.format import WHOLE_HISTORY
from hedge_fund.dashboard.portfolio import EXPOSURE_NOTE, positions_table
from hedge_fund.viz.load import Workspace
from hedge_fund.viz.views import portfolio_view
from tests.dashboard.builders import open_page
from tests.dashboard.scripts import (
    portfolio_of_root,
    portfolio_read_error,
    portfolio_with_flat_target,
    portfolio_with_targets,
    portfolio_without_target,
)
from tests.viz.builders import DAY0, make_cycle, make_position, make_snapshots, make_target


def _charts(at: AppTest) -> int:
    return len(at.get("plotly_chart"))


def _captions(at: AppTest) -> list[str]:
    return [caption.value for caption in at.caption]


def test_the_exposure_note_says_these_are_targets_at_each_rebalance() -> None:
    assert "target weights at each rebalance" in EXPOSURE_NOTE
    assert "not the holdings between rebalances" in EXPOSURE_NOTE


def test_the_positions_table_formats_target_weights() -> None:
    target = make_target(
        DAY0, make_position("AAA", "Tech", 0.30), make_position("BBB", "Energy", -0.10)
    )
    view = portfolio_view(
        make_snapshots(100.0, 101.0, 102.0), None, (make_cycle(DAY0, target=target),)
    )

    table = positions_table(view)

    assert list(table.columns) == ["Ticker", "Sector", "Side", "Target weight"]
    assert table.values.tolist() == [
        ["AAA", "Tech", "long", "30.0%"],
        ["BBB", "Energy", "short", "-10.0%"],
    ]


def test_a_non_finite_weight_reads_n_a() -> None:
    nan_view = portfolio_view(
        make_snapshots(100.0, 101.0, 102.0),
        None,
        (make_cycle(DAY0, target=make_target(DAY0, make_position("AAA", "Tech", math.nan))),),
    )
    inf_view = portfolio_view(
        make_snapshots(100.0, 101.0, 102.0),
        None,
        (make_cycle(DAY0, target=make_target(DAY0, make_position("AAA", "Tech", math.inf))),),
    )
    assert math.isnan(nan_view.positions[0].weight)

    assert positions_table(nan_view).iloc[0]["Target weight"] == "n/a"
    assert positions_table(inf_view).iloc[0]["Target weight"] == "n/a"


def test_the_positions_table_of_an_empty_view_has_its_columns_and_no_rows() -> None:
    table = positions_table(portfolio_view((), None, ()))

    assert list(table.columns) == ["Ticker", "Sector", "Side", "Target weight"]
    assert len(table) == 0


def test_the_demo_portfolio_draws_four_charts_and_the_positions(
    demo_root: Path, workspace: Workspace
) -> None:
    view = portfolio_view(workspace.snapshots, workspace.benchmark, workspace.cycles)

    at = open_page(portfolio_of_root, demo_root)

    assert len(at.exception) == 0
    assert [title.value for title in at.title] == ["Portfolio & NAV"]
    assert _charts(at) == 4
    [table] = at.dataframe
    assert len(table.value) == len(view.positions)
    assert EXPOSURE_NOTE in _captions(at)
    assert any(WHOLE_HISTORY in caption for caption in _captions(at))


def test_an_empty_root_explains_how_to_fill_it() -> None:
    at = open_page(portfolio_of_root)

    assert len(at.exception) == 0
    assert _charts(at) == 0
    assert "pixi run -e dev demo" in at.info[0].value


def test_an_unreadable_root_fails_closed_on_the_portfolio() -> None:
    at = open_page(portfolio_read_error)

    assert len(at.exception) == 0
    assert _charts(at) == 0
    assert len(at.dataframe) == 0
    assert "denied" in at.error[0].value


def test_a_partial_benchmark_is_labelled_under_the_nav_chart() -> None:
    at = open_page(portfolio_with_targets, None, 60.0, 54.0, 66.0)

    assert len(at.exception) == 0
    assert _charts(at) == 4
    assert any(
        "partial window" in caption and "2024-01-03 to 2024-01-05" in caption
        for caption in _captions(at)
    )
    assert any("Target weights of the cycle of 2024-01-04" in c for c in _captions(at))
    assert EXPOSURE_NOTE in _captions(at)


def test_no_benchmark_says_so_under_the_nav_chart() -> None:
    at = open_page(portfolio_with_targets)

    assert _charts(at) == 4
    assert any("No SPY series" in caption for caption in _captions(at))


def test_without_a_target_only_the_nav_charts_are_drawn_and_the_page_explains() -> None:
    at = open_page(portfolio_without_target)

    assert len(at.exception) == 0
    assert _charts(at) == 2
    assert len(at.dataframe) == 0
    assert "pixi run -e dev demo" in at.info[0].value


def test_a_target_with_no_positions_draws_exposure_but_no_sector_chart_or_table() -> None:
    at = open_page(portfolio_with_flat_target)

    assert len(at.exception) == 0
    assert _charts(at) == 3
    assert len(at.dataframe) == 0
