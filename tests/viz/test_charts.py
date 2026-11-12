"""The D1 charts: figure traces from a view, themed by a palette."""

from __future__ import annotations

from datetime import date

import pytest

from hedge_fund.viz.charts import (
    drawdown_chart,
    exposure_chart,
    nav_vs_benchmark,
    sector_exposure_chart,
)
from hedge_fund.viz.load import Workspace
from hedge_fund.viz.theme import DARK, LIGHT
from hedge_fund.viz.views import PortfolioView, portfolio_view
from tests.viz.builders import make_cycle, make_nav, make_position, make_snapshots, make_target


def _view() -> PortfolioView:
    first = make_target(
        date(2024, 1, 2), make_position("AAA", "Tech", 0.30), make_position("BBB", "Energy", -0.10)
    )
    second = make_target(
        date(2024, 2, 1), make_position("AAA", "Tech", 0.20), make_position("BBB", "Energy", -0.10)
    )  # gross 0.30, net 0.10; first: gross 0.40, net 0.20
    return portfolio_view(
        make_snapshots(100.0, 120.0, 90.0, 110.0),
        make_nav(50.0, 51.0, 49.0, 52.0),
        (make_cycle(date(2024, 1, 2), target=first), make_cycle(date(2024, 2, 1), target=second)),
    )


def test_nav_vs_benchmark_draws_the_strategy_and_a_dashed_spy() -> None:
    figure = nav_vs_benchmark(_view(), LIGHT)

    strategy, spy = figure.data
    assert (strategy.name, spy.name) == ("Strategy", "SPY")
    assert list(strategy.y) == pytest.approx([100.0, 120.0, 90.0, 110.0])
    assert list(spy.y) == pytest.approx([100.0, 102.0, 98.0, 104.0])
    assert strategy.line.color == LIGHT.strategy
    assert spy.line.color == LIGHT.benchmark
    assert spy.line.dash == "dash"
    assert figure.layout.template.layout.paper_bgcolor == LIGHT.background


def test_nav_vs_benchmark_without_a_benchmark_has_one_trace() -> None:
    view = portfolio_view(make_snapshots(100.0, 101.0, 102.0), None, ())

    figure = nav_vs_benchmark(view, DARK)

    assert [trace.name for trace in figure.data] == ["Strategy"]
    assert figure.layout.template.layout.paper_bgcolor == DARK.background


def test_drawdown_chart_is_a_filled_series_of_non_positive_values() -> None:
    [trace] = drawdown_chart(_view(), LIGHT).data

    assert list(trace.y) == pytest.approx([0.0, 0.0, -0.25, 110 / 120 - 1])
    assert trace.fill == "tozeroy"
    assert tuple(trace.x)[0] == date(2024, 1, 2)


def test_exposure_chart_has_gross_and_net_per_cycle() -> None:
    gross, net = exposure_chart(_view(), LIGHT).data

    assert (gross.name, net.name) == ("Gross", "Net")
    assert list(gross.y) == pytest.approx([0.40, 0.30])
    assert list(net.y) == pytest.approx([0.20, 0.10])
    assert tuple(gross.x) == (date(2024, 1, 2), date(2024, 2, 1))
    assert net.line.dash == "dash"


def test_sector_exposure_chart_stacks_long_and_short_by_sector() -> None:
    figure = sector_exposure_chart(_view(), LIGHT)

    long, short = figure.data
    assert (long.name, short.name) == ("Long", "Short")
    assert tuple(long.x) == tuple(short.x) == ("Energy", "Tech")
    assert list(long.y) == pytest.approx([0.0, 0.20])
    assert list(short.y) == pytest.approx([-0.10, 0.0])
    assert figure.layout.barmode == "relative"


def test_charts_of_an_empty_view_build_without_points() -> None:
    empty = portfolio_view((), None, ())

    for build in (nav_vs_benchmark, drawdown_chart, exposure_chart, sector_exposure_chart):
        assert all(len(trace.x or ()) == 0 for trace in build(empty, LIGHT).data)


def test_the_demo_workspace_draws_every_chart(workspace: Workspace) -> None:
    view = portfolio_view(workspace.snapshots, workspace.benchmark, workspace.cycles)

    for figure in (
        nav_vs_benchmark(view, DARK),
        drawdown_chart(view, DARK),
        exposure_chart(view, DARK),
        sector_exposure_chart(view, DARK),
    ):
        assert len(figure.data) >= 1
        assert len(figure.data[0].x) >= 2
