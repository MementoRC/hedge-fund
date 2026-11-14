"""The charts: figure traces from a view, themed by a palette."""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable
from datetime import date

import plotly.graph_objects as go
import pytest

from hedge_fund.viz.charts import (
    FUNNEL_STAGES,
    drawdown_chart,
    exposure_chart,
    funnel_chart,
    llm_cost_chart,
    nav_vs_benchmark,
    sector_exposure_chart,
)
from hedge_fund.viz.load import Workspace
from hedge_fund.viz.theme import DARK, LIGHT, Palette
from hedge_fund.viz.views import CostRow, Funnel, PortfolioView, backtest_view, portfolio_view
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


def test_nav_vs_benchmark_plots_a_partial_benchmark_over_its_covered_dates() -> None:
    view = portfolio_view(
        make_snapshots(100.0, 120.0, 90.0, 110.0),
        make_nav(60.0, 54.0, 66.0, start=date(2024, 1, 3)),
        (),
    )

    strategy, spy = nav_vs_benchmark(view, LIGHT).data

    assert list(strategy.x) == [date(2024, 1, d) for d in (2, 3, 4, 5)]
    assert list(spy.x) == [date(2024, 1, d) for d in (3, 4, 5)]
    assert list(spy.y) == pytest.approx([120.0, 108.0, 132.0])


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


@pytest.mark.parametrize(
    ("build", "y_title", "percent"),
    [
        (nav_vs_benchmark, "Index", False),
        (drawdown_chart, "Drawdown", True),
        (exposure_chart, "Share of capital", True),
        (sector_exposure_chart, "Share of capital", True),
    ],
    ids=["nav", "drawdown", "exposure", "sector"],
)
def test_the_y_axis_is_titled_and_percent_axes_format_ticks_and_hover(
    build: Callable[[PortfolioView, Palette], go.Figure], y_title: str, percent: bool
) -> None:
    yaxis = build(_view(), LIGHT).layout.yaxis

    assert yaxis.title.text == y_title
    assert yaxis.tickformat == (".0%" if percent else None)
    assert yaxis.hoverformat == (".1%" if percent else None)


def test_a_nan_in_a_series_is_passed_through_to_the_trace() -> None:
    view = _view()
    drawn = dataclasses.replace(
        view, drawdown=((date(2024, 1, 2), 0.0), (date(2024, 1, 3), math.nan))
    )

    [trace] = drawdown_chart(drawn, LIGHT).data

    assert trace.y[0] == 0.0
    assert math.isnan(trace.y[1])
    assert tuple(trace.x) == (date(2024, 1, 2), date(2024, 1, 3))


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


def test_nav_vs_benchmark_draws_a_backtest_view_like_a_portfolio_view() -> None:
    snapshots = make_snapshots(100.0, 120.0, 90.0, 110.0)
    benchmark = make_nav(50.0, 51.0, 49.0, 52.0)

    portfolio = nav_vs_benchmark(portfolio_view(snapshots, benchmark, ()), LIGHT)
    backtest = nav_vs_benchmark(backtest_view((), snapshots, benchmark), LIGHT)

    assert [trace.name for trace in backtest.data] == ["Strategy", "SPY"]
    assert [list(t.y) for t in backtest.data] == [list(t.y) for t in portfolio.data]


def test_a_partial_backtest_benchmark_is_drawn_over_its_dates_rebased_to_the_nav_there() -> None:
    view = backtest_view(
        (),
        make_snapshots(100.0, 120.0, 90.0, 110.0),
        make_nav(60.0, 54.0, start=date(2024, 1, 3)),  # two covered dates: no window return
    )

    strategy, spy = nav_vs_benchmark(view, LIGHT).data

    assert view.benchmark_window == (date(2024, 1, 3), date(2024, 1, 4))
    assert view.benchmark_partial is True
    assert list(strategy.x) == [date(2024, 1, d) for d in (2, 3, 4, 5)]
    assert list(spy.x) == [date(2024, 1, 3), date(2024, 1, 4)]
    assert list(spy.y) == pytest.approx([120.0, 108.0])  # starts at the strategy index there


def test_the_funnel_starts_at_the_shortlist() -> None:
    figure = funnel_chart(Funnel(shortlist=8, researched=6, portfolio=4, orders=3), LIGHT)

    [trace] = figure.data
    assert FUNNEL_STAGES == ("Shortlist", "Researched", "Target positions", "Orders")
    assert trace.type == "funnel"
    assert list(trace.y) == list(FUNNEL_STAGES)
    assert list(trace.x) == [8, 6, 4, 3]
    assert trace.marker.color == LIGHT.strategy
    assert figure.layout.title.text == "Cycle funnel (from the shortlist)"


def test_the_llm_cost_chart_has_budget_and_spend_bars_with_gaps_for_missing_values() -> None:
    costs = (
        CostRow(date(2024, 1, 2), 1.0, 0.12),
        CostRow(date(2024, 1, 3), None, math.nan),
    )

    figure = llm_cost_chart(costs, DARK)

    budget, spent = figure.data
    assert (budget.name, spent.name) == ("Budget", "Spent")
    assert tuple(budget.x) == (date(2024, 1, 2), date(2024, 1, 3))
    assert list(budget.y) == [1.0, None]
    assert spent.y[0] == 0.12
    assert math.isnan(spent.y[1])  # NaN passes through; the chart leaves a gap
    assert (budget.marker.color, spent.marker.color) == (DARK.benchmark, DARK.strategy)
    assert figure.layout.barmode == "group"
    assert figure.layout.yaxis.title.text == "USD"
    assert figure.layout.yaxis.tickprefix == "$"


def test_the_new_charts_of_nothing_build_without_points() -> None:
    assert all(len(trace.x or ()) == 0 for trace in llm_cost_chart((), LIGHT).data)
    assert list(funnel_chart(Funnel(0, 0, 0, 0), LIGHT).data[0].x) == [0, 0, 0, 0]
