"""The Research & attribution page: names, verification per horizon, attribution, LLM cost."""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from hedge_fund.dashboard.data import RootState
from hedge_fund.dashboard.research import (
    ATTRIBUTION_COLUMNS,
    ATTRIBUTION_NOTE,
    COST_COLUMNS,
    HORIZON_COLUMNS,
    NAME_COLUMNS,
    NO_COSTS,
    NO_THESIS,
    attribution_table,
    cost_table,
    gap_message,
    horizons_table,
    names_table,
)
from hedge_fund.domain.research import ResearchResult
from hedge_fund.viz.format import escape_markdown, fmt_date
from hedge_fund.viz.load import RootKind, Workspace
from hedge_fund.viz.views import HorizonRow, ResearchView, research_view
from tests.dashboard.builders import (
    LATER,
    backtest_root,
    live_root,
    make_attribution,
    make_name,
    make_verification,
    open_page,
    rich_state,
)
from tests.dashboard.scripts import page_in_state, page_of_builder, page_of_root
from tests.viz.builders import DAY0


def _research(state: RootState) -> ResearchView:
    workspace = state.workspace
    assert workspace is not None
    return research_view(
        workspace.research, workspace.cycles, workspace.verifications, workspace.attributions
    )


def test_the_names_table_shows_side_verdict_rating_dispersion_and_agreement() -> None:
    table = names_table(_research(rich_state()))

    assert (
        list(table.columns)
        == NAME_COLUMNS
        == [
            "Ticker",
            "Sector",
            "Side",
            "Verdict",
            "Mean rating",
            "Dispersion",
            "Agreement",
            "Valid runs",
        ]
    )
    assert table.values.tolist() == [
        ["AAA", "Tech", "long", "confirm", "2.00", "0.00", "1.00", 1],
        ["BBB", "Tech", "long", "insufficient", "n/a", "n/a", "n/a", 0],
    ]


def test_a_nan_rating_reads_n_a() -> None:
    view = research_view((ResearchResult(DAY0, (make_name(mean=math.nan),)),), (), (), ())

    assert math.isnan(view.names[0].mean)  # type: ignore[arg-type]
    assert names_table(view).iloc[0]["Mean rating"] == "n/a"


def test_the_horizon_table_shows_hit_rate_and_ic() -> None:
    table = horizons_table(_research(rich_state()))

    assert (
        list(table.columns)
        == HORIZON_COLUMNS
        == [
            "Horizon",
            "Scored",
            "Hit rate",
            "IC (rank correlation)",
        ]
    )
    assert table.values.tolist() == [[5, 5, "50.0%", "0.20"]]


def test_a_missing_hit_rate_and_a_nan_ic_read_n_a() -> None:
    view = research_view((), (), (make_verification(hit_rate=None, ic=math.nan),), ())

    [row] = view.horizons
    assert row.hit_rate is None
    assert math.isnan(row.ic)  # type: ignore[arg-type]
    assert dataclasses.replace(row, ic=None) == HorizonRow(5, 5, None, None)
    assert horizons_table(view).values.tolist() == [[5, 5, "n/a", "n/a"]]


def test_an_infinite_ic_reads_n_a() -> None:
    view = research_view((), (), (make_verification(ic=math.inf),), ())

    assert horizons_table(view).iloc[0]["IC (rank correlation)"] == "n/a"


def test_attribution_tables_split_pnl_into_its_components() -> None:
    view = _research(rich_state())
    assert view.attribution is not None

    by_name = attribution_table(view.attribution.by_name, "Ticker")
    by_sector = attribution_table(view.attribution.by_sector, "Sector name")

    assert ATTRIBUTION_COLUMNS == ["Market", "Sector", "Residual", "Trading", "Total"]
    assert list(by_name.columns) == ["Ticker", *ATTRIBUTION_COLUMNS]
    assert by_name.values.tolist() == [["AAA", "$7.00", "$1.00", "-$0.50", "-$0.25", "$7.25"]]
    assert list(by_sector.columns) == ["Sector name", *ATTRIBUTION_COLUMNS]
    assert by_sector.values.tolist() == [["Tech", "$7.00", "$1.00", "-$0.50", "-$0.25", "$7.25"]]


def test_a_non_finite_attribution_reads_n_a() -> None:
    view = research_view((), (), (), (make_attribution(market=math.nan),))
    assert view.attribution is not None

    row = attribution_table({"All names": view.attribution.total}, "Book").iloc[0]

    assert row["Book"] == "All names"
    assert row["Total"] == "n/a"  # market NaN makes the sum NaN
    assert row["Market"] == "n/a"
    assert row["Sector"] == "$1.00"


def test_the_cost_table_lists_budget_and_spend_per_cycle() -> None:
    table = cost_table(_research(rich_state()))

    assert list(table.columns) == COST_COLUMNS == ["Cycle", "LLM budget", "LLM spent"]
    assert table.values.tolist() == [
        [fmt_date(DAY0), "$1.00", "$0.12"],
        [fmt_date(LATER), "n/a", "n/a"],
    ]


def test_empty_research_tables_keep_their_columns() -> None:
    view = research_view((), (), (), ())

    assert list(names_table(view).columns) == NAME_COLUMNS
    assert list(horizons_table(view).columns) == HORIZON_COLUMNS
    assert list(cost_table(view).columns) == COST_COLUMNS
    assert len(names_table(view)) == len(horizons_table(view)) == len(cost_table(view)) == 0
    assert list(attribution_table({}, "Ticker").columns) == ["Ticker", *ATTRIBUTION_COLUMNS]


def test_a_gap_on_a_demo_or_backtest_root_explains_why_it_cannot_be_filled() -> None:
    for kind in (RootKind.DEMO, RootKind.BACKTEST):
        message = gap_message(kind, "attribution", "hedge-fund attribute")

        assert f"this is a {kind.value} root" in message
        assert "keeps its fills in memory" in message
        assert "cannot run on it" in message


def test_a_gap_on_any_other_root_names_the_command() -> None:
    message = gap_message(None, "thesis verification", "hedge-fund verify")

    assert "No completed thesis verification in this root yet" in message
    assert "`hedge-fund verify`" in message
    assert "pixi run -e dev demo" not in message  # the demo has no verification (Decision 9)


def _charts(at: AppTest) -> int:
    return len(at.get("plotly_chart"))


def _infos(at: AppTest) -> list[str]:
    return [info.value for info in at.info]


def test_the_demo_research_shows_names_and_explains_the_backtest_gaps(
    demo_root: Path, workspace: Workspace
) -> None:
    view = research_view(
        workspace.research, workspace.cycles, workspace.verifications, workspace.attributions
    )

    at = open_page(page_of_root, demo_root, "research")

    assert len(at.exception) == 0
    assert len(at.error) == 0
    assert [title.value for title in at.title] == ["Research & attribution"]
    names, costs = at.dataframe
    assert len(names.value) == len(view.names)
    assert len(costs.value) == len(view.costs)
    assert len(at.expander) == len(view.names)
    infos = _infos(at)
    assert len(infos) == 2  # verification and attribution
    assert all("this is a demo root" in info and "cannot run on it" in info for info in infos)
    assert _charts(at) == 1


def test_a_backtest_root_names_itself_in_the_gaps(demo_root: Path, tmp_path: Path) -> None:
    at = open_page(page_of_root, backtest_root(demo_root, tmp_path), "research")

    assert len(at.exception) == 0
    assert all("this is a backtest root" in info for info in _infos(at))


def test_a_root_with_everything_shows_every_section() -> None:
    at = open_page(page_of_builder, None, "research", "rich_state")

    assert len(at.exception) == 0
    assert len(at.info) == 0
    assert len(at.dataframe) == 6  # names, horizons, by name, by sector, book, cost
    assert at.dataframe[1].value.iloc[0]["IC (rank correlation)"] == "0.20"
    assert len(at.expander) == 2
    markdown = [m.value for m in at.main.markdown]
    assert r"Buy\: wide moat" in markdown
    assert NO_THESIS in markdown  # BBB has no valid run
    assert ATTRIBUTION_NOTE in [c.value for c in at.main.caption]
    assert _charts(at) == 1


def test_an_empty_root_explains_every_section() -> None:
    at = open_page(page_of_builder, None, "research", "make_state")

    assert len(at.exception) == 0
    names, verification, attribution = _infos(at)
    assert "holds no research" in names
    assert "`pixi run -e dev demo`" in names
    assert "`hedge-fund verify`" in verification
    assert "`hedge-fund attribute`" in attribution
    assert len(at.dataframe) == 0
    assert _charts(at) == 0
    assert NO_COSTS in [c.value for c in at.main.caption]
    assert "No cycle reports, so there is no LLM cost to show." in NO_COSTS
    assert "`pixi run -e dev demo`" in NO_COSTS


def test_a_thesis_with_markdown_characters_is_shown_literally() -> None:
    at = open_page(page_of_builder, None, "research", "markdown_thesis_state")

    assert len(at.exception) == 0
    markdown = [m.value for m in at.main.markdown]
    assert escape_markdown("$5 *bold*") in markdown  # sent escaped: renders "$5 *bold*"
    assert "$5 *bold*" not in markdown


def test_the_thesis_label_shows_the_ticker_literally() -> None:
    at = open_page(page_of_builder, None, "research", "markdown_ticker_state")

    assert len(at.exception) == 0
    assert [expander.label for expander in at.expander] == [r"Thesis: T\`boom\` costs \$5"]


def test_a_missing_root_names_the_demo_command() -> None:
    at = open_page(page_of_root, None, "research")

    assert len(at.exception) == 0
    [info] = _infos(at)
    assert "pixi run -e dev demo" in info
    assert len(at.dataframe) == 0


def test_a_real_live_root_names_the_commands_under_a_neutral_caption(tmp_path: Path) -> None:
    at = open_page(page_of_root, live_root(tmp_path), "research")

    assert len(at.exception) == 0
    assert len(at.error) == 0
    assert [caption.value for caption in at.sidebar.caption] == ["Live root"]
    _, verification, attribution = _infos(at)
    assert "`hedge-fund verify`" in verification
    assert "`hedge-fund attribute`" in attribution
    [costs] = at.dataframe
    assert len(costs.value) == 1


@pytest.mark.parametrize(
    ("case", "caption"),
    [("live", "Live root"), ("refused", "Not a demo or backtest root: ")],
)
def test_a_classify_root_refusal_still_shows_the_research(case: str, caption: str) -> None:
    at = open_page(page_in_state, None, "research", case)

    assert len(at.exception) == 0
    assert len(at.error) == 0
    [shown] = [c.value for c in at.sidebar.caption]
    assert shown.startswith(caption)
    assert len(at.dataframe) == 6
    assert _charts(at) == 1


def test_a_root_that_could_not_be_read_fails_closed_on_the_research() -> None:
    at = open_page(page_in_state, None, "research", "read_error")

    assert len(at.exception) == 0
    assert "denied" in at.error[0].value
    assert len(at.dataframe) == 0
    assert len(at.expander) == 0
    assert _charts(at) == 0
