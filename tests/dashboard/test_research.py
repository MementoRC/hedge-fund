"""The Research & attribution page: names, verification per horizon, attribution, LLM cost."""

from __future__ import annotations

import dataclasses
import math

from hedge_fund.dashboard.data import RootState
from hedge_fund.dashboard.format import fmt_date
from hedge_fund.dashboard.research import (
    ATTRIBUTION_COLUMNS,
    COST_COLUMNS,
    HORIZON_COLUMNS,
    NAME_COLUMNS,
    attribution_table,
    cost_table,
    gap_message,
    horizons_table,
    names_table,
)
from hedge_fund.domain.research import ResearchResult
from hedge_fund.viz.load import RootKind
from hedge_fund.viz.views import HorizonRow, ResearchView, research_view
from tests.dashboard.builders import (
    LATER,
    make_attribution,
    make_name,
    make_verification,
    rich_state,
)
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
