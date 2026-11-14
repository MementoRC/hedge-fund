"""The Cycles page: the cycle list, the funnel from the shortlist, vetoes, approval, LLM cost."""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path

from hedge_fund.dashboard.cycles import CYCLE_COLUMNS, cycle_label, cycle_table
from hedge_fund.dashboard.data import inspect_root
from hedge_fund.dashboard.format import fmt_date
from hedge_fund.viz.views import cycle_view
from tests.dashboard.builders import (
    LATER,
    full_cycle,
    live_root,
    make_budget,
    make_spend,
    rich_state,
)
from tests.viz.builders import DAY0, make_cycle


def test_the_live_root_builder_writes_a_cycle_that_decodes(tmp_path: Path) -> None:
    """Tasks 4, 6 and 8 rely on this real root: its cycle must round-trip through JSON."""
    state = inspect_root(live_root(tmp_path))

    assert state.workspace is not None
    assert state.workspace.unreadable == ()
    assert state.is_live is True
    [report] = state.workspace.cycles
    assert cycle_view((report,)).rows == cycle_view((full_cycle(DAY0),)).rows


def test_the_columns_start_the_funnel_at_the_shortlist() -> None:
    assert CYCLE_COLUMNS == [
        "Date",
        "Outcome",
        "Dry run",
        "Shortlist",
        "Researched",
        "Target positions",
        "Orders",
        "Risk vetoes",
        "Approval",
        "LLM budget",
        "LLM spent",
    ]


def test_the_cycle_table_lists_every_cycle_with_its_funnel_and_llm_cost() -> None:
    workspace = rich_state().workspace
    assert workspace is not None

    table = cycle_table(cycle_view(workspace.cycles))

    assert list(table.columns) == CYCLE_COLUMNS
    assert table.values.tolist() == [
        [fmt_date(DAY0), "blocked", "no", 2, 1, 1, 1, 1, "blocked", "$1.00", "$0.12"],
        [fmt_date(LATER), "dry-run", "yes", 0, 0, 0, 0, 0, "n/a", "n/a", "n/a"],
    ]


def test_non_finite_llm_figures_read_n_a() -> None:
    report = dataclasses.replace(
        full_cycle(), budget=make_budget(math.inf), spend=make_spend(math.nan)
    )

    [row] = cycle_view((report,)).rows

    assert math.isnan(row.spent)  # type: ignore[arg-type]
    clean = cycle_view((dataclasses.replace(report, budget=None, spend=None),)).rows[0]
    assert dataclasses.replace(row, budget=None, spent=None) == clean
    table = cycle_table(cycle_view((report,)))
    assert (table.iloc[0]["LLM budget"], table.iloc[0]["LLM spent"]) == ("n/a", "n/a")


def test_an_empty_cycle_table_keeps_its_columns() -> None:
    table = cycle_table(cycle_view(()))

    assert list(table.columns) == CYCLE_COLUMNS
    assert len(table) == 0


def test_the_cycle_label_names_the_date_and_the_outcome() -> None:
    [row] = cycle_view((make_cycle(DAY0),)).rows

    assert cycle_label(row) == f"{fmt_date(DAY0)} (submitted)"
