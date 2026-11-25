"""The Cycles page: the cycle list, the funnel from the shortlist, vetoes, approval, LLM cost."""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from hedge_fund.dashboard import cycles
from hedge_fund.dashboard.cycles import CYCLE_COLUMNS, FUNNEL_NOTE, cycle_label, cycle_table
from hedge_fund.dashboard.data import inspect_root
from hedge_fund.viz.format import fmt_date
from hedge_fund.viz.load import Workspace
from hedge_fund.viz.views import cycle_view
from tests.dashboard.builders import (
    LATER,
    busy_root,
    full_cycle,
    live_root,
    make_budget,
    make_spend,
    open_page,
    rich_state,
)
from tests.dashboard.scripts import page_in_state, page_of_builder, page_of_root
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


def _charts(at: AppTest) -> int:
    return len(at.get("plotly_chart"))


def _captions(at: AppTest) -> list[str]:
    return [caption.value for caption in at.main.caption]


def _markdown(at: AppTest) -> str:
    return "\n".join(markdown.value for markdown in at.main.markdown)


def _metrics(at: AppTest) -> dict[str, str]:
    return {metric.label: metric.value for metric in at.metric}


def test_the_funnel_note_says_where_the_funnel_starts() -> None:
    assert "starts at the shortlist" in FUNNEL_NOTE
    assert "no universe or quality-gate counts" in FUNNEL_NOTE
    assert "target at rebalance" in FUNNEL_NOTE


def test_the_demo_cycles_list_every_cycle_and_open_the_newest(
    demo_root: Path, workspace: Workspace
) -> None:
    view = cycle_view(workspace.cycles)

    at = open_page(page_of_root, demo_root, "cycles")

    assert len(at.exception) == 0
    assert len(at.error) == 0
    assert [title.value for title in at.title] == ["Cycles"]
    [table] = at.dataframe
    assert len(table.value) == len(view.rows)
    assert at.selectbox[0].options[0] == cycle_label(view.rows[-1])
    assert _charts(at) == 1
    assert FUNNEL_NOTE in _captions(at)
    assert _metrics(at)["LLM budget"] == "n/a"  # the demo backtest runs without a budget


def test_a_missing_root_names_the_demo_command() -> None:
    at = open_page(page_of_root, None, "cycles")

    assert len(at.exception) == 0
    assert len(at.dataframe) == 0
    assert _charts(at) == 0
    assert "pixi run -e dev demo" in at.info[0].value


def test_a_root_without_cycle_reports_names_the_cycle_command(tmp_path: Path) -> None:
    at = open_page(page_of_root, tmp_path, "cycles")

    assert len(at.exception) == 0
    assert len(at.error) == 0  # an empty directory has a kind_error: that is no error
    [info] = at.info
    assert "holds no cycle reports" in info.value
    assert "`hedge-fund cycle`" in info.value
    assert "`pixi run -e dev demo`" in info.value


def test_a_real_live_root_shows_its_cycles_under_a_neutral_caption(tmp_path: Path) -> None:
    at = open_page(page_of_root, live_root(tmp_path), "cycles")

    assert len(at.exception) == 0
    assert [caption.value for caption in at.sidebar.caption] == ["Live root"]
    assert len(at.sidebar.warning) == 0
    assert len(at.error) == 0
    [table] = at.dataframe
    assert len(table.value) == 1
    assert r"- gross\: gross 1\.70 \> 1\.60" in _markdown(at)


@pytest.mark.parametrize(
    ("case", "caption"),
    [("live", "Live root"), ("refused", "Not a demo or backtest root: ")],
)
def test_a_classify_root_refusal_still_shows_the_cycles(case: str, caption: str) -> None:
    at = open_page(page_in_state, None, "cycles", case)

    assert len(at.exception) == 0
    assert len(at.error) == 0
    [shown] = [c.value for c in at.sidebar.caption]
    assert shown.startswith(caption)
    [table] = at.dataframe
    assert len(table.value) == 2
    assert _charts(at) == 1


def test_a_root_that_could_not_be_read_fails_closed_on_the_cycles() -> None:
    at = open_page(page_in_state, None, "cycles", "read_error")

    assert len(at.exception) == 0
    assert "denied" in at.error[0].value
    assert len(at.dataframe) == 0
    assert _charts(at) == 0
    assert len(at.metric) == 0


def test_the_newest_cycle_opens_first_and_another_can_be_picked() -> None:
    at = open_page(page_of_builder, None, "cycles", "rich_state")

    assert at.selectbox[0].options == [
        f"{fmt_date(LATER)} (dry-run)",
        f"{fmt_date(DAY0)} (blocked)",
    ]
    assert _metrics(at) == {
        "Outcome": "dry-run",
        "Approval": "n/a",
        "LLM budget": "n/a",
        "LLM spent": "n/a",
    }
    assert "No risk veto." in _captions(at)

    at.selectbox[0].select_index(1).run()

    assert len(at.exception) == 0
    assert [subheader.value for subheader in at.subheader] == [f"Cycle of {fmt_date(DAY0)}"]
    assert _metrics(at) == {
        "Outcome": "blocked",
        "Approval": "blocked",
        "LLM budget": "$1.00",
        "LLM spent": "$0.12",
    }
    assert r"- gross\: gross 1\.70 \> 1\.60" in _markdown(at)


def test_a_halted_cycle_shows_the_halt_the_error_and_the_unresolved_tickers() -> None:
    at = open_page(page_of_builder, None, "cycles", "trouble_state")

    assert len(at.exception) == 0
    assert len(at.error) == 0  # the cycle's own error is report text, not a page error
    markdown = _markdown(at)
    assert "**Halted** by a risk breaker." in markdown
    assert r"**Cycle error:** RuntimeError\: boom" in markdown
    assert "Unresolved tickers (no price or fundamentals): 2." in _captions(at)


def test_veto_and_error_text_is_sent_escaped() -> None:
    at = open_page(page_of_builder, None, "cycles", "escaped_text_state")

    assert len(at.exception) == 0
    markdown = _markdown(at)
    assert r"- gross\: \$5 limit" in markdown
    assert r"**Cycle error:** RuntimeError\: \`boom\`" in markdown


def test_switching_to_a_root_with_fewer_cycles_opens_its_newest_cycle(tmp_path: Path) -> None:
    busy, live = busy_root(tmp_path), live_root(tmp_path)
    at = open_page(page_of_root, busy, "cycles")
    at.selectbox[0].select_index(0).run()  # the LATER cycle: row index 1 of two
    assert [s.value for s in at.subheader] == [f"Cycle of {fmt_date(LATER)}"]

    at.sidebar.text_input(key="root").set_value(str(live)).run()  # one cycle only

    assert len(at.exception) == 0
    assert inspect_root(live).kind_error is not None  # a live root, refused as normal
    [table] = at.dataframe
    assert len(table.value) == 1
    assert at.selectbox[0].options == [f"{fmt_date(DAY0)} (blocked)"]
    assert [s.value for s in at.subheader] == [f"Cycle of {fmt_date(DAY0)}"]


@pytest.mark.parametrize("picked", [5, -2, -1, None])
def test_an_out_of_range_or_missing_selection_falls_back_to_the_newest_cycle(
    monkeypatch: pytest.MonkeyPatch, picked: int | None
) -> None:
    workspace = rich_state().workspace
    assert workspace is not None
    view = cycle_view(workspace.cycles)
    monkeypatch.setattr(cycles.st, "selectbox", lambda *args, **kwargs: picked)

    assert cycles._selected(view, Path("r")) == view.rows[-1]
