"""Cycles page: the table of all cycles, the newest cycle's funnel, LLM cost across cycles."""

from __future__ import annotations

from hedge_fund.domain.cycle import CycleReport
from hedge_fund.sitegen.pages import (
    CYCLE_HEADER,
    FUNNEL_NOTE,
    NO_COSTS,
    NO_CYCLES,
    NO_FUNNEL,
    cycles_page,
)
from hedge_fund.viz.views import CostRow, cycle_view, research_view
from tests.dashboard.builders import LATER, full_cycle
from tests.sitegen.helpers import PLAIN, chart_ids, figures, numbers
from tests.viz.builders import DAY0, make_cycle


def _page(*cycles: CycleReport) -> str:
    return cycles_page(cycle_view(cycles), research_view((), cycles, (), ()).costs, PLAIN)


def _shortlist(page: str) -> float:
    return numbers(figures(page)["cycles-funnel-light"]["data"][0]["x"])[0]


def test_the_table_lists_every_cycle_in_load_order() -> None:
    page = _page(full_cycle(DAY0), make_cycle(LATER))

    assert "| " + " | ".join(CYCLE_HEADER) + " |" in page
    assert r"| 2024\-01\-02 | blocked | 2 | 1 | 1 | 1 | 1 | blocked | $1\.00 | $0\.12 |" in page
    assert r"| 2024\-01\-04 | submitted | 0 | 0 | 0 | 0 | 0 | n/a | n/a | n/a |" in page
    assert page.index("2024\\-01\\-02 |") < page.index("2024\\-01\\-04 |")


def test_the_funnel_is_the_newest_cycle_by_date() -> None:
    page = _page(full_cycle(LATER), make_cycle(DAY0))

    assert chart_ids(page) == ["cycles-funnel", "cycles-llm-cost"]
    assert "## Funnel of the newest cycle (2024-01-04)" in page
    assert _shortlist(page) == 2


def test_a_tie_on_the_date_picks_the_later_cycle_in_load_order() -> None:
    later_full = _page(make_cycle(DAY0), full_cycle(DAY0))
    later_empty = _page(full_cycle(DAY0), make_cycle(DAY0))

    assert _shortlist(later_full) == 2
    assert "cycles-funnel" not in chart_ids(later_empty)
    assert NO_FUNNEL in later_empty


def test_an_all_zero_newest_funnel_gives_the_note_not_a_chart() -> None:
    page = _page(full_cycle(DAY0), make_cycle(LATER))  # the newest has all four counts at 0

    assert NO_FUNNEL == (
        "The newest cycle stopped before the shortlist, so there is no funnel to chart."
    )
    assert NO_FUNNEL in page
    assert FUNNEL_NOTE not in page
    assert chart_ids(page) == ["cycles-llm-cost"]


def test_no_cycles_gives_the_note_and_no_chart() -> None:
    page = _page()

    assert NO_CYCLES == "No cycles in this root."
    assert NO_CYCLES in page
    assert chart_ids(page) == []


def test_no_costs_gives_the_note_instead_of_the_cost_chart() -> None:
    page = cycles_page(cycle_view((full_cycle(DAY0),)), (), PLAIN)

    assert chart_ids(page) == ["cycles-funnel"]
    assert NO_COSTS in page


def test_cost_rows_without_any_budget_or_spend_give_the_note_not_an_empty_chart() -> None:
    costs = (CostRow(DAY0, None, None), CostRow(LATER, None, None))  # e.g. the demo root
    page = cycles_page(cycle_view((full_cycle(DAY0), full_cycle(LATER))), costs, PLAIN)

    assert chart_ids(page) == ["cycles-funnel"]
    assert NO_COSTS in page


def test_one_recorded_spend_is_enough_to_draw_the_cost_chart() -> None:
    costs = (CostRow(DAY0, None, None), CostRow(LATER, None, 0.5))
    page = cycles_page(cycle_view((full_cycle(DAY0), full_cycle(LATER))), costs, PLAIN)

    assert chart_ids(page) == ["cycles-funnel", "cycles-llm-cost"]


def test_the_cost_chart_carries_budget_and_spend_rounded() -> None:
    costs = (CostRow(DAY0, 1.0, 1 / 3),)
    page = cycles_page(cycle_view((full_cycle(DAY0),)), costs, PLAIN)

    spent = figures(page)["cycles-llm-cost-light"]["data"][1]["y"]
    assert numbers(spent) == [0.333333]
