"""Portfolio page: Overview headlines, the return note, NAV/drawdown/exposure/sector charts."""

from __future__ import annotations

from datetime import timedelta

from plotly.offline import get_plotlyjs

from hedge_fund.sitegen.pages import (
    DEMO_BANNER,
    NO_EXPOSURE,
    NO_NAV,
    NO_SECTORS,
    SITE_NO_SPY,
    portfolio_page,
)
from hedge_fund.sitegen.palette import DARK_PALETTE, LIGHT_PALETTE
from hedge_fund.viz.views import overview_view, portfolio_view
from tests.dashboard.builders import NAV_VALUES
from tests.sitegen.helpers import DEMO, PLAIN, chart_ids, figures, scripts
from tests.viz.builders import (
    DAY0,
    make_cycle,
    make_nav,
    make_position,
    make_snapshots,
    make_target,
)

SNAPSHOTS = make_snapshots(*NAV_VALUES)
FULL_IDS = ["portfolio-nav", "portfolio-drawdown", "portfolio-exposure", "portfolio-sectors"]


def _page(*closes: float, sector: str = "Tech", demo: bool = False) -> str:
    benchmark = make_nav(*closes, start=DAY0 + timedelta(days=1)) if closes else None
    target = make_target(
        DAY0, make_position("AAA", sector, 0.3), make_position("BBB", "Energy", -0.1)
    )
    cycles = (make_cycle(DAY0, target=target),)
    return portfolio_page(
        overview_view(SNAPSHOTS, benchmark, cycles),
        portfolio_view(SNAPSHOTS, benchmark, cycles),
        DEMO if demo else PLAIN,
    )


def test_the_page_draws_four_charts_with_fixed_ids() -> None:
    page = _page()

    assert page.startswith("# Portfolio\n")
    assert chart_ids(page) == FULL_IDS


def test_every_chart_has_a_light_and_a_dark_json_block() -> None:
    shown = figures(_page())

    for chart_id in FULL_IDS:
        assert shown[f"{chart_id}-light"]["layout"]["paper_bgcolor"] == LIGHT_PALETTE.background
        assert shown[f"{chart_id}-dark"]["layout"]["paper_bgcolor"] == DARK_PALETTE.background


def test_the_headlines_are_a_formatted_table() -> None:
    page = _page()

    assert "| Figure | Value | Note |" in page
    assert r"| NAV | $115\.00 | As of 2024\-01\-05 |" in page
    assert r"| Max drawdown | 25\.0% | Full history |" in page


def test_the_return_note_follows_the_headlines() -> None:
    assert SITE_NO_SPY in _page()
    with_spy = _page(50.0, 51.0, 52.0)
    assert "Returns marked (SPY window) are over that span" in with_spy
    assert "SPY covers only 2024-01-03 to 2024-01-05 (partial window)" in with_spy


def test_the_demo_banner_shows_only_on_a_demo_root() -> None:
    assert _page(demo=True).split("\n\n")[1] == DEMO_BANNER
    assert DEMO_BANNER not in _page()


def test_an_empty_root_shows_n_a_and_notes_instead_of_charts() -> None:
    page = portfolio_page(overview_view((), None, ()), portfolio_view((), None, ()), PLAIN)

    assert "| NAV | n/a | As of n/a |" in page
    assert page.count(NO_NAV) == 2  # NAV and drawdown
    assert NO_EXPOSURE in page
    assert NO_SECTORS in page
    assert chart_ids(page) == []


def test_a_target_without_positions_has_exposure_but_no_sector_chart() -> None:
    cycles = (make_cycle(DAY0, target=make_target(DAY0)),)
    page = portfolio_page(
        overview_view(SNAPSHOTS, None, cycles), portfolio_view(SNAPSHOTS, None, cycles), PLAIN
    )

    assert chart_ids(page) == FULL_IDS[:3]
    assert NO_SECTORS in page


def test_json_from_store_text_has_no_raw_lt_or_amp() -> None:
    page = _page(sector="A&B</script><b>")

    assert all("<" not in text and "&" not in text for text in scripts(page).values())
    assert "A&B</script><b>" in figures(page)["portfolio-sectors-light"]["data"][0]["x"]


def test_no_inline_plotly_call_and_no_embedded_plotly_js() -> None:
    page = _page()

    assert "Plotly" not in page
    assert page.count("<script") == page.count('<script type="application/json"')
    assert get_plotlyjs()[:300] not in page
