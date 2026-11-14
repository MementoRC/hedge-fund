"""The entry point: ``app.py`` opens the Overview; the page wrappers render their pages."""

from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

from tests.dashboard.builders import open_page
from tests.dashboard.scripts import portfolio_page_with_sidebar

APP = Path(__file__).resolve().parents[2] / "app.py"


def test_the_app_opens_on_the_overview_with_the_demo_command_when_there_is_no_demo() -> None:
    at = AppTest.from_file(APP, default_timeout=60).run()

    assert len(at.exception) == 0
    assert [title.value for title in at.title] == ["Overview"]
    assert "pixi run -e dev demo" in at.info[0].value
    assert at.sidebar.text_input(key="root").value == "data/demo"


def test_the_app_shows_the_demo_overview_after_the_root_is_picked(demo_root: Path) -> None:
    at = AppTest.from_file(APP, default_timeout=60).run()

    at.sidebar.text_input(key="root").set_value(str(demo_root)).run()

    assert len(at.exception) == 0
    assert "NAV" in [metric.label for metric in at.metric]
    assert any("DEMO" in markdown.value for markdown in at.sidebar.markdown)


def test_the_portfolio_page_wrapper_renders_the_portfolio(demo_root: Path) -> None:
    at = open_page(portfolio_page_with_sidebar, demo_root)

    assert len(at.exception) == 0
    assert [title.value for title in at.title] == ["Portfolio & NAV"]
    assert len(at.get("plotly_chart")) == 4
