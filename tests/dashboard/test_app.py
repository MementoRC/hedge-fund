"""The entry point: ``app.py`` opens the Overview; the page wrappers render their pages."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from streamlit.testing.v1 import AppTest

from hedge_fund.dashboard import shell
from tests.dashboard.builders import open_page
from tests.dashboard.scripts import page_wrapper, portfolio_page_with_sidebar

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


@pytest.mark.parametrize(
    ("name", "title"),
    [
        ("cycles", "Cycles"),
        ("research", "Research & attribution"),
        ("backtests", "Backtests"),
    ],
)
def test_the_d2_page_wrappers_render_their_pages(demo_root: Path, name: str, title: str) -> None:
    at = open_page(page_wrapper, demo_root, name)

    assert len(at.exception) == 0
    assert len(at.error) == 0
    assert [t.value for t in at.title] == [title]
    assert any("DEMO" in markdown.value for markdown in at.sidebar.markdown)


def test_main_navigates_the_five_pages_in_order(monkeypatch: pytest.MonkeyPatch) -> None:
    pages: list[dict[str, Any]] = []
    runs: list[str] = []

    class StubPage:
        def run(self) -> None:
            runs.append("run")

    def fake_page(func: Any, **kwargs: Any) -> dict[str, Any]:
        page = {"func": func, **kwargs}
        pages.append(page)
        return page

    def fake_navigation(given: list[Any]) -> StubPage:
        assert given == pages
        return StubPage()

    monkeypatch.setattr(shell.st, "set_page_config", lambda **kw: None)
    monkeypatch.setattr(shell.st, "Page", fake_page)
    monkeypatch.setattr(shell.st, "navigation", fake_navigation)
    monkeypatch.setattr(shell, "render_sidebar", lambda: runs.append("sidebar"))

    shell.main()

    assert [(p["title"], p["url_path"], p.get("default", False)) for p in pages] == [
        ("Overview", "overview", True),
        ("Portfolio & NAV", "portfolio", False),
        ("Cycles", "cycles", False),
        ("Research & attribution", "research", False),
        ("Backtests", "backtests", False),
    ]
    assert [p["func"] for p in pages] == [
        shell.overview_page,
        shell.portfolio_page,
        shell.cycles_page,
        shell.research_page,
        shell.backtests_page,
    ]
    assert runs == ["sidebar", "run"]
