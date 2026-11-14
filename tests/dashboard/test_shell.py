"""The sidebar every page shows."""

from __future__ import annotations

from pathlib import Path

from hedge_fund.adapters.cycle.json_report import RUNS_DIR
from tests.dashboard.builders import backtest_root, mismatched_root, open_page
from tests.dashboard.scripts import sidebar_only, sidebar_with_state


def _bodies(elements: object) -> list[str]:
    return [element.value for element in elements]  # type: ignore[attr-defined]


def _bad_cycle_files(root: Path, count: int) -> None:
    (root / RUNS_DIR).mkdir(parents=True)
    for index in range(count):
        (root / RUNS_DIR / f"bad{index:02d}.json").write_text("{", encoding="utf-8")


def test_the_default_sidebar_offers_the_demo_root() -> None:
    at = open_page(sidebar_only)

    assert len(at.exception) == 0
    assert at.sidebar.text_input(key="root").value == "data/demo"
    assert len(at.sidebar.radio) == 0
    assert len(at.sidebar.warning) == 0
    assert len(at.sidebar.caption) == 0
    assert not any("DEMO" in body for body in _bodies(at.sidebar.markdown))


def test_the_demo_world_gets_the_demo_badge(demo_root: Path) -> None:
    at = open_page(sidebar_only, demo_root)

    assert len(at.exception) == 0
    assert any("DEMO" in body for body in _bodies(at.sidebar.markdown))
    assert len(at.sidebar.warning) == 0


def test_a_backtest_root_has_no_demo_badge(demo_root: Path, tmp_path: Path) -> None:
    at = open_page(sidebar_only, backtest_root(demo_root, tmp_path))

    assert not any("DEMO" in body for body in _bodies(at.sidebar.markdown))
    assert _bodies(at.sidebar.caption) == ["Backtest root."]


def test_a_marker_for_another_run_gets_no_badge_although_the_loader_read_it(
    demo_root: Path, tmp_path: Path
) -> None:
    at = open_page(sidebar_only, mismatched_root(demo_root, tmp_path))

    assert len(at.exception) == 0
    assert not any("DEMO" in body for body in _bodies(at.sidebar.markdown))
    [caption] = _bodies(at.sidebar.caption)
    assert caption.startswith("Not a demo or backtest root: ")
    assert "other-run" in caption


def test_an_empty_directory_is_named_as_unclassified(tmp_path: Path) -> None:
    at = open_page(sidebar_only, tmp_path)

    [caption] = _bodies(at.sidebar.caption)
    assert caption.startswith("Not a demo or backtest root: ")
    assert len(at.sidebar.warning) == 0


def test_one_unreadable_file_is_a_sidebar_warning_not_an_exception(tmp_path: Path) -> None:
    root = tmp_path / "r"
    _bad_cycle_files(root, 1)

    at = open_page(sidebar_only, root)

    assert len(at.exception) == 0
    [warning] = _bodies(at.sidebar.warning)
    assert warning.startswith("1 unreadable file;")
    assert "runs/bad00.json" in warning


def test_many_unreadable_files_are_counted_and_the_list_is_capped(tmp_path: Path) -> None:
    root = tmp_path / "r"
    _bad_cycle_files(root, 12)

    at = open_page(sidebar_only, root)

    [warning] = _bodies(at.sidebar.warning)
    assert warning.startswith("12 unreadable files;")
    assert "runs/bad09.json" in warning
    assert "runs/bad10.json" not in warning
    assert "- and 2 more" in warning


def test_a_root_that_could_not_be_read_shows_the_reason_as_a_warning() -> None:
    at = open_page(sidebar_with_state, None, "read_error")

    assert len(at.exception) == 0
    [warning] = _bodies(at.sidebar.warning)
    assert "denied" in warning
    assert len(at.sidebar.caption) == 0


def test_an_unclassified_root_with_no_error_prints_no_none() -> None:
    at = open_page(sidebar_with_state, None, "plain")

    assert len(at.exception) == 0
    assert len(at.sidebar.caption) == 0
    assert len(at.sidebar.warning) == 0
    assert not any("None" in body for body in _bodies(at.sidebar.markdown))
