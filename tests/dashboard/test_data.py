"""Root inspection, the file fingerprint and the cached loader behind every page."""

from __future__ import annotations

from pathlib import Path

import pytest

from hedge_fund.dashboard import data
from hedge_fund.dashboard.data import (
    DEFAULT_ROOT,
    RootState,
    inspect_root,
    load_state,
    store_fingerprint,
)
from hedge_fund.viz.load import RootKind
from tests.dashboard.builders import backtest_root, mismatched_root


def test_the_default_root_is_the_demo_directory() -> None:
    assert DEFAULT_ROOT == "data/demo"


def test_a_missing_directory_has_no_workspace(tmp_path: Path) -> None:
    root = tmp_path / "absent"

    assert inspect_root(root) == RootState(root, None, None, None)


def test_the_demo_world_is_classified_as_a_demo(demo_root: Path) -> None:
    state = inspect_root(demo_root)

    assert state.kind is RootKind.DEMO
    assert state.kind_error is None
    assert state.workspace is not None
    assert state.workspace.unreadable == ()
    assert len(state.workspace.snapshots) > 3


def test_a_backtest_root_is_classified_as_a_backtest(demo_root: Path, tmp_path: Path) -> None:
    state = inspect_root(backtest_root(demo_root, tmp_path))

    assert state.kind is RootKind.BACKTEST
    assert state.kind_error is None


def test_an_empty_directory_exists_but_is_not_classified(tmp_path: Path) -> None:
    state = inspect_root(tmp_path)

    assert state.kind is None
    assert state.kind_error is not None
    assert "expected exactly one readable backtest summary" in state.kind_error
    assert state.workspace is not None
    assert state.workspace.snapshots == ()


def test_a_marker_for_another_run_is_no_demo_even_though_the_workspace_read_it(
    demo_root: Path, tmp_path: Path
) -> None:
    state = inspect_root(mismatched_root(demo_root, tmp_path))

    assert state.workspace is not None
    assert state.workspace.demo is not None  # the loader parsed the marker ...
    assert state.kind is None  # ... but only classify_root decides demo vs real
    assert state.kind_error is not None
    assert "other-run" in state.kind_error


def test_the_fingerprint_lists_regular_files_by_relative_name(tmp_path: Path) -> None:
    (tmp_path / "sub").mkdir()
    (tmp_path / "b.json").write_text("{}", encoding="utf-8")
    (tmp_path / "sub" / "a.json").write_text("[]", encoding="utf-8")
    (tmp_path / "dangling").symlink_to(tmp_path / "nowhere")

    names = [name for name, _, _ in store_fingerprint(tmp_path)]

    assert names == ["b.json", "sub/a.json"]


def test_the_fingerprint_changes_when_a_file_changes(tmp_path: Path) -> None:
    (tmp_path / "a.json").write_text("{}", encoding="utf-8")
    before = store_fingerprint(tmp_path)

    (tmp_path / "a.json").write_text('{"changed": true}', encoding="utf-8")

    assert store_fingerprint(tmp_path) != before


def test_the_fingerprint_of_a_missing_directory_is_empty(tmp_path: Path) -> None:
    assert store_fingerprint(tmp_path / "absent") == ()


def test_load_state_reads_once_until_a_file_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[Path] = []
    real = data.inspect_root

    def spy(root: Path) -> RootState:
        calls.append(root)
        return real(root)

    monkeypatch.setattr(data, "inspect_root", spy)
    root = tmp_path / "r"
    root.mkdir()

    load_state(str(root))
    load_state(str(root))
    assert len(calls) == 1

    (root / "new.json").write_text("{}", encoding="utf-8")
    load_state(str(root))
    assert len(calls) == 2


def test_a_blank_root_falls_back_to_the_default() -> None:
    state = load_state("   ")

    assert state.root == Path(DEFAULT_ROOT)
    assert state.workspace is None
