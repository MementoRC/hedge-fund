"""Loading a store root into domain objects, and classifying roots fail-closed."""

from __future__ import annotations

import json
import shutil
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from hedge_fund.adapters.account import JsonSnapshotStore
from hedge_fund.adapters.data.frozen_store import write_store
from hedge_fund.adapters.demo_marker import DEMO_MARKER, demo_store_dir
from hedge_fund.domain.account import AccountSnapshot
from hedge_fund.engine.account.history import nav_history
from hedge_fund.engine.backtest.metrics import performance
from hedge_fund.viz.load import RootError, RootKind, Workspace, classify_root, load_workspace
from tests.factories import make_frozen_data


def _copy(demo_root: Path, tmp_path: Path, name: str = "copy") -> Path:
    root = tmp_path / name
    shutil.copytree(demo_root, root)
    return root


def test_the_demo_root_loads_every_kind(workspace: Workspace) -> None:
    assert workspace.demo is not None
    assert workspace.demo.seed == 1
    assert len(workspace.cycles) == 2
    assert len(workspace.backtests) == 1
    assert workspace.research
    assert workspace.snapshots
    assert workspace.settlements == workspace.attributions == workspace.verifications == ()
    assert workspace.unreadable == ()


def test_the_demo_benchmark_is_spy_on_the_nav_dates(workspace: Workspace) -> None:
    nav = nav_history(workspace.snapshots, as_of=max(s.as_of for s in workspace.snapshots))
    assert workspace.benchmark is not None
    assert [p.date for p in workspace.benchmark] == [p.date for p in nav]
    stored = workspace.backtests[0].metrics.benchmark.total_return
    assert performance(workspace.benchmark).total_return == pytest.approx(stored)


def test_a_backtest_root_has_no_benchmark_unless_given_its_store(
    demo_root: Path, tmp_path: Path
) -> None:
    root = _copy(demo_root, tmp_path)
    (root / DEMO_MARKER).unlink()

    assert load_workspace(root).benchmark is None
    given = load_workspace(root, store=demo_store_dir(demo_root))
    assert given.benchmark is not None
    assert given.unreadable == ()


def test_a_store_that_is_not_the_runs_store_gives_no_benchmark(
    demo_root: Path, tmp_path: Path
) -> None:
    root = _copy(demo_root, tmp_path)
    (root / DEMO_MARKER).unlink()
    other = tmp_path / "other.store"
    write_store(other, make_frozen_data(), created_at=datetime(2024, 6, 1, tzinfo=UTC))

    ws = load_workspace(root, store=other)

    assert ws.benchmark is None
    assert len(ws.unreadable) == 1
    assert ws.unreadable[0].startswith("store other.store: ValueError: manifest digest ")


def test_an_empty_root_loads_empty_without_a_warning(tmp_path: Path) -> None:
    ws = load_workspace(tmp_path)

    assert ws.demo is None
    assert ws.snapshots == ws.cycles == ws.research == ws.backtests == ()
    assert ws.benchmark is None
    assert ws.unreadable == ()


def test_unreadable_files_become_warnings_never_exceptions(tmp_path: Path) -> None:
    for directory in ("runs", "settlements", "attributions", "verifications", "backtest"):
        (tmp_path / directory).mkdir()
        (tmp_path / directory / "bad.json").write_text("{", encoding="utf-8")
    (tmp_path / "account").mkdir()
    (tmp_path / "account" / "2024-01-02_aaaaaaaaaaaa.json").write_text("{", encoding="utf-8")
    (tmp_path / "research").mkdir()
    (tmp_path / "research" / "2024-01-02_aaaaaaaaaaaa.json").write_text("{", encoding="utf-8")
    (tmp_path / DEMO_MARKER).write_text("{", encoding="utf-8")

    ws = load_workspace(tmp_path)

    assert ws.snapshots == ws.cycles == ws.research == ws.backtests == ()
    prefixes = {warning.split(":")[0] for warning in ws.unreadable}
    assert prefixes == {
        "runs/bad.json",
        "settlements/bad.json",
        "attributions/bad.json",
        "verifications/bad.json",
        "backtest/bad.json",
        "account",
        "research/2024-01-02",
        DEMO_MARKER,
    }


def test_a_store_that_does_not_load_is_a_warning_and_no_benchmark(tmp_path: Path) -> None:
    snapshots = JsonSnapshotStore(tmp_path)
    for day in (date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)):
        snapshots.save(
            AccountSnapshot(
                as_of=day,
                observed_at=datetime(day.year, day.month, day.day, 21, 30, tzinfo=UTC),
                equity=100.0,
                holdings=(),
            )
        )

    ws = load_workspace(tmp_path, store=tmp_path / "missing.store")

    assert len(ws.snapshots) == 3
    assert ws.benchmark is None
    assert [w.split(":")[0] for w in ws.unreadable] == ["store missing.store"]


def test_classify_a_demo_root(demo_root: Path) -> None:
    assert classify_root(demo_root) is RootKind.DEMO


def test_classify_a_backtest_root(demo_root: Path, tmp_path: Path) -> None:
    root = _copy(demo_root, tmp_path)
    (root / DEMO_MARKER).unlink()

    assert classify_root(root) is RootKind.BACKTEST


def test_a_marker_for_another_run_is_refused(demo_root: Path, tmp_path: Path) -> None:
    root = _copy(demo_root, tmp_path)
    marker = json.loads((root / DEMO_MARKER).read_text(encoding="utf-8"))
    marker["run_id"] = "someone-else"
    (root / DEMO_MARKER).write_text(json.dumps(marker), encoding="utf-8")

    with pytest.raises(RootError, match="someone-else"):
        classify_root(root)


def test_a_marker_for_another_store_is_refused(demo_root: Path, tmp_path: Path) -> None:
    root = _copy(demo_root, tmp_path)
    marker = json.loads((root / DEMO_MARKER).read_text(encoding="utf-8"))
    marker["store_digest"] = "0" * 64
    (root / DEMO_MARKER).write_text(json.dumps(marker), encoding="utf-8")

    with pytest.raises(RootError, match="store_digest"):
        classify_root(root)


def test_a_str_root_classifies_like_a_path(demo_root: Path) -> None:
    assert classify_root(str(demo_root)) is classify_root(demo_root)  # type: ignore[arg-type]


def test_a_marker_that_is_a_directory_is_refused(demo_root: Path, tmp_path: Path) -> None:
    root = _copy(demo_root, tmp_path)
    (root / DEMO_MARKER).unlink()
    (root / DEMO_MARKER).mkdir()

    with pytest.raises(RootError, match="not a demo marker"):
        classify_root(root)


def test_a_dangling_symlink_marker_is_refused(demo_root: Path, tmp_path: Path) -> None:
    root = _copy(demo_root, tmp_path)
    (root / DEMO_MARKER).unlink()
    (root / DEMO_MARKER).symlink_to(tmp_path / "missing.json")

    with pytest.raises(RootError, match="not a demo marker"):
        classify_root(root)


def test_a_marker_that_is_a_directory_loads_as_no_demo_with_a_warning(tmp_path: Path) -> None:
    (tmp_path / DEMO_MARKER).mkdir()

    ws = load_workspace(tmp_path)

    assert ws.demo is None
    assert ws.benchmark is None
    assert len(ws.unreadable) == 1
    assert ws.unreadable[0].startswith(f"{DEMO_MARKER}: ValueError: ")
    assert "not a demo marker" in ws.unreadable[0]


def test_a_marker_that_does_not_parse_is_refused(demo_root: Path, tmp_path: Path) -> None:
    root = _copy(demo_root, tmp_path)
    (root / DEMO_MARKER).write_text("{", encoding="utf-8")

    with pytest.raises(RootError, match="not a demo marker"):
        classify_root(root)


def test_a_live_root_without_a_backtest_summary_is_refused(tmp_path: Path) -> None:
    (tmp_path / "runs").mkdir()

    with pytest.raises(RootError, match="exactly one readable backtest summary"):
        classify_root(tmp_path)


def test_a_summary_not_named_for_its_run_is_refused(demo_root: Path, tmp_path: Path) -> None:
    root = _copy(demo_root, tmp_path)
    (root / DEMO_MARKER).unlink()
    [summary] = list((root / "backtest").glob("*.json"))
    summary.rename(summary.with_name("renamed.json"))

    with pytest.raises(RootError, match="is not named for run"):
        classify_root(root)


def test_two_summaries_are_refused(demo_root: Path, tmp_path: Path) -> None:
    root = _copy(demo_root, tmp_path)
    [summary] = list((root / "backtest").glob("*.json"))
    shutil.copy(summary, root / "backtest" / "other.json")

    with pytest.raises(RootError, match="exactly one readable backtest summary"):
        classify_root(root)


def test_a_summary_that_does_not_decode_is_refused(demo_root: Path, tmp_path: Path) -> None:
    root = _copy(demo_root, tmp_path)
    (root / "backtest" / "other.json").write_text("{", encoding="utf-8")

    with pytest.raises(RootError, match="exactly one readable backtest summary"):
        classify_root(root)
