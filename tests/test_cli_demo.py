"""``hedge-fund demo build``: layout, marker, determinism and the --force safety rules."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest
import typer
from typer.testing import CliRunner, Result

from hedge_fund import cli
from hedge_fund.adapters.backtest.report import BACKTEST_DIR
from hedge_fund.adapters.data.frozen_store import MANIFEST
from hedge_fund.adapters.demo_marker import DEMO_MARKER, read_demo_marker, write_demo_marker
from hedge_fund.domain.demo import DemoMarker

# The golden test's pins: a seed-1 demo is that store, frozen at the same instant and replayed
# with the same cash over the same window.
GOLDEN_STORE_DIGEST = "6fa8739dad872645e7752901a53d601558d8919287e197171ee7dfba864d0bcd"
GOLDEN_RUN_ID = "2024-01-10_2024-03-08_da376fb95d23"
runner = CliRunner()


def _build(root: Path, *args: str) -> Result:
    return runner.invoke(cli.app, ["demo", "build", "--root", str(root), *args])


def _store(root: Path) -> Path:
    return root.with_name(f"{root.name}.store")


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("demo") / "demo"
    result = _build(root)
    assert result.exit_code == 0, result.output
    return root


def test_the_build_writes_the_store_the_run_and_the_marker(built: Path) -> None:
    marker = read_demo_marker(built)

    assert marker == DemoMarker(seed=1, store_digest=GOLDEN_STORE_DIGEST, run_id=GOLDEN_RUN_ID)
    manifest = _store(built) / MANIFEST
    assert hashlib.sha256(manifest.read_bytes()).hexdigest() == marker.store_digest
    assert (built / "backtest" / f"{GOLDEN_RUN_ID}.json").is_file()
    assert any((built / "runs").glob("*.json"))


def test_a_forced_rebuild_replaces_everything_and_repeats_the_digest(built: Path) -> None:
    (built / "stale.txt").write_text("old", encoding="utf-8")

    result = _build(built, "--force")

    assert result.exit_code == 0, result.output
    assert not (built / "stale.txt").exists()
    assert read_demo_marker(built) == DemoMarker(1, GOLDEN_STORE_DIGEST, GOLDEN_RUN_ID)


def test_a_seed_below_one_is_refused_before_anything_is_deleted(built: Path) -> None:
    before = read_demo_marker(built)

    result = _build(built, "--force", "--seed", "0")

    assert result.exit_code != 0
    assert read_demo_marker(built) == before
    assert (_store(built) / MANIFEST).is_file()
    assert any((built / "runs").glob("*.json"))


def test_the_seed_is_passed_through_and_recorded(tmp_path: Path) -> None:
    root = tmp_path / "demo2"

    result = _build(root, "--seed", "2")

    assert result.exit_code == 0, result.output
    marker = read_demo_marker(root)
    assert marker is not None
    assert marker.seed == 2
    assert marker.store_digest != GOLDEN_STORE_DIGEST


def test_the_marker_is_written_last(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def failing(**_: object) -> None:
        raise typer.Exit(1)

    monkeypatch.setattr(cli, "backtest_run", failing)
    root = tmp_path / "demo"

    result = _build(root)

    assert result.exit_code == 1
    assert (_store(root) / MANIFEST).is_file()
    assert not (root / DEMO_MARKER).exists()


def test_a_run_that_leaves_no_single_summary_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "backtest_run", lambda **_: None)

    result = _build(tmp_path / "demo")

    assert result.exit_code == 1
    assert "expected exactly one readable backtest summary" in result.output


def test_a_store_that_cannot_be_written_is_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(seed: int) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(cli.world, "make_golden_data", boom)

    result = _build(tmp_path / "demo")

    assert result.exit_code == 1
    assert "cannot write the demo store: RuntimeError: boom" in result.output


@pytest.mark.parametrize("which", ["root", "store"])
def test_an_existing_demo_is_refused_without_force(tmp_path: Path, which: str) -> None:
    root = tmp_path / "demo"
    target = root if which == "root" else _store(root)
    target.mkdir()
    (target / "keep.txt").write_text("x", encoding="utf-8")

    result = _build(root)

    assert result.exit_code == 1
    assert "already exists" in result.output
    assert (target / "keep.txt").is_file()


def test_force_refuses_a_root_without_a_marker(tmp_path: Path) -> None:
    root = tmp_path / "demo"
    root.mkdir()
    (root / "keep.txt").write_text("x", encoding="utf-8")

    result = _build(root, "--force")

    assert result.exit_code == 1
    assert DEMO_MARKER in result.output
    assert (root / "keep.txt").is_file()


def test_force_refuses_a_root_whose_marker_does_not_parse(tmp_path: Path) -> None:
    root = tmp_path / "demo"
    root.mkdir()
    (root / DEMO_MARKER).write_text("{", encoding="utf-8")

    result = _build(root, "--force")

    assert result.exit_code == 1
    assert "not a demo marker" in result.output
    assert (root / DEMO_MARKER).is_file()


def test_force_refuses_a_root_whose_marker_is_a_directory(tmp_path: Path) -> None:
    root = tmp_path / "demo"
    (root / DEMO_MARKER).mkdir(parents=True)

    result = _build(root, "--force")

    assert result.exit_code == 1
    assert "not a demo marker" in result.output
    assert (root / DEMO_MARKER).is_dir()


def test_force_refuses_a_store_without_a_manifest(tmp_path: Path) -> None:
    root = tmp_path / "demo"
    root.mkdir()
    write_demo_marker(root, DemoMarker(1, "d" * 64, "r"))
    store = _store(root)
    store.mkdir()
    (store / "keep.txt").write_text("x", encoding="utf-8")

    result = _build(root, "--force")

    assert result.exit_code == 1
    assert MANIFEST in result.output
    assert (store / "keep.txt").is_file()
    assert (root / DEMO_MARKER).is_file()


@pytest.mark.parametrize("force", [False, True])
def test_a_dangling_symlink_root_is_refused_and_left_alone(tmp_path: Path, force: bool) -> None:
    root = tmp_path / "demo"
    root.symlink_to(tmp_path / "missing")

    with pytest.raises(ValueError, match=r"already exists|has no DEMO\.json"):
        cli._clear_demo(root, _store(root), force=force)

    assert root.is_symlink()
    assert not (tmp_path / "missing").exists()


def test_force_refuses_a_demo_whose_run_summary_is_gone(tmp_path: Path) -> None:
    root = tmp_path / "demo"
    assert _build(root).exit_code == 0
    (root / BACKTEST_DIR / f"{GOLDEN_RUN_ID}.json").unlink()

    result = _build(root, "--force")

    assert result.exit_code == 1
    assert GOLDEN_RUN_ID in result.output
    assert (root / DEMO_MARKER).is_file()
    assert (_store(root) / MANIFEST).is_file()


def test_a_summary_for_another_store_fails_the_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_write_store = cli.write_store

    def wrong_digest(*args: Any, **kwargs: Any) -> str:
        real_write_store(*args, **kwargs)
        return "f" * 64

    monkeypatch.setattr(cli, "write_store", wrong_digest)
    root = tmp_path / "demo"

    result = _build(root)

    assert result.exit_code == 1
    assert "store_digest" in result.output
    assert not (root / DEMO_MARKER).exists()


def test_force_never_wipes_a_store_whose_root_is_absent(tmp_path: Path) -> None:
    root = tmp_path / "data"
    store = _store(root)
    store.mkdir()
    (store / "keep.txt").write_text("x", encoding="utf-8")

    result = _build(root, "--force")

    assert result.exit_code == 1
    assert (store / "keep.txt").is_file()
