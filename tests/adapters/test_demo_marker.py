"""The DEMO.json marker: written last, read strictly."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from hedge_fund.adapters import demo_marker
from hedge_fund.adapters.demo_marker import (
    DEMO_MARKER,
    demo_store_dir,
    read_demo_marker,
    write_demo_marker,
)
from hedge_fund.domain.demo import DemoMarker

MARKER = DemoMarker(seed=3, store_digest="a" * 64, run_id="2024-01-10_2024-03-08_abc123def456")


def test_the_marker_round_trips_with_the_documented_shape(tmp_path: Path) -> None:
    path = write_demo_marker(tmp_path, MARKER)

    assert path == tmp_path / DEMO_MARKER
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "kind": "demo",
        "format": 1,
        "seed": 3,
        "store_digest": "a" * 64,
        "run_id": "2024-01-10_2024-03-08_abc123def456",
    }
    assert read_demo_marker(tmp_path) == MARKER


def test_a_root_without_a_marker_reads_as_none(tmp_path: Path) -> None:
    assert read_demo_marker(tmp_path) is None


def test_a_marker_that_is_a_directory_is_refused(tmp_path: Path) -> None:
    (tmp_path / DEMO_MARKER).mkdir()

    with pytest.raises(ValueError, match="not a demo marker"):
        read_demo_marker(tmp_path)


def test_a_dangling_symlink_marker_is_refused(tmp_path: Path) -> None:
    (tmp_path / DEMO_MARKER).symlink_to(tmp_path / "missing.json")

    with pytest.raises(ValueError, match="not a demo marker"):
        read_demo_marker(tmp_path)


def test_a_symlink_to_a_valid_marker_parses(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    write_demo_marker(real, MARKER)
    root = tmp_path / "root"
    root.mkdir()
    (root / DEMO_MARKER).symlink_to(real / DEMO_MARKER)

    assert read_demo_marker(root) == MARKER


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores directory permissions")
def test_an_unsearchable_parent_is_a_value_error(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    write_demo_marker(root, MARKER)
    root.chmod(0o000)
    try:
        with pytest.raises(ValueError, match="not a demo marker"):
            read_demo_marker(root)
    finally:
        root.chmod(0o700)


def test_the_marker_is_never_overwritten(tmp_path: Path) -> None:
    write_demo_marker(tmp_path, MARKER)

    with pytest.raises(FileExistsError):
        write_demo_marker(tmp_path, MARKER)


def test_the_store_sits_beside_the_root() -> None:
    assert demo_store_dir(Path("data/demo")) == Path("data/demo.store")


def test_a_path_component_that_is_a_file_is_refused(tmp_path: Path) -> None:
    not_a_dir = tmp_path / "file"
    not_a_dir.write_text("x", encoding="utf-8")

    with pytest.raises(ValueError, match="not a demo marker"):
        read_demo_marker(not_a_dir)


def test_a_symlink_loop_marker_is_refused(tmp_path: Path) -> None:
    (tmp_path / DEMO_MARKER).symlink_to(tmp_path / DEMO_MARKER)

    with pytest.raises(ValueError, match="not a demo marker"):
        read_demo_marker(tmp_path)


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="no FIFOs on this platform")
def test_a_fifo_marker_is_refused_without_blocking(tmp_path: Path) -> None:
    os.mkfifo(tmp_path / DEMO_MARKER)

    with pytest.raises(ValueError, match="not a demo marker"):
        read_demo_marker(tmp_path)


def test_a_permission_error_while_examining_the_marker_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def denied(path: object) -> None:
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(demo_marker, "os", SimpleNamespace(lstat=denied))

    with pytest.raises(ValueError, match="not a demo marker"):
        read_demo_marker(tmp_path)


GOOD = {"kind": "demo", "format": 1, "seed": 1, "store_digest": "a" * 64, "run_id": "r"}


def _marker(**fields: Any) -> DemoMarker:
    return DemoMarker(**{"seed": 1, "store_digest": "a" * 64, "run_id": "r", **fields})


@pytest.mark.parametrize(
    "marker",
    [
        _marker(seed=True),
        _marker(store_digest=5),
        _marker(run_id=None),
        _marker(store_digest="d"),
        _marker(store_digest="A" * 64),
        _marker(store_digest="a" * 64 + "\n"),
        _marker(run_id=""),
    ],
)
def test_the_writer_refuses_a_marker_the_reader_would_reject(
    tmp_path: Path, marker: DemoMarker
) -> None:
    with pytest.raises(ValueError, match="demo marker"):
        write_demo_marker(tmp_path, marker)

    assert not (tmp_path / DEMO_MARKER).exists()


@pytest.mark.parametrize(
    "body",
    [
        json.dumps({**GOOD, "store_digest": "d"}),
        json.dumps({**GOOD, "store_digest": "A" * 64}),
        json.dumps({**GOOD, "store_digest": "a" * 63}),
        json.dumps({**GOOD, "run_id": ""}),
        "{",
        "[]",
        json.dumps({**GOOD, "extra": 1}),
        json.dumps({k: v for k, v in GOOD.items() if k != "seed"}),
        json.dumps({**GOOD, "kind": "backtest"}),
        json.dumps({**GOOD, "format": 2}),
        json.dumps({**GOOD, "seed": True}),
        json.dumps({**GOOD, "seed": "1"}),
        json.dumps({**GOOD, "store_digest": 5}),
        json.dumps({**GOOD, "run_id": None}),
    ],
)
def test_a_marker_that_is_not_exactly_a_demo_marker_is_refused(tmp_path: Path, body: str) -> None:
    (tmp_path / DEMO_MARKER).write_text(body, encoding="utf-8")

    with pytest.raises(ValueError, match="not a demo marker"):
        read_demo_marker(tmp_path)


@pytest.mark.parametrize("run_id", ["../x", "a/b", "..", ".", "/abs"])
def test_a_run_id_that_is_not_a_plain_file_name_is_refused_by_reader_and_writer(
    tmp_path: Path, run_id: str
) -> None:
    with pytest.raises(ValueError, match="not a plain file name"):
        write_demo_marker(tmp_path, _marker(run_id=run_id))
    assert not (tmp_path / DEMO_MARKER).exists()

    (tmp_path / DEMO_MARKER).write_text(json.dumps({**GOOD, "run_id": run_id}), encoding="utf-8")
    with pytest.raises(ValueError, match="not a plain file name"):
        read_demo_marker(tmp_path)
