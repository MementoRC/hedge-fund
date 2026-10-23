"""Tests for the atomic file-write helper."""

from pathlib import Path

import pytest

from hedge_fund.adapters.data._atomic import atomic_write


def test_atomic_write_creates_parent_and_file(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "out.txt"
    atomic_write(target, lambda path: path.write_text("hello", encoding="utf-8"))
    assert target.read_text(encoding="utf-8") == "hello"
    assert [p.name for p in target.parent.iterdir()] == ["out.txt"]


def test_atomic_write_replaces_existing_file(tmp_path: Path) -> None:
    target = tmp_path / "out.txt"
    target.write_text("old", encoding="utf-8")
    atomic_write(target, lambda path: path.write_text("new", encoding="utf-8"))
    assert target.read_text(encoding="utf-8") == "new"


def test_atomic_write_failure_leaves_no_files(tmp_path: Path) -> None:
    target = tmp_path / "out.txt"

    def explode(path: Path) -> None:
        path.write_text("partial", encoding="utf-8")
        raise RuntimeError("disk full")

    with pytest.raises(RuntimeError, match="disk full"):
        atomic_write(target, explode)
    assert list(tmp_path.iterdir()) == []
