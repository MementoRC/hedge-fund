"""Tests for the atomic, durable file-write helpers."""

import os
import stat
from pathlib import Path

import pytest

from hedge_fund.adapters.data import _atomic
from hedge_fund.adapters.data._atomic import atomic_write


def _write(text: str):
    return lambda path: path.write_text(text, encoding="utf-8")


def _record_fsync(monkeypatch: pytest.MonkeyPatch) -> list[tuple[int, bool]]:
    """Record each os.fsync call as (fd, fd_is_a_directory)."""
    seen: list[tuple[int, bool]] = []
    real = os.fsync

    def spy(fd: int) -> None:
        seen.append((fd, stat.S_ISDIR(os.fstat(fd).st_mode)))
        real(fd)

    monkeypatch.setattr(os, "fsync", spy)
    return seen


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


def test_atomic_write_removes_temp_file_when_publish_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_replace(src: str, dst: str) -> None:
        raise OSError("rename refused")

    monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises(OSError, match="rename refused"):
        atomic_write(tmp_path / "x.json", _write("{}"))
    assert not (tmp_path / "x.json").exists()
    assert list(tmp_path.iterdir()) == []


def test_atomic_write_fsyncs_the_temp_file_then_the_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = _record_fsync(monkeypatch)
    atomic_write(tmp_path / "out.txt", _write("hello"))
    assert [is_dir for _, is_dir in seen] == [False, True]


def test_atomic_create_writes_and_fsyncs_the_temp_file_then_the_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = _record_fsync(monkeypatch)
    target = tmp_path / "nested" / "out.txt"
    _atomic.atomic_create(target, _write("hello"))
    assert target.read_text(encoding="utf-8") == "hello"
    assert [is_dir for _, is_dir in seen] == [False, True]
    assert [p.name for p in target.parent.iterdir()] == ["out.txt"]


def test_atomic_create_refuses_an_existing_file_and_leaves_it_intact(tmp_path: Path) -> None:
    target = tmp_path / "out.txt"
    target.write_text("old", encoding="utf-8")
    with pytest.raises(FileExistsError):
        _atomic.atomic_create(target, _write("new"))
    assert target.read_text(encoding="utf-8") == "old"
    assert [p.name for p in tmp_path.iterdir()] == ["out.txt"]


def test_atomic_create_loses_a_race_instead_of_overwriting(tmp_path: Path) -> None:
    target = tmp_path / "out.txt"

    def rival_publishes_first(path: Path) -> None:
        path.write_text("loser", encoding="utf-8")
        target.write_text("winner", encoding="utf-8")

    with pytest.raises(FileExistsError):
        _atomic.atomic_create(target, rival_publishes_first)
    assert target.read_text(encoding="utf-8") == "winner"
    assert [p.name for p in tmp_path.iterdir()] == ["out.txt"]


def test_atomic_create_failure_leaves_no_files(tmp_path: Path) -> None:
    def explode(path: Path) -> None:
        path.write_text("partial", encoding="utf-8")
        raise RuntimeError("disk full")

    with pytest.raises(RuntimeError, match="disk full"):
        _atomic.atomic_create(tmp_path / "out.txt", explode)
    assert list(tmp_path.iterdir()) == []
