"""The directory loader: decoded items in file-name order, bad files named, never raised."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from hedge_fund.adapters._load import load_dir
from hedge_fund.domain.loaded import Loaded


def _decode(obj: Any) -> int:
    return int(obj["n"])


def test_a_missing_directory_loads_nothing(tmp_path: Path) -> None:
    assert load_dir(tmp_path / "absent", _decode) == Loaded((), {})


def test_a_bad_file_between_good_ones_keeps_names_aligned_with_items(tmp_path: Path) -> None:
    (tmp_path / "a.json").write_text('{"n": 1}', encoding="utf-8")
    (tmp_path / "bad.json").write_text("{", encoding="utf-8")
    (tmp_path / "c.json").write_text('{"n": 3}', encoding="utf-8")

    loaded = load_dir(tmp_path, _decode)

    assert loaded.names == ("a.json", "c.json")
    assert loaded.items == (1, 3)
    assert list(loaded.unreadable) == ["bad.json"]


def test_a_directory_named_like_a_report_is_unreadable_not_raised(tmp_path: Path) -> None:
    (tmp_path / "x.json").mkdir()

    loaded = load_dir(tmp_path, _decode)

    assert loaded.items == ()
    assert list(loaded.unreadable) == ["x.json"]


def test_files_decode_in_name_order_and_bad_ones_are_named(tmp_path: Path) -> None:
    (tmp_path / "b.json").write_text('{"n": 2}', encoding="utf-8")
    (tmp_path / "a.json").write_text('{"n": 1}', encoding="utf-8")
    (tmp_path / "broken.json").write_text("{", encoding="utf-8")
    (tmp_path / "wrong.json").write_text('{"m": 1}', encoding="utf-8")
    (tmp_path / "note.txt").write_text("ignored", encoding="utf-8")

    loaded = load_dir(tmp_path, _decode)

    assert loaded.items == (1, 2)
    assert loaded.names == ("a.json", "b.json")
    assert list(loaded.unreadable) == ["broken.json", "wrong.json"]
    assert loaded.unreadable["broken.json"].startswith("JSONDecodeError: ")
    assert loaded.unreadable["wrong.json"] == "KeyError: 'n'"
