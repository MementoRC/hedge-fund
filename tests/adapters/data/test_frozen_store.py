"""The frozen backtest store: round trip, content-addressed names, fail-closed loading."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from hedge_fund.adapters.data.frozen_store import (
    KINDS,
    MANIFEST,
    FrozenStoreError,
    check_fresh_root,
    load_store,
    write_store,
)
from tests.factories import make_frozen_data
from tests.fakes import assert_closes_equal

CREATED = datetime(2026, 11, 5, 12, 0, tzinfo=UTC)


def _write(root: Path) -> str:
    return write_store(root, make_frozen_data(), created_at=CREATED)


def _manifest(root: Path) -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads((root / MANIFEST).read_text(encoding="utf-8"))
    return loaded


def _edit_manifest(root: Path, edit: Callable[[dict[str, Any]], None]) -> None:
    manifest = _manifest(root)
    edit(manifest)
    (root / MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")


def test_round_trip_preserves_every_field(tmp_path: Path) -> None:
    data = make_frozen_data()
    write_store(tmp_path / "store", data, created_at=CREATED)
    loaded = load_store(tmp_path / "store")
    assert (loaded.start, loaded.end) == (data.start, data.end)
    assert_closes_equal(loaded.closes, data.closes)
    assert_closes_equal(loaded.opens, data.opens)
    assert loaded.splits == data.splits
    assert loaded.dividends == data.dividends
    pd.testing.assert_frame_equal(loaded.facts, data.facts)
    assert dict(loaded.sectors) == dict(data.sectors)
    assert dict(loaded.ciks) == dict(data.ciks)
    assert loaded.membership == data.membership
    assert loaded.unpriced == data.unpriced
    assert loaded.no_sector == data.no_sector
    assert loaded.no_cik == data.no_cik
    assert loaded.sic_sectors == data.sic_sectors


def test_empty_event_tables_round_trip(tmp_path: Path) -> None:
    data = make_frozen_data(splits=(), dividends=(), membership=())
    write_store(tmp_path, data, created_at=CREATED)
    loaded = load_store(tmp_path)
    assert loaded.splits == ()
    assert loaded.dividends == ()
    assert loaded.membership == ()


def test_files_are_named_by_their_digest_and_listed_in_the_manifest(tmp_path: Path) -> None:
    _write(tmp_path)
    manifest = _manifest(tmp_path)
    assert sorted(manifest["files"]) == sorted(KINDS)
    for kind, entry in manifest["files"].items():
        assert re.fullmatch(rf"{kind}_[0-9a-f]{{12}}\.(parquet|json)", entry["name"])
        assert entry["name"].split("_")[1].startswith(entry["sha256"][:12])
        body = (tmp_path / entry["name"]).read_bytes()
        assert hashlib.sha256(body).hexdigest() == entry["sha256"]
    assert manifest["coverage"] == {"start": "2024-01-01", "end": "2024-02-09"}
    assert manifest["created_at"] == CREATED.isoformat()
    assert manifest["unpriced"] == ["CCC"]


def test_write_returns_the_manifest_digest(tmp_path: Path) -> None:
    digest = _write(tmp_path)
    assert digest == hashlib.sha256((tmp_path / MANIFEST).read_bytes()).hexdigest()


def test_write_refuses_a_non_empty_root_and_leaves_it_alone(tmp_path: Path) -> None:
    (tmp_path / "stray.txt").write_text("x", encoding="utf-8")
    with pytest.raises(FrozenStoreError, match="not empty"):
        _write(tmp_path)
    assert [path.name for path in tmp_path.iterdir()] == ["stray.txt"]


def test_check_fresh_root_accepts_an_absent_or_empty_root(tmp_path: Path) -> None:
    check_fresh_root(tmp_path / "absent")
    check_fresh_root(tmp_path)


def test_check_fresh_root_refuses_a_file(tmp_path: Path) -> None:
    path = tmp_path / "file"
    path.write_text("x", encoding="utf-8")
    with pytest.raises(FrozenStoreError, match="not empty"):
        check_fresh_root(path)


def test_a_tampered_file_fails_the_load(tmp_path: Path) -> None:
    _write(tmp_path)
    target = tmp_path / _manifest(tmp_path)["files"]["closes"]["name"]
    target.write_bytes(target.read_bytes() + b"\0")
    with pytest.raises(FrozenStoreError, match="digest mismatch"):
        load_store(tmp_path)


def test_a_file_renamed_away_from_its_digest_fails_the_load(tmp_path: Path) -> None:
    _write(tmp_path)
    entry = _manifest(tmp_path)["files"]["closes"]
    (tmp_path / entry["name"]).rename(tmp_path / "closes_000000000000.parquet")
    _edit_manifest(
        tmp_path, lambda m: m["files"]["closes"].update(name="closes_000000000000.parquet")
    )
    with pytest.raises(FrozenStoreError, match="digest mismatch"):
        load_store(tmp_path)


def test_a_missing_manifest_fails_the_load(tmp_path: Path) -> None:
    _write(tmp_path)
    (tmp_path / MANIFEST).unlink()
    with pytest.raises(FrozenStoreError, match="no manifest.json"):
        load_store(tmp_path)


def test_an_unreadable_manifest_fails_the_load(tmp_path: Path) -> None:
    _write(tmp_path)
    (tmp_path / MANIFEST).write_text("{", encoding="utf-8")
    with pytest.raises(FrozenStoreError, match="unreadable manifest"):
        load_store(tmp_path)


def test_a_manifest_without_a_key_fails_the_load(tmp_path: Path) -> None:
    _write(tmp_path)
    _edit_manifest(tmp_path, lambda m: m.pop("coverage"))
    with pytest.raises(FrozenStoreError, match="manifest lacks"):
        load_store(tmp_path)


def test_an_unknown_format_fails_the_load(tmp_path: Path) -> None:
    _write(tmp_path)
    _edit_manifest(tmp_path, lambda m: m.update(format=2))
    with pytest.raises(FrozenStoreError, match="format 2"):
        load_store(tmp_path)


def test_a_missing_kind_fails_the_load(tmp_path: Path) -> None:
    _write(tmp_path)
    _edit_manifest(tmp_path, lambda m: m["files"].pop("sectors"))
    with pytest.raises(FrozenStoreError, match="manifest files"):
        load_store(tmp_path)


def test_a_missing_file_fails_the_load(tmp_path: Path) -> None:
    _write(tmp_path)
    (tmp_path / _manifest(tmp_path)["files"]["facts"]["name"]).unlink()
    with pytest.raises(FrozenStoreError, match="is missing"):
        load_store(tmp_path)


@pytest.mark.parametrize("name", ["../closes_000000000000.parquet", "opens_000000000000.parquet"])
def test_a_name_outside_the_root_or_of_another_kind_fails_the_load(
    tmp_path: Path, name: str
) -> None:
    _write(tmp_path)
    _edit_manifest(tmp_path, lambda m: m["files"]["closes"].update(name=name))
    with pytest.raises(FrozenStoreError, match="bad file name"):
        load_store(tmp_path)
