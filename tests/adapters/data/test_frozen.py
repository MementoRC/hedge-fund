"""Tests for content-addressed frozen snapshot files."""

import re
from pathlib import Path

from hedge_fund.adapters.data.frozen import load_snapshot, save_snapshot
from hedge_fund.domain.snapshot import PriceSnapshot
from tests.factories import make_closes, snapshot_from
from tests.fakes import assert_closes_equal

SECTORS = {"AAA": "Energy", "BBB": "Energy", "CCC": "Financials"}


def _snapshot(scale: float = 1.0, sectors: dict[str, str] = SECTORS) -> PriceSnapshot:
    return snapshot_from(make_closes(list(SECTORS)) * scale, sectors)


def test_round_trip_preserves_frame_sectors_and_as_of(tmp_path: Path) -> None:
    snapshot = _snapshot()
    loaded = load_snapshot(save_snapshot(snapshot, tmp_path))
    assert loaded.as_of == snapshot.as_of
    assert dict(loaded.sectors) == SECTORS
    assert_closes_equal(loaded.closes, snapshot.closes)


def test_name_is_deterministic(tmp_path: Path) -> None:
    first = save_snapshot(_snapshot(), tmp_path / "a")
    second = save_snapshot(_snapshot(), tmp_path / "b")
    assert first.name == second.name
    assert re.fullmatch(r"2026-09-30_[0-9a-f]{12}\.parquet", first.name)
    assert first.with_suffix(".json").is_file()


def test_repeat_save_is_a_noop(tmp_path: Path) -> None:
    first = save_snapshot(_snapshot(), tmp_path)
    meta_before = first.with_suffix(".json").read_text(encoding="utf-8")
    mtime_before = first.stat().st_mtime_ns
    again = save_snapshot(_snapshot(), tmp_path)
    assert again == first
    assert first.stat().st_mtime_ns == mtime_before
    assert first.with_suffix(".json").read_text(encoding="utf-8") == meta_before
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(
        [first.name, first.with_suffix(".json").name]
    )


def test_different_prices_give_a_different_file(tmp_path: Path) -> None:
    assert save_snapshot(_snapshot(), tmp_path) != save_snapshot(_snapshot(1.01), tmp_path)


def test_different_sectors_give_a_different_file(tmp_path: Path) -> None:
    other = {**SECTORS, "CCC": "Energy"}
    assert save_snapshot(_snapshot(), tmp_path) != save_snapshot(_snapshot(sectors=other), tmp_path)


def test_resaving_a_loaded_snapshot_keeps_its_name(tmp_path: Path) -> None:
    path = save_snapshot(_snapshot(), tmp_path)
    assert save_snapshot(load_snapshot(path), tmp_path / "copy").name == path.name
