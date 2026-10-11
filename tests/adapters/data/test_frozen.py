"""Tests for content-addressed frozen snapshot files."""

import re
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hedge_fund.adapters.data.frozen import load_snapshot, save_snapshot
from hedge_fund.domain.snapshot import PriceSnapshot
from tests.factories import make_closes, snapshot_from
from tests.fakes import assert_closes_equal

SECTORS = {"AAA": "Energy", "BBB": "Energy", "CCC": "Financials"}
SECTORS_AB = {"AAA": "Energy", "BBB": "Financials"}


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


def test_non_canonical_nan_and_negative_zero_hash_like_canonical(tmp_path: Path) -> None:
    odd_nan = np.frombuffer(np.uint64(0x7FF8000000000001).tobytes(), dtype=np.float64)[0]
    assert np.isnan(odd_nan)
    canonical = make_closes(list(SECTORS))
    odd = canonical.copy()
    canonical.iloc[0, 0] = np.nan
    canonical.iloc[1, 1] = 0.0
    odd.iloc[0, 0] = odd_nan
    odd.iloc[1, 1] = -0.0
    first = save_snapshot(snapshot_from(canonical, SECTORS), tmp_path / "a")
    second = save_snapshot(snapshot_from(odd, SECTORS), tmp_path / "b")
    assert first.name == second.name


def test_price_digest_is_pinned(tmp_path: Path) -> None:
    # Changing the price digest orphans every existing frozen price snapshot: update this
    # literal only with a deliberate migration.
    closes = pd.DataFrame(
        {"AAA": [10.0, np.nan, 12.5], "BBB": [-0.0, 20.25, 21.0]},
        index=pd.to_datetime(["2026-09-28", "2026-09-29", "2026-09-30"]),
    )
    snapshot = PriceSnapshot(as_of=date(2026, 9, 30), closes=closes, sectors=SECTORS_AB)
    assert save_snapshot(snapshot, tmp_path).name == "2026-09-30_baebe88ef290.parquet"


def test_load_rejects_tampered_parquet(tmp_path: Path) -> None:
    snapshot = _snapshot()
    path = save_snapshot(snapshot, tmp_path)
    (snapshot.closes * 2.0).to_parquet(path)
    with pytest.raises(ValueError, match=path.name):
        load_snapshot(path)


def test_resaving_a_loaded_snapshot_keeps_its_name(tmp_path: Path) -> None:
    path = save_snapshot(_snapshot(), tmp_path)
    assert save_snapshot(load_snapshot(path), tmp_path / "copy").name == path.name
