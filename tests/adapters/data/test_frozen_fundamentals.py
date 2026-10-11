"""Tests for content-addressed frozen FundamentalsSnapshot files."""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hedge_fund.adapters.data.frozen import load_fundamentals, save_fundamentals
from hedge_fund.domain.fundamentals import METRIC_NAMES, PERIOD_COLUMNS, FundamentalsSnapshot
from tests.factories import make_fundamentals


def _assert_same(actual: FundamentalsSnapshot, expected: FundamentalsSnapshot) -> None:
    assert actual.as_of == expected.as_of
    for name in ("current", "prior", "periods"):
        pd.testing.assert_frame_equal(
            getattr(actual, name),
            getattr(expected, name),
            check_index_type=False,
            check_column_type=False,
        )


def _replace(snapshot: FundamentalsSnapshot, **frames: pd.DataFrame) -> FundamentalsSnapshot:
    parts = {"current": snapshot.current, "prior": snapshot.prior, "periods": snapshot.periods}
    parts.update(frames)
    return FundamentalsSnapshot(as_of=snapshot.as_of, **parts)


def test_round_trip_preserves_all_three_frames_with_nan_and_nat(tmp_path: Path) -> None:
    base = make_fundamentals()
    current = base.current.copy()
    current.iloc[0, 0] = np.nan
    periods = base.periods.copy()
    periods.loc["BBB", "prior_end"] = pd.NaT
    snapshot = _replace(base, current=current, periods=periods)
    _assert_same(load_fundamentals(save_fundamentals(snapshot, tmp_path)), snapshot)


def test_file_lives_in_the_fundamentals_subdir_with_a_content_name(tmp_path: Path) -> None:
    path = save_fundamentals(make_fundamentals(), tmp_path)
    assert path.parent == tmp_path / "fundamentals"
    assert re.fullmatch(r"2026-09-30_[0-9a-f]{12}\.parquet", path.name)
    assert path.with_suffix(".json").is_file()


def test_sidecar_carries_as_of_metric_and_period_names(tmp_path: Path) -> None:
    path = save_fundamentals(make_fundamentals(), tmp_path)
    meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    assert meta["as_of"] == "2026-09-30"
    assert meta["metrics"] == list(METRIC_NAMES)
    assert meta["periods"] == list(PERIOD_COLUMNS)


def test_name_is_deterministic(tmp_path: Path) -> None:
    first = save_fundamentals(make_fundamentals(), tmp_path / "a")
    second = save_fundamentals(make_fundamentals(), tmp_path / "b")
    assert first.name == second.name


def test_repeat_save_is_a_noop(tmp_path: Path) -> None:
    first = save_fundamentals(make_fundamentals(), tmp_path)
    meta_before = first.with_suffix(".json").read_text(encoding="utf-8")
    mtime_before = first.stat().st_mtime_ns
    again = save_fundamentals(make_fundamentals(), tmp_path)
    assert again == first
    assert first.stat().st_mtime_ns == mtime_before
    assert first.with_suffix(".json").read_text(encoding="utf-8") == meta_before


@pytest.mark.parametrize("name", ["current", "prior"])
def test_metric_change_gives_a_different_file(tmp_path: Path, name: str) -> None:
    base = make_fundamentals()
    changed = getattr(base, name).copy()
    changed.iloc[0, 0] += 1.0
    other = _replace(base, **{name: changed})
    assert save_fundamentals(base, tmp_path) != save_fundamentals(other, tmp_path)


def test_period_change_gives_a_different_file(tmp_path: Path) -> None:
    base = make_fundamentals()
    periods = base.periods.copy()
    periods.loc["AAA", "last_filed"] = pd.Timestamp("2026-08-02")
    other = _replace(base, periods=periods)
    assert save_fundamentals(base, tmp_path) != save_fundamentals(other, tmp_path)


def test_non_canonical_nan_and_negative_zero_hash_like_canonical(tmp_path: Path) -> None:
    odd_nan = np.frombuffer(np.uint64(0x7FF8000000000001).tobytes(), dtype=np.float64)[0]
    base = make_fundamentals()
    canonical, odd = base.current.copy(), base.current.copy()
    canonical.iloc[0, 0], canonical.iloc[1, 1] = np.nan, 0.0
    odd.iloc[0, 0], odd.iloc[1, 1] = odd_nan, -0.0
    first = save_fundamentals(_replace(base, current=canonical), tmp_path / "a")
    second = save_fundamentals(_replace(base, current=odd), tmp_path / "b")
    assert first.name == second.name


def test_load_rejects_tampered_parquet(tmp_path: Path) -> None:
    path = save_fundamentals(make_fundamentals(), tmp_path)
    frame = pd.read_parquet(path)
    frame.iloc[0, 0] = frame.iloc[0, 0] * 2.0
    frame.to_parquet(path)
    with pytest.raises(ValueError, match=path.name):
        load_fundamentals(path)


def test_resaving_a_loaded_snapshot_keeps_its_name(tmp_path: Path) -> None:
    path = save_fundamentals(make_fundamentals(), tmp_path)
    assert save_fundamentals(load_fundamentals(path), tmp_path / "copy").name == path.name
