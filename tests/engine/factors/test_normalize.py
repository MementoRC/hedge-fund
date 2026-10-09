"""Tests for winsorization and sector-neutral z-scores."""

import numpy as np
import pandas as pd
import pytest

from hedge_fund.engine.factors.normalize import sector_zscore, winsorize


def test_winsorize_clips_only_tails() -> None:
    values = pd.Series(np.arange(101, dtype=float))
    result = winsorize(values)
    assert result.iloc[0] == pytest.approx(1.0)
    assert result.iloc[100] == pytest.approx(99.0)
    pd.testing.assert_series_equal(result.iloc[1:100], values.iloc[1:100])


def test_zscore_standardizes_within_each_sector() -> None:
    values = pd.Series([1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 45.0], index=list("abcdefgh"))
    sectors = pd.Series(["X"] * 4 + ["Y"] * 4, index=values.index)
    z, fallback = sector_zscore(values, sectors)
    for name in ("X", "Y"):
        members = sectors.index[sectors == name]
        assert z[members].mean() == pytest.approx(0.0, abs=1e-12)
        assert z[members].std(ddof=0) == pytest.approx(1.0)
    assert not fallback.any()


def test_small_sector_falls_back_to_universe() -> None:
    values = pd.Series([1.0, 2.0, 3.0, 4.0, 100.0, 200.0], index=list("abcdef"))
    sectors = pd.Series(["X"] * 4 + ["Y"] * 2, index=values.index)
    z, fallback = sector_zscore(values, sectors)
    small = ["e", "f"]
    expected = (values[small] - values.mean()) / values.std(ddof=0)
    pd.testing.assert_series_equal(z[small], expected)
    assert fallback[small].all()
    assert not fallback[["a", "b", "c", "d"]].any()


def test_zero_variance_sector_falls_back_to_universe() -> None:
    values = pd.Series([5.0, 5.0, 5.0, 1.0, 2.0, 3.0], index=list("abcdef"))
    sectors = pd.Series(["X"] * 3 + ["Y"] * 3, index=values.index)
    z, fallback = sector_zscore(values, sectors)
    flat = ["a", "b", "c"]
    expected = (values[flat] - values.mean()) / values.std(ddof=0)
    pd.testing.assert_series_equal(z[flat], expected)
    assert fallback[flat].all()
    assert not fallback[["d", "e", "f"]].any()


def test_zero_universe_variance_gives_zero_z() -> None:
    values = pd.Series([7.0] * 4, index=list("abcd"))
    sectors = pd.Series(["X"] * 4, index=values.index)
    z, fallback = sector_zscore(values, sectors)
    assert (z == 0.0).all()
    assert fallback.all()
