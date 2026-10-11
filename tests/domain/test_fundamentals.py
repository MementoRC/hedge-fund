"""Tests for FundamentalsSnapshot validation."""

import numpy as np
import pandas as pd
import pytest

from hedge_fund.domain.fundamentals import METRIC_NAMES, PERIOD_COLUMNS, FundamentalsSnapshot
from tests.factories import make_fundamentals


def _build(**overrides: pd.DataFrame) -> FundamentalsSnapshot:
    base = make_fundamentals()
    frames = {"current": base.current, "prior": base.prior, "periods": base.periods}
    frames.update(overrides)
    return FundamentalsSnapshot(as_of=base.as_of, **frames)


def test_valid_snapshot_keeps_frames() -> None:
    snapshot = make_fundamentals()
    assert list(snapshot.current.index) == ["AAA", "BBB", "CCC"]
    assert list(snapshot.current.columns) == list(METRIC_NAMES)
    assert list(snapshot.periods.columns) == list(PERIOD_COLUMNS)


def test_nan_metrics_and_nat_prior_end_are_allowed() -> None:
    base = make_fundamentals()
    current = base.current.copy()
    current.iloc[0, 0] = np.nan
    periods = base.periods.copy()
    periods.loc["AAA", "prior_end"] = pd.NaT
    snapshot = _build(current=current, periods=periods)
    assert pd.isna(snapshot.periods.loc["AAA", "prior_end"])


@pytest.mark.parametrize("name", ["current", "prior", "periods"])
def test_empty_frame_is_rejected(name: str) -> None:
    frame = getattr(make_fundamentals(), name).iloc[0:0]
    with pytest.raises(ValueError, match=f"{name} is empty"):
        _build(**{name: frame})


def test_indexes_must_match() -> None:
    prior = make_fundamentals().prior.set_axis(["AAA", "BBB", "ZZZ"])
    with pytest.raises(ValueError, match="same index"):
        _build(prior=prior)


def test_index_must_be_sorted() -> None:
    base = make_fundamentals()
    order = ["BBB", "AAA", "CCC"]
    with pytest.raises(ValueError, match="sorted"):
        _build(
            current=base.current.loc[order],
            prior=base.prior.loc[order],
            periods=base.periods.loc[order],
        )


def test_index_must_be_unique() -> None:
    base = make_fundamentals()
    labels = ["AAA", "AAA", "CCC"]
    with pytest.raises(ValueError, match="unique"):
        _build(
            current=base.current.set_axis(labels),
            prior=base.prior.set_axis(labels),
            periods=base.periods.set_axis(labels),
        )


@pytest.mark.parametrize("name", ["current", "prior"])
def test_metric_columns_must_be_metrics_in_order(name: str) -> None:
    frame = getattr(make_fundamentals(), name)
    with pytest.raises(ValueError, match=f"{name} columns"):
        _build(**{name: frame[list(reversed(METRIC_NAMES))]})
    with pytest.raises(ValueError, match=f"{name} columns"):
        _build(**{name: frame.drop(columns=["cash"])})


def test_periods_columns_must_be_exact() -> None:
    periods = make_fundamentals().periods.rename(columns={"last_filed": "filed"})
    with pytest.raises(ValueError, match="periods columns"):
        _build(periods=periods)


def test_periods_columns_must_be_in_order() -> None:
    periods = make_fundamentals().periods[list(reversed(PERIOD_COLUMNS))]
    with pytest.raises(ValueError, match="periods columns"):
        _build(periods=periods)


def test_periods_must_be_datetime() -> None:
    periods = make_fundamentals().periods.astype("str")
    with pytest.raises(ValueError, match="datetime64"):
        _build(periods=periods)


@pytest.mark.parametrize(("name", "value"), [("current", np.inf), ("prior", -np.inf)])
def test_infinite_metric_values_are_rejected(name: str, value: float) -> None:
    frame = getattr(make_fundamentals(), name).copy()
    frame.iloc[1, 2] = value
    with pytest.raises(ValueError, match="inf"):
        _build(**{name: frame})
