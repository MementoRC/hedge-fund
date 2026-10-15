"""Tests for the factor crowding check."""

import dataclasses

import pytest

from hedge_fund.domain.scores import CrowdingReport, FactorScore, ScoreResult
from hedge_fund.domain.snapshot import PriceSnapshot
from hedge_fund.engine.factors.crowding import (
    DEFAULT_CROWDING,
    WINDOW_ROWS,
    CrowdingParams,
    crowding,
)
from tests.factories import make_closes, snapshot_from

TICKERS = ["A", "B", "C", "D"]
ONE_SECTOR = dict.fromkeys(TICKERS, "Tech")
NAN = float("nan")
WORST = CrowdingParams(quantile=0.0, min_leg=1, threshold=0.0, window_rows=1)


def _result(snap: PriceSnapshot, z: dict[str, dict[str, float]]) -> ScoreResult:
    """ScoreResult with the given z per ticker, composite = mean z, sorted like ``score()``."""
    scores = [
        FactorScore(
            ticker=t,
            sector=snap.sectors[t],
            raw=dict(zs),
            z=dict(zs),
            composite=sum(zs.values()) / len(zs),
            fallback_universe_z=False,
        )
        for t, zs in z.items()
    ]
    scores.sort(key=lambda s: (-s.composite, s.ticker))
    return ScoreResult(as_of=snap.as_of, scores=tuple(scores), excluded={})


def _valid() -> tuple[PriceSnapshot, ScoreResult]:
    snap = snapshot_from(make_closes(TICKERS), ONE_SECTOR)
    return snap, _result(snap, {t: {"f": float(i)} for i, t in enumerate(TICKERS)})


def _empty(snap: PriceSnapshot) -> ScoreResult:
    return ScoreResult(as_of=snap.as_of, scores=(), excluded={"A": "missing_data"})


# --- params ----------------------------------------------------------------


def test_window_rows_matches_default_scoring_window() -> None:
    assert WINDOW_ROWS == 253


def test_default_params() -> None:
    expected = CrowdingParams(quantile=0.2, min_leg=5, threshold=0.10, window_rows=253)
    assert expected == DEFAULT_CROWDING


# --- validation ------------------------------------------------------------


def test_as_of_mismatch_rejected() -> None:
    snap, result = _valid()
    earlier = snapshot_from(snap.closes.iloc[:-1], ONE_SECTOR)
    with pytest.raises(ValueError, match="as_of"):
        crowding(earlier, result)


def test_z_key_mismatch_rejected() -> None:
    snap, _ = _valid()
    z = {"A": {"f": 1.0}, "B": {"f": 0.0, "g": 0.0}, "C": {"f": -1.0}, "D": {"f": -2.0}}
    with pytest.raises(ValueError, match="z keys"):
        crowding(snap, _result(snap, z))


def test_scored_ticker_missing_from_closes_rejected() -> None:
    snap, result = _valid()
    partial = snapshot_from(snap.closes[TICKERS[:3]], ONE_SECTOR)
    with pytest.raises(ValueError, match=r"missing from closes: \['D'\]"):
        crowding(partial, result)


@pytest.mark.parametrize(
    ("params", "match"),
    [
        (CrowdingParams(quantile=0.0), "quantile"),
        (CrowdingParams(quantile=0.51), "quantile"),
        (CrowdingParams(quantile=NAN), "quantile"),
        (CrowdingParams(min_leg=1), "min_leg"),
        (CrowdingParams(threshold=0.0), "threshold"),
        (CrowdingParams(threshold=1.01), "threshold"),
        (CrowdingParams(threshold=NAN), "threshold"),
        (CrowdingParams(window_rows=1), "window_rows"),
    ],
)
def test_invalid_params_rejected(params: CrowdingParams, match: str) -> None:
    snap, result = _valid()
    with pytest.raises(ValueError, match=match):
        crowding(snap, result, params=params)


@pytest.mark.parametrize(
    ("params", "match"),
    [
        (WORST, "quantile"),
        (dataclasses.replace(WORST, quantile=0.2), "min_leg"),
        (dataclasses.replace(WORST, quantile=0.2, min_leg=2), "threshold"),
        (dataclasses.replace(WORST, quantile=0.2, min_leg=2, threshold=0.1), "window_rows"),
    ],
)
def test_param_checks_run_in_order(params: CrowdingParams, match: str) -> None:
    snap, result = _valid()
    with pytest.raises(ValueError, match=match):
        crowding(snap, result, params=params)


def test_input_checks_run_before_param_checks() -> None:
    snap, _ = _valid()
    bad_z = _result(snap, {"A": {"f": 1.0}, "B": {"g": 0.0}})
    good = _result(snap, {"A": {"f": 1.0}, "B": {"f": 0.0}})
    only_a = snapshot_from(snap.closes[["A"]], ONE_SECTOR)  # B is missing
    earlier_only_a = snapshot_from(snap.closes[["A"]].iloc[:-1], ONE_SECTOR)
    with pytest.raises(ValueError, match="as_of"):
        crowding(earlier_only_a, bad_z, params=WORST)
    with pytest.raises(ValueError, match="z keys"):
        crowding(only_a, bad_z, params=WORST)
    with pytest.raises(ValueError, match="missing from closes"):
        crowding(only_a, good, params=WORST)


# --- degenerate input ------------------------------------------------------


def test_empty_result_gives_empty_report() -> None:
    snap, _ = _valid()
    expected = CrowdingReport(as_of=snap.as_of, baseline_corr=None, factors=())
    assert crowding(snap, _empty(snap)) == expected


def test_empty_result_still_validates_params() -> None:
    snap, _ = _valid()
    with pytest.raises(ValueError, match="quantile"):
        crowding(snap, _empty(snap), params=CrowdingParams(quantile=0.0))


def test_boundary_params_accepted() -> None:
    snap, _ = _valid()
    edge = CrowdingParams(quantile=0.5, min_leg=2, threshold=1.0, window_rows=2)
    expected = CrowdingReport(as_of=snap.as_of, baseline_corr=None, factors=())
    assert crowding(snap, _empty(snap), params=edge) == expected
