"""Tests for the factor crowding check."""

import dataclasses

import numpy as np
import pandas as pd
import pytest

from hedge_fund.domain.scores import CrowdingReport, FactorCrowding, FactorScore, ScoreResult
from hedge_fund.domain.snapshot import PriceSnapshot
from hedge_fund.engine.factors.crowding import (
    DEFAULT_CROWDING,
    WINDOW_ROWS,
    CrowdingParams,
    crowding,
)
from tests.factories import END_DATE, make_closes, snapshot_from

TICKERS = ["A", "B", "C", "D"]
ONE_SECTOR = dict.fromkeys(TICKERS, "Tech")
NAN = float("nan")
WORST = CrowdingParams(quantile=0.0, min_leg=1, threshold=0.0, window_rows=1)

# Test 1 layout: 4 tickers in 1 sector, 6 rows of closes (5 return days).
# Log return of ticker t on day d = 0.01 * RESIDUAL[t][d] + MARKET[d].
# H1, H2, H3 are orthogonal, have zero time-mean and |H|^2 = 4.
H1 = np.array([1.0, -1.0, 1.0, -1.0, 0.0])
H2 = np.array([1.0, 1.0, -1.0, -1.0, 0.0])
H3 = np.array([1.0, -1.0, -1.0, 1.0, 0.0])
RESIDUAL = {
    "A": H1,  # [ 1, -1,  1, -1, 0]
    "B": H2,  # [ 1,  1, -1, -1, 0]
    "C": -2 * H1 + H2 + 2 * H3,  # [ 1,  1, -5,  3, 0]
    "D": H1 - 2 * H2 - 2 * H3,  # [-3, -1,  5, -1, 0]
}
MARKET = np.array([0.004, -0.002, 0.003, 0.001, -0.005])
# Each day the four RESIDUAL values sum to 0, so the sector mean return is MARKET and the
# sector-demeaned residuals are exactly 0.01 * RESIDUAL[t] (the scale cancels in corr).
# Every residual has zero time-mean, so corr(X, Y) = X.Y / (|X| |Y|), |A| = |B| = 2,
# |C| = |D| = 6:
#   A.B =   0 ->  0      A.C = -8 -> -2/3     A.D = 4 -> 1/3
#   B.C =   4 ->  1/3    B.D = -8 -> -2/3     C.D = -32 -> -8/9
# baseline = (0 - 2/3 + 1/3 + 1/3 - 2/3 - 8/9) / 6 = (-14/9) / 6 = -7/27
# quantile 0.5 of 4 names gives m = 2.
# f: long {A, B} = 0 -> 0 + 7/27 = 7/27 (>= 0.10: crowded)
#    short {D, C} = -8/9 -> -24/27 + 7/27 = -17/27
# g: long {A, C} = -2/3 -> -18/27 + 7/27 = -11/27
#    short {D, B} = -2/3 -> -11/27 (not crowded)
HAND = CrowdingParams(quantile=0.5, min_leg=2, window_rows=6)
HAND_Z = {
    "A": {"f": 2.0, "g": 2.0},
    "B": {"f": 1.0, "g": -1.0},
    "C": {"f": -1.0, "g": 1.0},
    "D": {"f": -2.0, "g": -2.0},
}
HAND_BASELINE = -7 / 27
HAND_FACTORS = (
    FactorCrowding("f", pytest.approx(7 / 27), pytest.approx(-17 / 27), 2, 2, True),
    FactorCrowding("g", pytest.approx(-11 / 27), pytest.approx(-11 / 27), 2, 2, False),
)


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


def _hand_closes() -> pd.DataFrame:
    """6 rows starting at 100 whose log returns are 0.01 * RESIDUAL[t] + MARKET."""
    index = pd.bdate_range(end=END_DATE, periods=6)
    log_prices = {
        t: np.cumsum(np.concatenate(([0.0], 0.01 * r + MARKET))) for t, r in RESIDUAL.items()
    }
    return pd.DataFrame({t: 100.0 * np.exp(p) for t, p in log_prices.items()}, index=index)


def _hand_snapshot() -> PriceSnapshot:
    return snapshot_from(_hand_closes(), ONE_SECTOR)


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


# --- hand-computed case ----------------------------------------------------


def test_hand_computed_baseline_and_legs() -> None:
    snap = _hand_snapshot()
    report = crowding(snap, _result(snap, HAND_Z), params=HAND)
    assert report.as_of == snap.as_of
    assert report.baseline_corr == pytest.approx(HAND_BASELINE)
    assert report.factors == HAND_FACTORS


def test_short_closes_give_none_without_raising() -> None:
    snap = snapshot_from(_hand_closes().iloc[-1:], ONE_SECTOR)  # 1 row: 0 return rows
    report = crowding(snap, _result(snap, HAND_Z), params=HAND)
    assert report.baseline_corr is None
    assert report.factors == (
        FactorCrowding("f", None, None, 2, 2, False),
        FactorCrowding("g", None, None, 2, 2, False),
    )


def test_small_sectors_use_universe_mean() -> None:
    # Four 1-name sectors fall back to the universe mean, which here equals the
    # one-sector mean. Without the fallback every residual would be 0 (baseline None).
    snap = snapshot_from(_hand_closes(), {t: f"S{t}" for t in TICKERS})
    report = crowding(snap, _result(snap, HAND_Z), params=HAND)
    assert report.baseline_corr == pytest.approx(HAND_BASELINE)
    assert report.factors == HAND_FACTORS


def test_ties_broken_by_ticker_ascending() -> None:
    # A, B and C tie on z. Long (z desc, ticker asc) = {A, B}: 0 -> 7/27.
    # Short (z asc, ticker asc) = {D, A}: 1/3 -> 9/27 + 7/27 = 16/27.
    snap = _hand_snapshot()
    z = {"A": {"f": 0.0}, "B": {"f": 0.0}, "C": {"f": 0.0}, "D": {"f": -1.0}}
    (factor,) = crowding(snap, _result(snap, z), params=HAND).factors
    assert factor.long_excess == pytest.approx(7 / 27)
    assert factor.short_excess == pytest.approx(16 / 27)


def test_interior_nan_is_forward_filled() -> None:
    gappy = _hand_closes()
    gappy.iloc[2, 0] = np.nan
    gappy_snap = snapshot_from(gappy, ONE_SECTOR)
    filled_snap = snapshot_from(gappy.ffill(), ONE_SECTOR)
    gappy_report = crowding(gappy_snap, _result(gappy_snap, HAND_Z), params=HAND)
    assert gappy_report == crowding(filled_snap, _result(filled_snap, HAND_Z), params=HAND)


def test_rows_before_window_are_ignored() -> None:
    closes = _hand_closes()
    early = pd.DataFrame(
        {
            "A": [50.0, 80.0, 120.0],
            "B": [90.0, 60.0, 100.0],
            "C": [100.0, 130.0, 70.0],
            "D": [110.0, 95.0, 105.0],
        },
        index=pd.bdate_range(end=closes.index[0] - pd.offsets.BDay(), periods=3),
    )
    longer = snapshot_from(pd.concat([early, closes]), ONE_SECTOR)
    base = snapshot_from(closes, ONE_SECTOR)
    trimmed = crowding(longer, _result(longer, HAND_Z), params=HAND)
    assert trimmed == crowding(base, _result(base, HAND_Z), params=HAND)
    wide = dataclasses.replace(HAND, window_rows=9)
    assert crowding(longer, _result(longer, HAND_Z), params=wide) != trimmed
