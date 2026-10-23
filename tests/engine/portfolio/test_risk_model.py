"""Tests for the risk model: params, Ledoit-Wolf shrinkage, returns window, beta and vol."""

import dataclasses

import numpy as np
import pandas as pd
import pytest

from hedge_fund.domain.shortlist import Side
from hedge_fund.engine.portfolio.correlation import CorrelationParams, correlation_filter
from hedge_fund.engine.portfolio.risk_model import (
    DEFAULT_RISK,
    RiskParams,
    ledoit_wolf,
    risk_model,
    simple_returns,
    validate_risk_params,
)

NAN = float("nan")
NO_PRICES = pd.DataFrame()
NO_BENCHMARK = pd.Series(dtype=float)
SPY_RETURNS = [0.01, -0.02, 0.015, 0.005]
WIGGLE = [0.01, -0.02, 0.015, 0.005, -0.01, 0.02]  # six returns -> seven closes


def _prices(returns: list[float], start: float = 100.0) -> list[float]:
    """Close path whose simple returns are ``returns`` (up to float rounding)."""
    prices = [start]
    for value in returns:
        prices.append(prices[-1] * (1 + value))
    return prices


def _with_nan(prices: list[float], *rows: int) -> list[float]:
    return [NAN if row in rows else price for row, price in enumerate(prices)]


def _index(n: int) -> pd.DatetimeIndex:
    return pd.bdate_range(end="2026-09-30", periods=n)


def _closes(columns: dict[str, list[float]]) -> pd.DataFrame:
    n = len(next(iter(columns.values())))
    return pd.DataFrame(columns, index=_index(n))


def _spy(prices: list[float]) -> pd.Series:
    return pd.Series(prices, index=_index(len(prices)))


# --- params ----------------------------------------------------------------


def test_default_params() -> None:
    assert RiskParams(lookback=252, min_obs=200) == DEFAULT_RISK


@pytest.mark.parametrize("field", ["lookback", "min_obs"])
@pytest.mark.parametrize("value", [10.0, NAN, True, "x"])
def test_rejects_non_int_counts(field: str, value: object) -> None:
    params = dataclasses.replace(DEFAULT_RISK, **{field: value})
    with pytest.raises(ValueError, match=f"{field} must be an int"):
        risk_model(NO_PRICES, NO_BENCHMARK, params=params)


@pytest.mark.parametrize(
    ("lookback", "min_obs"),
    [(252, 1), (252, 0), (10, 11), (1, 1)],
)
def test_rejects_min_obs_outside_two_to_lookback(lookback: int, min_obs: int) -> None:
    params = RiskParams(lookback=lookback, min_obs=min_obs)
    with pytest.raises(ValueError, match="2 <= min_obs <= lookback"):
        risk_model(NO_PRICES, NO_BENCHMARK, params=params)


@pytest.mark.parametrize(
    ("lookback", "min_obs"),
    [(10, 10), (10, 2), (2, 2)],
)
def test_accepts_boundary_valid_params(lookback: int, min_obs: int) -> None:
    validate_risk_params(RiskParams(lookback=lookback, min_obs=min_obs))


# --- Ledoit-Wolf -----------------------------------------------------------


def test_ledoit_wolf_matches_the_closed_form() -> None:
    # Column means 0.5 and 0.25; demeaned X = [[1, 2], [0, 0], [-1, -2]], T = 3, N = 2.
    # S = X'X / 3 = [[2/3, 4/3], [4/3, 8/3]]; m = tr(S) / 2 = 5/3.
    # S - mI = [[-1, 4/3], [4/3, 1]] -> d2 = (1 + 16/9 + 16/9 + 1) / 2 = 25/9.
    # ||x_t x_t' - S||^2 per row: 25/9, 100/9, 25/9 -> b2_bar = (150/9) / (2 * 9) = 25/27.
    # delta = (25/27) / (25/9) = 1/3; shrunk = (1/3)(5/3) I + (2/3) S.
    returns = np.array([[1.5, 2.25], [0.5, 0.25], [-0.5, -1.75]])
    shrunk, delta = ledoit_wolf(returns)
    assert delta == pytest.approx(1 / 3)
    np.testing.assert_allclose(shrunk, [[1.0, 8 / 9], [8 / 9, 7 / 3]])


def test_ledoit_wolf_delta_is_zero_when_sample_is_scaled_identity() -> None:
    # X'X / 4 = I exactly, so S = m I with m = 1 and d2 = 0.
    returns = np.array([[1.0, 1.0], [1.0, -1.0], [-1.0, 1.0], [-1.0, -1.0]])
    shrunk, delta = ledoit_wolf(returns)
    assert delta == 0.0
    np.testing.assert_allclose(shrunk, np.eye(2))


def test_ledoit_wolf_caps_delta_at_one() -> None:
    # X = [[1, 0], [0, 1], [-1, -1]] (already demeaned): S = [[2/3, 1/3], [1/3, 2/3]],
    # m = 2/3, d2 = (2/9) / 2 = 1/9; per-row norms 7/9, 7/9, 10/9 -> b2_bar = (24/9) / 18 = 4/27.
    # b2_bar > d2, so b2 = d2 and delta = 1: the estimate is m I.
    returns = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]])
    shrunk, delta = ledoit_wolf(returns)
    assert delta == 1.0
    np.testing.assert_allclose(shrunk, (2 / 3) * np.eye(2))


def test_ledoit_wolf_single_name_is_the_sample_variance() -> None:
    # N = 1: S = m I always, so d2 = 0 and delta = 0. Returns [0.01, -0.01]: variance 1e-4.
    shrunk, delta = ledoit_wolf(np.array([[0.01], [-0.01]]))
    assert delta == 0.0
    np.testing.assert_allclose(shrunk, [[1e-4]])


# --- risk_model ------------------------------------------------------------

FOUR = RiskParams(lookback=4, min_obs=4)


def test_beta_of_a_doubled_series_is_two() -> None:
    closes = _closes({"A": _prices([2 * r for r in SPY_RETURNS])})
    model = risk_model(closes, _spy(_prices(SPY_RETURNS)), params=FOUR)
    assert model.betas == {"A": pytest.approx(2.0)}


def test_annualized_variance_over_the_last_lookback_rows() -> None:
    # Seven closes, lookback 4: only the last four returns (closes 2..6) count, so the
    # NaN closes in rows 0 and 1 do not matter. Returns [0.01, -0.01, 0.01, -0.01]:
    # mean 0, variance 1e-4. One name: d2 = 0, delta = 0, sigma = 252 * 1e-4 = 0.0252.
    a = [NAN, NAN, *_prices([0.01, -0.01, 0.01, -0.01])]
    spy = _prices([0.03, -0.01, *SPY_RETURNS])
    model = risk_model(_closes({"A": a}), _spy(spy), params=FOUR)
    assert model.observations == 4
    assert model.delta == 0.0
    assert model.excluded == {}
    assert model.sigma.loc["A", "A"] == pytest.approx(0.0252)
    assert model.vols == {"A": pytest.approx(0.0252**0.5)}


def test_benchmark_dates_absent_from_the_snapshot_are_ignored() -> None:
    closes = _closes({"A": _prices([2 * r for r in SPY_RETURNS])})
    spy = _spy(_prices(SPY_RETURNS))
    weekend = pd.Series([1000.0], index=[pd.Timestamp("2026-09-27")])  # a Sunday
    model = risk_model(closes, pd.concat([spy, weekend]).sort_index(), params=FOUR)
    assert model.betas == {"A": pytest.approx(2.0)}


def test_sigma_and_correlation_frames() -> None:
    # B mirrors A, so their sample correlation is -1; Z never moves, so its variance is 0
    # and every correlation involving Z is NaN (treated as uncorrelated downstream).
    closes = _closes(
        {
            "A": _prices(SPY_RETURNS),
            "B": _prices([-r for r in SPY_RETURNS]),
            "Z": [50.0] * 5,
        }
    )
    model = risk_model(closes, _spy(_prices(SPY_RETURNS)), params=FOUR)
    assert list(model.sigma.index) == ["A", "B", "Z"]
    assert list(model.sigma.columns) == ["A", "B", "Z"]
    np.testing.assert_allclose(model.sigma.to_numpy(), model.sigma.to_numpy().T)
    assert model.corr.loc["A", "B"] == pytest.approx(-1.0)
    assert np.isnan(model.corr.loc["A", "Z"])
    assert np.isnan(model.corr.loc["Z", "Z"])
    assert model.betas == {"A": pytest.approx(1.0), "B": pytest.approx(-1.0), "Z": 0.0}
    assert 0.0 < model.delta < 1.0


def test_short_history_is_excluded() -> None:
    # B's NaN close in row 2 kills returns 2 and 3: 2 valid returns < min_obs 3.
    closes = _closes({"A": _prices(SPY_RETURNS), "B": _with_nan(_prices(SPY_RETURNS), 2)})
    model = risk_model(closes, _spy(_prices(SPY_RETURNS)), params=RiskParams(lookback=4, min_obs=3))
    assert model.excluded == {"B": "history"}
    assert list(model.vols) == ["A"]


SIX = RiskParams(lookback=6, min_obs=4)


def test_alignment_drops_the_alphabetically_last_on_a_tie() -> None:
    # Valid return rows: A 1-6, B 1-4 (NaN close 5), C 3-6 (NaN close 1). All rows shared:
    # 3-4, only 2 < 4. B and C tie on 4 returns, so C (alphabetically last) goes; A and B
    # then share rows 1-4 = 4 rows.
    closes = _closes(
        {
            "A": _prices(WIGGLE),
            "B": _with_nan(_prices(WIGGLE), 5),
            "C": _with_nan(_prices(WIGGLE), 1),
        }
    )
    model = risk_model(closes, _spy(_prices(WIGGLE[::-1])), params=SIX)
    assert model.excluded == {"C": "history"}
    assert list(model.vols) == ["A", "B"]
    assert model.observations == 4


def test_alignment_drops_the_fewest_returns_first() -> None:
    # Valid rows: A 3-6 (4 returns), B 1-5 (5), C 1-6 (6). Shared 3-5 = 3 < 4: drop A even
    # though it sorts first; B and C then share rows 1-5 = 5.
    closes = _closes(
        {
            "A": _with_nan(_prices(WIGGLE), 1),
            "B": _with_nan(_prices(WIGGLE), 6),
            "C": _prices(WIGGLE),
        }
    )
    model = risk_model(closes, _spy(_prices(WIGGLE[::-1])), params=SIX)
    assert model.excluded == {"A": "history"}
    assert model.observations == 5


def test_alignment_repeats_until_enough_rows_are_shared() -> None:
    # Valid rows: A 1-6, B 1-4 (NaN close 5), C 3-6 (NaN close 1), D 1, 2, 5, 6 (NaN close 3).
    # Pass 1: all four share no row; B, C, D tie on 4 returns -> drop D (alphabetically last).
    # Pass 2: A, B, C share rows 3-4 = 2 < 4; B and C tie -> drop C. Pass 3: A, B share 1-4.
    closes = _closes(
        {
            "A": _prices(WIGGLE),
            "B": _with_nan(_prices(WIGGLE), 5),
            "C": _with_nan(_prices(WIGGLE), 1),
            "D": _with_nan(_prices(WIGGLE), 3),
        }
    )
    model = risk_model(closes, _spy(_prices(WIGGLE[::-1])), params=SIX)
    assert model.excluded == {"C": "history", "D": "history"}
    assert list(model.vols) == ["A", "B"]
    assert model.observations == 4


def test_alignment_can_drop_every_name() -> None:
    # SPY valid rows 1, 2, 5, 6 (NaN close 3); A valid rows 3-6. Shared 5-6 = 2 < 4.
    closes = _closes({"A": _with_nan(_prices(WIGGLE), 1)})
    model = risk_model(closes, _spy(_with_nan(_prices(WIGGLE[::-1]), 3)), params=SIX)
    assert model.excluded == {"A": "history"}
    assert model.sigma.empty
    assert model.corr.empty
    assert (model.betas, model.vols, model.delta, model.observations) == ({}, {}, 0.0, 0)


@pytest.mark.parametrize("bad_close", [0.0, -5.0])
def test_non_positive_closes_are_missing(bad_close: float) -> None:
    # A non-positive close in row 3 is missing, so the returns on rows 3 (its own, -100% for
    # zero) and 4 (x / 0 - 1 = inf) are both NaN. Rows 1, 2, 5, 6 remain shared = 4.
    bad_prices = _prices(WIGGLE)
    bad_prices[3] = bad_close
    closes = _closes({"A": bad_prices, "B": _prices(WIGGLE[::-1])})
    model = risk_model(closes, _spy(_prices(WIGGLE[::-1])), params=SIX)
    assert model.excluded == {}
    assert model.observations == 4
    assert np.isfinite(model.sigma.to_numpy()).all()
    assert np.isfinite(model.corr.to_numpy()).all()
    assert np.isfinite(list(model.betas.values())).all()


def test_correlations_are_clipped_so_max_rho_one_never_filters() -> None:
    # Returns that are exact multiples of one another are perfectly correlated; rounding in
    # the sample moments can push the raw ratio a hair past 1, which the clip removes.
    base = [0.013, -0.027, 0.0171, 0.0049, -0.0113, 0.0211]
    closes = _closes(
        {
            "A": _prices(base),
            "B": _prices([3 * r for r in base]),
            "C": _prices([0.7 * r for r in base]),
            "D": _prices([0.1 * r for r in base]),
        }
    )
    model = risk_model(closes, _spy(_prices(WIGGLE[::-1])), params=SIX)
    values = model.corr.to_numpy()
    assert ((values >= -1.0) & (values <= 1.0)).all()
    scores = dict.fromkeys(model.corr.index, 1.0)
    sides = dict.fromkeys(scores, Side.LONG)
    kept, excluded = correlation_filter(
        scores, sides, model.corr, params=CorrelationParams(max_rho=1)
    )
    assert kept == ("A", "B", "C", "D")
    assert excluded == {}


def test_rejects_short_benchmark() -> None:
    # SPY NaN close in row 2: returns 2 and 3 are NaN, leaving 2 < min_obs 4.
    closes = _closes({"A": _prices(SPY_RETURNS)})
    spy = _spy(_with_nan(_prices(SPY_RETURNS), 2))
    with pytest.raises(ValueError, match="benchmark has 2 returns"):
        risk_model(closes, spy, params=FOUR)


def test_rejects_benchmark_without_variance() -> None:
    closes = _closes({"A": _prices(SPY_RETURNS)})
    with pytest.raises(ValueError, match="zero variance"):
        risk_model(closes, _spy([400.0] * 5), params=FOUR)


def test_simple_returns_mask_non_positive_closes() -> None:
    # The 0.0 close is treated as missing, so the returns on both adjacent rows are NaN.
    out = simple_returns(pd.Series([100.0, 110.0, 0.0, 121.0, 133.1]))
    assert np.isnan(out.iloc[[0, 2, 3]]).all()
    assert out.iloc[[1, 4]].tolist() == pytest.approx([0.1, 0.1])
