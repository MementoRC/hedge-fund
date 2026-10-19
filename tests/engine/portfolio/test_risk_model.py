"""Tests for the risk model: params, Ledoit-Wolf shrinkage, returns window, beta and vol."""

import dataclasses

import numpy as np
import pandas as pd
import pytest

from hedge_fund.engine.portfolio.risk_model import (
    DEFAULT_RISK,
    RiskParams,
    _validate_params,
    ledoit_wolf,
    risk_model,
)

NAN = float("nan")
NO_PRICES = pd.DataFrame()
NO_BENCHMARK = pd.Series(dtype=float)


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
    _validate_params(RiskParams(lookback=lookback, min_obs=min_obs))


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
