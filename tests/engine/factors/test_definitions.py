"""Closed-form checks for factor definitions."""

import numpy as np
import pandas as pd
import pytest

from hedge_fund.engine.factors.definitions import (
    FACTORS,
    high_52w_proximity,
    low_vol,
    momentum_12_1,
    reversal_1m,
)

WINDOW = 253  # 252 daily returns


def _constant_growth(rate: float) -> pd.DataFrame:
    return pd.DataFrame({"AAA": 100.0 * (1 + rate) ** np.arange(WINDOW)})


def test_momentum_constant_growth() -> None:
    result = momentum_12_1(_constant_growth(0.001))
    assert result["AAA"] == pytest.approx(1.001**231 - 1)


def test_momentum_uses_rows_t_minus_252_and_t_minus_21() -> None:
    prices = np.full(WINDOW, 7.0)
    prices[0] = 50.0
    prices[231] = 100.0
    result = momentum_12_1(pd.DataFrame({"AAA": prices}))
    assert result["AAA"] == pytest.approx(1.0)


def test_low_vol_is_zero_for_constant_growth() -> None:
    result = low_vol(_constant_growth(0.001))
    assert result["AAA"] == pytest.approx(0.0, abs=1e-12)


def test_low_vol_alternating_returns() -> None:
    step = 0.02
    log_prices = np.log(100.0) + step * (np.arange(WINDOW) % 2)
    result = low_vol(pd.DataFrame({"AAA": np.exp(log_prices)}))
    assert result["AAA"] == pytest.approx(-step * np.sqrt(252))


def test_reversal_constant_growth() -> None:
    result = reversal_1m(_constant_growth(0.001))
    assert result["AAA"] == pytest.approx(1.001**21 - 1)


def test_reversal_uses_rows_t_and_t_minus_21() -> None:
    prices = np.full(WINDOW, 7.0)
    prices[-1] = 110.0
    prices[-1 - 21] = 100.0
    result = reversal_1m(pd.DataFrame({"AAA": prices}))
    assert result["AAA"] == pytest.approx(0.1)


def test_reversal_min_history_boundary() -> None:
    rows = 22
    assert np.isfinite(reversal_1m(_constant_growth(0.001).iloc[:rows])["AAA"])
    with pytest.raises(IndexError):
        reversal_1m(_constant_growth(0.001).iloc[: rows - 1])


def test_high_52w_proximity_rise_then_fall() -> None:
    prices = np.concatenate(
        [np.linspace(100.0, 200.0, 150), np.linspace(200.0, 150.0, WINDOW - 150)]
    )
    result = high_52w_proximity(pd.DataFrame({"AAA": prices}))
    assert result["AAA"] == pytest.approx(150.0 / 200.0)


def test_high_52w_proximity_ignores_peak_outside_last_252_rows() -> None:
    prices = np.full(WINDOW, 100.0)
    prices[0] = 500.0
    result = high_52w_proximity(pd.DataFrame({"AAA": prices}))
    assert result["AAA"] == pytest.approx(1.0)


def test_high_52w_proximity_at_high_is_exactly_one() -> None:
    result = high_52w_proximity(_constant_growth(0.001))
    assert result["AAA"] == 1.0


def test_high_52w_proximity_short_window_is_not_guarded_by_function() -> None:
    # iloc[-252:] silently takes all 251 rows; the boundary lives in min_history.
    prices = pd.DataFrame({"AAA": np.linspace(100.0, 50.0, 251)})
    assert high_52w_proximity(prices)["AAA"] == pytest.approx(0.5)
    spec = next(s for s in FACTORS if s.name == "high_52w_proximity")
    assert spec.min_history == 252


def test_reversal_sign_ranks_recent_loser_above_recent_winner() -> None:
    closes = pd.DataFrame(100.0, index=pd.RangeIndex(WINDOW), columns=["L", "W", "M"])
    closes.iloc[-1] = [90.0, 110.0, 100.0]
    spec = next(s for s in FACTORS if s.name == "reversal_1m")
    raw = reversal_1m(closes)
    assert raw["L"] < raw["W"]
    assert spec.sign * raw["L"] > spec.sign * raw["W"]


def test_factor_registry() -> None:
    assert [spec.name for spec in FACTORS] == [
        "momentum_12_1",
        "low_vol",
        "reversal_1m",
        "high_52w_proximity",
    ]
    assert [spec.sign for spec in FACTORS] == [1, 1, -1, 1]
    assert [spec.min_history for spec in FACTORS] == [WINDOW, WINDOW, 22, 252]
