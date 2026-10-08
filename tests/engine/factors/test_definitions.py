"""Closed-form checks for factor definitions."""

import numpy as np
import pandas as pd
import pytest

from hedge_fund.engine.factors.definitions import FACTORS, low_vol, momentum_12_1

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


def test_factor_registry() -> None:
    assert [spec.name for spec in FACTORS] == ["momentum_12_1", "low_vol"]
    assert {spec.min_history for spec in FACTORS} == {WINDOW}
    assert {spec.sign for spec in FACTORS} == {1}
