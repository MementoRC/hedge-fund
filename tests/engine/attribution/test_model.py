"""day_model: betas over the trailing window ending at t-1, then a one-row regression at t."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from hedge_fund.engine.attribution.model import day_model
from hedge_fund.engine.portfolio.risk_model import RiskParams

NAN = float("nan")
SPY = [0.5, 0.01, 0.01, 0.01, -0.02, 0.015, 0.02]


def _scaled(beta: float, on_day: float) -> list[float]:
    """A wild row 0, ``beta`` times SPY on the window rows 1-5, then ``on_day`` on DAY."""
    return [-0.9, *(beta * s for s in SPY[1:6]), on_day]


RETURNS = pd.DataFrame(
    {
        "A1": _scaled(2.0, 0.022),
        "A2": _scaled(0.5, 0.007),
        "B1": _scaled(1.0, 0.009),
        "B2": _scaled(1.5, 0.014),
        "N1": _scaled(1.0, 0.3),
        "N2": _scaled(0.8, 0.4),
        "SHORT1": [NAN, NAN, NAN, NAN, 0.01, 0.02, 0.5],
        "Z1": [NAN, 0.03, -0.01, 0.02, NAN, NAN, 0.3],
        "OUT": _scaled(3.0, 0.9),
    },
    index=pd.bdate_range(end="2026-09-30", periods=7),
)
BENCHMARK = pd.Series(SPY, index=RETURNS.index)
SECTORS: dict[str, str | None] = {
    "A1": "S1",
    "A2": "S1",
    "B1": "S2",
    "B2": "S2",
    "N1": None,
    "N2": None,
    "SHORT1": "S1",
    "Z1": "S2",
    "OUT": "S1",
}
UNIVERSE = ("A1", "A2", "B1", "B2", "N1", "N2", "SHORT1", "Z1")
PARAMS = RiskParams(lookback=5, min_obs=3)
DAY = date(2026, 9, 30)


def test_betas_come_from_the_prior_window_only() -> None:
    model = day_model(RETURNS, BENCHMARK, SECTORS, UNIVERSE, PARAMS, DAY)
    assert model is not None
    assert model.betas == pytest.approx({"A1": 2.0, "A2": 0.5, "B1": 1.0, "B2": 1.5})


def test_the_day_regression_recovers_the_factor_returns() -> None:
    model = day_model(RETURNS, BENCHMARK, SECTORS, UNIVERSE, PARAMS, DAY)
    assert model is not None
    assert model.market == pytest.approx(0.01)
    assert model.sector_returns == pytest.approx({"S1": 0.002, "S2": -0.001})
    assert model.sectors == {"A1": "S1", "A2": "S1", "B1": "S2", "B2": "S2"}
    assert model.residuals == pytest.approx({"A1": 0.0, "A2": 0.0, "B1": 0.0, "B2": 0.0}, abs=1e-12)
    on_day = RETURNS.iloc[-1]
    for ticker, beta in model.betas.items():
        rebuilt = (
            beta * model.market
            + model.sector_returns[model.sectors[ticker]]
            + model.residuals[ticker]
        )
        assert rebuilt == pytest.approx(on_day[ticker], abs=1e-12)


def test_unusable_names_are_left_out_without_unregressing_the_day() -> None:
    model = day_model(RETURNS, BENCHMARK, SECTORS, UNIVERSE, PARAMS, DAY)
    assert model is not None
    # N1/N2: no sector; SHORT1: too few returns; Z1: flat benchmark rows; OUT: not universe.
    assert sorted(model.betas) == ["A1", "A2", "B1", "B2"]
    assert sorted(model.residuals) == ["A1", "A2", "B1", "B2"]


def test_no_usable_name_leaves_the_day_unregressed() -> None:
    no_sectors = dict.fromkeys(SECTORS)
    assert day_model(RETURNS, BENCHMARK, no_sectors, UNIVERSE, PARAMS, DAY) is None


def test_a_day_the_regression_cannot_use_is_unregressed() -> None:
    returns = RETURNS.copy()
    returns.loc[returns.index[-1], "A2"] = NAN  # S1 keeps one name with a return on DAY
    assert day_model(returns, BENCHMARK, SECTORS, UNIVERSE, PARAMS, DAY) is None


def test_sectors_short_of_names_leave_the_day_unregressed() -> None:
    assert day_model(RETURNS, BENCHMARK, SECTORS, ("A1", "B1"), PARAMS, DAY) is None
