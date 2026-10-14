"""Tests for the Piotroski and Altman quality scores."""

import numpy as np
import pandas as pd
import pytest

from hedge_fund.engine.factors.fundamental import FundamentalInputs
from hedge_fund.engine.factors.quality import (
    DEFAULT_GATES,
    PIOTROSKI_SIGNALS,
    QualityGates,
    altman_z,
    piotroski,
    piotroski_signals,
)

# Base values pass all nine signals; Altman Z = 3.275 (see test_altman_z_known_value).
CURRENT = {
    "net_income": 100.0,
    "operating_cash_flow": 150.0,
    "total_assets": 1000.0,
    "long_term_debt": 100.0,
    "current_assets": 500.0,
    "current_liabilities": 200.0,
    "shares_outstanding": 100.0,
    "gross_profit": 400.0,
    "revenue": 800.0,
    "retained_earnings": 300.0,
    "operating_income": 150.0,
    "total_liabilities": 500.0,
}
PRIOR = {
    "net_income": 80.0,
    "operating_cash_flow": 120.0,
    "total_assets": 1000.0,
    "long_term_debt": 200.0,
    "current_assets": 400.0,
    "current_liabilities": 250.0,
    "shares_outstanding": 100.0,
    "gross_profit": 300.0,
    "revenue": 700.0,
    "retained_earnings": 250.0,
    "operating_income": 120.0,
    "total_liabilities": 500.0,
}
CLOSE = 10.0
NAN = float("nan")


def _inputs(
    current: dict[str, float] | None = None,
    prior: dict[str, float] | None = None,
    close: float = CLOSE,
) -> FundamentalInputs:
    """One ticker ``T`` from the base values with ``current`` / ``prior`` overrides."""
    return FundamentalInputs(
        current=pd.DataFrame([CURRENT | (current or {})], index=["T"]),
        prior=pd.DataFrame([PRIOR | (prior or {})], index=["T"]),
        close=pd.Series({"T": close}),
    )


def _signal(name: str, current: dict[str, float], prior: dict[str, float]) -> int:
    return int(piotroski_signals(_inputs(current, prior)).at["T", name])


def test_signal_names_pinned() -> None:
    assert PIOTROSKI_SIGNALS == (
        "roa_positive",
        "cfo_positive",
        "roa_improved",
        "accruals",
        "leverage_not_up",
        "current_ratio_up",
        "no_dilution",
        "gross_margin_up",
        "asset_turnover_up",
    )


def test_base_inputs_pass_every_signal() -> None:
    signals = piotroski_signals(_inputs())
    assert list(signals.columns) == list(PIOTROSKI_SIGNALS)
    assert (signals.dtypes == "int64").all()
    assert signals.loc["T"].tolist() == [1] * 9
    f_score = piotroski(_inputs())
    assert f_score.dtype == "int64"
    assert f_score["T"] == 9


@pytest.mark.parametrize(
    ("name", "current", "prior"),
    [
        ("roa_positive", {"net_income": -1.0}, {}),
        ("cfo_positive", {"operating_cash_flow": -1.0}, {}),
        ("roa_improved", {"net_income": 80.0}, {}),  # equal ROA is not an improvement
        ("accruals", {"operating_cash_flow": 100.0}, {}),  # CFO == NI is not CFO > NI
        ("leverage_not_up", {"long_term_debt": 300.0}, {}),
        ("current_ratio_up", {"current_assets": 320.0}, {}),  # 1.6 == prior 1.6
        ("no_dilution", {"shares_outstanding": 101.0}, {}),
        ("gross_margin_up", {"gross_profit": 300.0}, {}),
        ("asset_turnover_up", {"revenue": 700.0}, {}),  # 0.7 == prior 0.7
    ],
)
def test_signal_false_case(name: str, current: dict[str, float], prior: dict[str, float]) -> None:
    assert _signal(name, current, prior) == 0


@pytest.mark.parametrize(
    ("name", "current", "prior"),
    [
        ("roa_positive", {"net_income": NAN}, {}),
        ("cfo_positive", {"operating_cash_flow": NAN}, {}),
        ("roa_improved", {}, {"net_income": NAN}),
        ("accruals", {"operating_cash_flow": NAN}, {}),
        ("leverage_not_up", {}, {"total_assets": NAN}),
        ("current_ratio_up", {}, {"current_liabilities": NAN}),
        ("no_dilution", {}, {"shares_outstanding": NAN}),
        ("gross_margin_up", {}, {"revenue": NAN}),
        ("asset_turnover_up", {}, {"total_assets": NAN}),
    ],
)
def test_nan_input_scores_zero(
    name: str, current: dict[str, float], prior: dict[str, float]
) -> None:
    assert _signal(name, current, prior) == 0


def test_nonpositive_total_assets_scores_zero() -> None:
    signals = piotroski_signals(_inputs(current={"total_assets": 0.0})).loc["T"]
    assert signals["roa_positive"] == 0
    assert signals["leverage_not_up"] == 0
    assert signals["asset_turnover_up"] == 0


def test_nonpositive_current_liabilities_scores_zero() -> None:
    assert _signal("current_ratio_up", {"current_liabilities": 0.0}, {}) == 0


def test_nonpositive_prior_total_assets_scores_zero() -> None:
    signals = piotroski_signals(_inputs(prior={"total_assets": 0.0})).loc["T"]
    assert signals["roa_improved"] == 0
    assert signals["leverage_not_up"] == 0
    assert signals["asset_turnover_up"] == 0


def test_debt_free_both_years_keeps_leverage_point() -> None:
    assert _signal("leverage_not_up", {"long_term_debt": 0.0}, {"long_term_debt": 0.0}) == 1


def test_nan_long_term_debt_counts_as_zero() -> None:
    assert _signal("leverage_not_up", {"long_term_debt": NAN}, {"long_term_debt": NAN}) == 1
    assert _signal("leverage_not_up", {"long_term_debt": NAN}, {}) == 1
    assert _signal("leverage_not_up", {}, {"long_term_debt": NAN}) == 0


def test_equal_shares_is_no_dilution() -> None:
    assert _signal("no_dilution", {"shares_outstanding": 100.0}, {}) == 1


def test_no_prior_period_caps_score_at_three() -> None:
    no_prior = dict.fromkeys(PRIOR, NAN)
    assert piotroski(_inputs(prior=no_prior))["T"] == 3


def test_piotroski_is_row_sum_of_signals() -> None:
    inputs = _inputs(current={"net_income": -1.0})  # loses roa_positive and roa_improved
    assert piotroski(inputs)["T"] == 7


def test_quality_gate_defaults() -> None:
    assert QualityGates() == QualityGates(min_piotroski=4, min_altman_z=1.81)
    assert QualityGates() == DEFAULT_GATES


def test_altman_z_known_value() -> None:
    # 1.2*300/1000 + 1.4*300/1000 + 3.3*150/1000 + 0.6*(100*10)/500 + 1.0*800/1000
    expected = 0.36 + 0.42 + 0.495 + 1.2 + 0.8
    assert altman_z(_inputs())["T"] == pytest.approx(expected)


def test_altman_z_negative_working_capital_allowed() -> None:
    z = altman_z(_inputs(current={"current_assets": 100.0}))["T"]  # WC = -100
    assert z == pytest.approx(3.275 - 1.2 * 0.4)


@pytest.mark.parametrize(
    ("current", "close"),
    [
        ({"retained_earnings": NAN}, CLOSE),
        ({"total_assets": 0.0}, CLOSE),
        ({"total_liabilities": 0.0}, CLOSE),
        ({}, 0.0),  # market cap <= 0
    ],
)
def test_altman_z_not_computable_is_nan(current: dict[str, float], close: float) -> None:
    assert np.isnan(altman_z(_inputs(current=current, close=close))["T"])


def test_altman_z_dtype_is_float() -> None:
    assert altman_z(_inputs()).dtype == "float64"
