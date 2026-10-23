"""Closed-form checks for fundamental factor definitions."""

import warnings

import numpy as np
import pandas as pd

from hedge_fund.domain.fundamentals import METRIC_NAMES
from hedge_fund.engine.factors.fundamental import (
    FUNDAMENTAL_FACTORS,
    FundamentalInputs,
    earnings_yield,
    gross_profitability,
    market_cap,
    revenue_growth,
    size,
)

TICKERS = ["AAA", "BBB", "CCC"]
NAN = np.nan


def _frame(values: dict[str, list[float]] | None) -> pd.DataFrame:
    """METRIC_NAMES frame over TICKERS: all NaN except the given metric columns."""
    frame = pd.DataFrame(NAN, index=TICKERS, columns=list(METRIC_NAMES))
    for metric, column in (values or {}).items():
        frame[metric] = column
    return frame


def _inputs(
    current: dict[str, list[float]] | None = None,
    prior: dict[str, list[float]] | None = None,
    close: list[float] | None = None,
) -> FundamentalInputs:
    """Inputs over TICKERS; close defaults to 10.0 for every ticker."""
    return FundamentalInputs(
        current=_frame(current),
        prior=_frame(prior),
        close=pd.Series(close or [10.0, 10.0, 10.0], index=TICKERS),
    )


def _assert_series(result: pd.Series, expected: list[float]) -> None:
    pd.testing.assert_series_equal(result, pd.Series(expected, index=TICKERS), check_names=False)


def test_market_cap_is_shares_times_close() -> None:
    inputs = _inputs({"shares_outstanding": [5.0, 2.0, 1.5]}, close=[10.0, 20.0, 4.0])
    _assert_series(market_cap(inputs), [50.0, 40.0, 6.0])


def test_market_cap_nonpositive_or_missing_is_nan() -> None:
    inputs = _inputs({"shares_outstanding": [0.0, -3.0, NAN]})
    _assert_series(market_cap(inputs), [NAN, NAN, NAN])


def test_earnings_yield_known_value() -> None:
    inputs = _inputs({"net_income": [5.0, 30.0, 1.0], "shares_outstanding": [10.0, 20.0, 1.0]})
    _assert_series(earnings_yield(inputs), [0.05, 0.15, 0.1])


def test_earnings_yield_negative_income_stays_negative() -> None:
    inputs = _inputs({"net_income": [-20.0, 0.0, 5.0], "shares_outstanding": [10.0, 10.0, 10.0]})
    _assert_series(earnings_yield(inputs), [-0.2, 0.0, 0.05])


def test_earnings_yield_nonpositive_market_cap_is_nan() -> None:
    inputs = _inputs({"net_income": [5.0, 5.0, 5.0], "shares_outstanding": [0.0, -10.0, 10.0]})
    _assert_series(earnings_yield(inputs), [NAN, NAN, 0.05])


def test_gross_profitability_known_value() -> None:
    inputs = _inputs({"gross_profit": [30.0, -10.0, 50.0], "total_assets": [100.0, 200.0, 50.0]})
    _assert_series(gross_profitability(inputs), [0.3, -0.05, 1.0])


def test_gross_profitability_nonpositive_assets_is_nan() -> None:
    inputs = _inputs({"gross_profit": [30.0, 30.0, 30.0], "total_assets": [0.0, -50.0, 100.0]})
    _assert_series(gross_profitability(inputs), [NAN, NAN, 0.3])


def test_gross_profitability_missing_gross_profit_is_nan() -> None:
    inputs = _inputs({"gross_profit": [NAN, 30.0, NAN], "total_assets": [100.0, 100.0, NAN]})
    _assert_series(gross_profitability(inputs), [NAN, 0.3, NAN])


def test_revenue_growth_known_value() -> None:
    inputs = _inputs({"revenue": [120.0, 80.0, 100.0]}, {"revenue": [100.0, 100.0, 100.0]})
    _assert_series(revenue_growth(inputs), [0.2, -0.2, 0.0])


def test_revenue_growth_nonpositive_or_missing_prior_is_nan() -> None:
    inputs = _inputs({"revenue": [100.0, 100.0, 100.0]}, {"revenue": [0.0, -50.0, NAN]})
    _assert_series(revenue_growth(inputs), [NAN, NAN, NAN])


def test_size_is_log_market_cap() -> None:
    inputs = _inputs({"shares_outstanding": [10.0, 20.0, 0.5]})
    _assert_series(size(inputs), [np.log(100.0), np.log(200.0), np.log(5.0)])


def test_size_of_nan_market_cap_is_nan_without_warning() -> None:
    inputs = _inputs({"shares_outstanding": [NAN, 0.0, -1.0]})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = size(inputs)
    _assert_series(result, [NAN, NAN, NAN])


def test_fundamental_factor_registry() -> None:
    assert [spec.name for spec in FUNDAMENTAL_FACTORS] == [
        "earnings_yield",
        "gross_profitability",
        "revenue_growth",
        "size",
    ]
    assert [spec.sign for spec in FUNDAMENTAL_FACTORS] == [1, 1, 1, -1]
    assert [spec.fn for spec in FUNDAMENTAL_FACTORS] == [
        earnings_yield,
        gross_profitability,
        revenue_growth,
        size,
    ]
