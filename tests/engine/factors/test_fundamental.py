"""Closed-form checks for fundamental factor definitions."""

import numpy as np
import pandas as pd

from hedge_fund.domain.fundamentals import METRIC_NAMES
from hedge_fund.engine.factors.fundamental import FundamentalInputs, market_cap

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
