"""The membership coverage of a frozen store over a run window.

Builds the inputs of the pure engine/backtest/coverage.py from the store's own frames.
"""

from __future__ import annotations

from datetime import date

from hedge_fund.domain.backtest import FrozenData
from hedge_fund.engine.backtest.calendar import trading_days
from hedge_fund.engine.backtest.coverage import CoverageReport, measure_coverage


def store_coverage(data: FrozenData, start: date, end: date) -> CoverageReport:
    """Member-days of ``data.membership`` over the store's trading days in ``start..end``,
    priced where the ticker has a non-NaN close and flagged where ``data.flagged`` covers."""
    closes = data.closes
    priced = {
        str(ticker): {stamp.date() for stamp in closes.index[closes[ticker].notna().to_numpy()]}
        for ticker in closes.columns
    }
    return measure_coverage(
        data.membership, trading_days(closes.index), priced, data.flagged, start, end
    )
