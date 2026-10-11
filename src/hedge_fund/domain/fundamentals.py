"""Point-in-time fundamentals vocabulary: metric names and period columns."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

METRIC_NAMES: tuple[str, ...] = (
    "revenue",
    "cost_of_revenue",
    "gross_profit",
    "operating_income",
    "net_income",
    "operating_cash_flow",
    "capex",
    "interest_expense",
    "depreciation_amortization",
    "total_assets",
    "current_assets",
    "cash",
    "current_liabilities",
    "total_liabilities",
    "liabilities_and_equity",
    "long_term_debt",
    "debt_current",
    "stockholders_equity",
    "retained_earnings",
    "shares_outstanding",
)
PERIOD_COLUMNS: tuple[str, ...] = ("current_end", "prior_end", "last_filed")


@dataclass(frozen=True, eq=False)
class FundamentalsSnapshot:
    """Metrics per ticker at E (``current``) and E′ (``prior``), plus ``periods`` dates.

    ``periods`` holds current_end, prior_end (NaT when no prior period) and last_filed.
    NaN metric cells are allowed. Frozen only prevents field reassignment; consumers must
    never mutate the frames.
    """

    as_of: date
    current: pd.DataFrame
    prior: pd.DataFrame
    periods: pd.DataFrame

    def __post_init__(self) -> None:
        frames = {"current": self.current, "prior": self.prior, "periods": self.periods}
        for name, frame in frames.items():
            if frame.empty:
                raise ValueError(f"{name} is empty")
        _check_index(self.current.index, self.prior.index, self.periods.index)
        for name in ("current", "prior"):
            if list(frames[name].columns) != list(METRIC_NAMES):
                raise ValueError(f"{name} columns must be METRIC_NAMES, in order")
        if list(self.periods.columns) != list(PERIOD_COLUMNS):
            raise ValueError(f"periods columns must be {list(PERIOD_COLUMNS)}")
        if not all(pd.api.types.is_datetime64_dtype(dtype) for dtype in self.periods.dtypes):
            raise ValueError("periods must be datetime64")
        try:
            metrics = np.concatenate(
                [self.current.to_numpy(dtype="float64"), self.prior.to_numpy(dtype="float64")]
            )
        except (TypeError, ValueError) as error:
            raise ValueError("current and prior must be numeric") from error
        if np.isinf(metrics).any():
            raise ValueError("metric values must be finite or NaN, not inf")


def _check_index(index: pd.Index, *others: pd.Index) -> None:
    if not all(other.equals(index) for other in others):
        raise ValueError("current, prior and periods must share the same index")
    if not index.is_monotonic_increasing:
        raise ValueError("index must be sorted ascending")
    if index.has_duplicates:
        raise ValueError("index must be unique")
