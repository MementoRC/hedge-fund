"""Point-in-time fundamentals vocabulary: metric names and period columns."""

from __future__ import annotations

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
