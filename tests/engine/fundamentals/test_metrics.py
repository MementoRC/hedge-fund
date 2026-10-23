"""Tests for the curated fundamental metric definitions."""

import re

from hedge_fund.domain.fundamentals import METRIC_NAMES, PERIOD_COLUMNS
from hedge_fund.engine.fundamentals.metrics import (
    CURATED_TAGS,
    METRICS,
    REPORT_FORMS,
    SHARES_DEI_TAG,
    TAGS_HASH,
    tags_hash,
)

EXPECTED: dict[str, tuple[str, tuple[str, ...]]] = {
    "revenue": (
        "flow",
        (
            "Revenues",
            "RevenueFromContractWithCustomerExcludingAssessedTax",
            "RevenueFromContractWithCustomerIncludingAssessedTax",
            "SalesRevenueNet",
        ),
    ),
    "cost_of_revenue": ("flow", ("CostOfRevenue", "CostOfGoodsAndServicesSold")),
    "gross_profit": ("flow", ("GrossProfit",)),
    "operating_income": ("flow", ("OperatingIncomeLoss",)),
    "net_income": ("flow", ("NetIncomeLoss",)),
    "operating_cash_flow": ("flow", ("NetCashProvidedByUsedInOperatingActivities",)),
    "capex": ("flow", ("PaymentsToAcquirePropertyPlantAndEquipment",)),
    "interest_expense": ("flow", ("InterestExpense",)),
    "depreciation_amortization": (
        "flow",
        ("DepreciationDepletionAndAmortization", "DepreciationAndAmortization"),
    ),
    "total_assets": ("stock", ("Assets",)),
    "current_assets": ("stock", ("AssetsCurrent",)),
    "cash": ("stock", ("CashAndCashEquivalentsAtCarryingValue",)),
    "current_liabilities": ("stock", ("LiabilitiesCurrent",)),
    "total_liabilities": ("stock", ("Liabilities",)),
    "liabilities_and_equity": ("stock", ("LiabilitiesAndStockholdersEquity",)),
    "long_term_debt": ("stock", ("LongTermDebtNoncurrent",)),
    "debt_current": ("stock", ("LongTermDebtCurrent", "DebtCurrent")),
    "stockholders_equity": (
        "stock",
        (
            "StockholdersEquity",
            "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
        ),
    ),
    "retained_earnings": ("stock", ("RetainedEarningsAccumulatedDeficit",)),
    "shares_outstanding": (
        "stock",
        ("dei:EntityCommonStockSharesOutstanding", "CommonStockSharesOutstanding"),
    ),
}


def _prefixed(tag: str) -> str:
    return tag if ":" in tag else f"us-gaap:{tag}"


def test_metric_table_names_match_the_domain_metric_names() -> None:
    assert tuple(m.name for m in METRICS) == METRIC_NAMES


def test_metrics_match_the_spec_table_in_order() -> None:
    assert tuple(EXPECTED) == METRIC_NAMES
    for spec in METRICS:
        kind, tags = EXPECTED[spec.name]
        assert spec.kind == kind
        assert spec.tags == tuple(_prefixed(tag) for tag in tags)


def test_there_are_twenty_metrics_nine_flow_and_eleven_stock() -> None:
    assert len(METRICS) == 20
    assert sum(spec.kind == "flow" for spec in METRICS) == 9
    assert sum(spec.kind == "stock" for spec in METRICS) == 11


def test_curated_tags_are_prefixed_unique_and_in_metric_order() -> None:
    assert tuple(tag for spec in METRICS for tag in spec.tags) == CURATED_TAGS
    assert len(set(CURATED_TAGS)) == len(CURATED_TAGS)
    assert all(tag.startswith(("us-gaap:", "dei:")) for tag in CURATED_TAGS)
    assert [tag for tag in CURATED_TAGS if tag.startswith("dei:")] == [SHARES_DEI_TAG]


def test_tags_hash_is_stable_and_order_sensitive() -> None:
    assert tags_hash(CURATED_TAGS) == TAGS_HASH
    assert re.fullmatch(r"[0-9a-f]{64}", TAGS_HASH)
    assert tags_hash(CURATED_TAGS[::-1]) != TAGS_HASH


def test_report_forms_and_period_columns() -> None:
    assert {"10-K", "10-Q", "10-K/A", "10-Q/A", "10-KT", "10-KT/A"} == REPORT_FORMS
    assert PERIOD_COLUMNS == ("current_end", "prior_end", "last_filed")
