"""Curated fundamental metrics: canonical names, kinds and XBRL tags in fallback order."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

MetricKind = Literal["flow", "stock"]

SHARES_DEI_TAG = "dei:EntityCommonStockSharesOutstanding"
REPORT_FORMS: frozenset[str] = frozenset({"10-K", "10-Q", "10-K/A", "10-Q/A", "10-KT", "10-KT/A"})


@dataclass(frozen=True)
class MetricSpec:
    """A canonical metric with XBRL tags in fallback order.

    ``kind`` is "flow" (trailing twelve months) or "stock" (balance-sheet instant). Fallback
    tags share one definition; tags that change meaning are excluded.
    """

    name: str
    tags: tuple[str, ...]
    kind: MetricKind


def _gaap(*names: str) -> tuple[str, ...]:
    return tuple(f"us-gaap:{name}" for name in names)


METRICS: tuple[MetricSpec, ...] = (
    MetricSpec(
        "revenue",
        _gaap(
            "Revenues",
            "RevenueFromContractWithCustomerExcludingAssessedTax",
            "RevenueFromContractWithCustomerIncludingAssessedTax",
            "SalesRevenueNet",
        ),
        "flow",
    ),
    MetricSpec("cost_of_revenue", _gaap("CostOfRevenue", "CostOfGoodsAndServicesSold"), "flow"),
    MetricSpec("gross_profit", _gaap("GrossProfit"), "flow"),
    MetricSpec("operating_income", _gaap("OperatingIncomeLoss"), "flow"),
    MetricSpec("net_income", _gaap("NetIncomeLoss"), "flow"),
    MetricSpec("operating_cash_flow", _gaap("NetCashProvidedByUsedInOperatingActivities"), "flow"),
    MetricSpec("capex", _gaap("PaymentsToAcquirePropertyPlantAndEquipment"), "flow"),
    MetricSpec("interest_expense", _gaap("InterestExpense"), "flow"),
    MetricSpec(
        "depreciation_amortization",
        _gaap("DepreciationDepletionAndAmortization", "DepreciationAndAmortization"),
        "flow",
    ),
    MetricSpec("total_assets", _gaap("Assets"), "stock"),
    MetricSpec("current_assets", _gaap("AssetsCurrent"), "stock"),
    MetricSpec("cash", _gaap("CashAndCashEquivalentsAtCarryingValue"), "stock"),
    MetricSpec("current_liabilities", _gaap("LiabilitiesCurrent"), "stock"),
    MetricSpec("total_liabilities", _gaap("Liabilities"), "stock"),
    MetricSpec("liabilities_and_equity", _gaap("LiabilitiesAndStockholdersEquity"), "stock"),
    MetricSpec("long_term_debt", _gaap("LongTermDebtNoncurrent"), "stock"),
    MetricSpec("debt_current", _gaap("LongTermDebtCurrent", "DebtCurrent"), "stock"),
    MetricSpec(
        "stockholders_equity",
        _gaap(
            "StockholdersEquity",
            "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
        ),
        "stock",
    ),
    MetricSpec("retained_earnings", _gaap("RetainedEarningsAccumulatedDeficit"), "stock"),
    MetricSpec(
        "shares_outstanding",
        (SHARES_DEI_TAG, *_gaap("CommonStockSharesOutstanding")),
        "stock",
    ),
)

CURATED_TAGS: tuple[str, ...] = tuple(tag for spec in METRICS for tag in spec.tags)


def tags_hash(tags: Sequence[str]) -> str:
    """SHA-256 hex digest of the ordered tag list; changes when tags or their order change."""
    return hashlib.sha256("\n".join(tags).encode("utf-8")).hexdigest()


TAGS_HASH = tags_hash(CURATED_TAGS)
