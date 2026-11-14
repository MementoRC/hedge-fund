"""Research & attribution page: the latest research per name, thesis verification per
horizon, attribution by name and sector, and LLM cost across cycles.

On a demo or backtest root (``RootState.kind`` from ``classify_root``) attribution and
verification cannot exist, and the page says why.
"""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from hedge_fund.dashboard.format import fmt_date, fmt_money, fmt_number, fmt_percent
from hedge_fund.domain.attribution import AttributionTotals
from hedge_fund.viz.load import RootKind
from hedge_fund.viz.views import ResearchView

NAME_COLUMNS = [
    "Ticker",
    "Sector",
    "Side",
    "Verdict",
    "Mean rating",
    "Dispersion",
    "Agreement",
    "Valid runs",
]
HORIZON_COLUMNS = ["Horizon", "Scored", "Hit rate", "IC (rank correlation)"]
ATTRIBUTION_COLUMNS = ["Market", "Sector", "Residual", "Trading", "Total"]
COST_COLUMNS = ["Cycle", "LLM budget", "LLM spent"]


def names_table(view: ResearchView) -> pd.DataFrame:
    # Pre-formatted strings (NaN reads "n/a"), so sorting the number columns is lexical.
    return pd.DataFrame(
        [
            {
                "Ticker": row.ticker,
                "Sector": row.sector,
                "Side": row.side.value,
                "Verdict": row.verdict.value,
                "Mean rating": fmt_number(row.mean),
                "Dispersion": fmt_number(row.dispersion),
                "Agreement": fmt_number(row.agreement),
                "Valid runs": row.valid,
            }
            for row in view.names
        ],
        columns=NAME_COLUMNS,
    )


def horizons_table(view: ResearchView) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Horizon": row.horizon,
                "Scored": row.scored,
                "Hit rate": fmt_percent(row.hit_rate),
                "IC (rank correlation)": fmt_number(row.ic),
            }
            for row in view.horizons
        ],
        columns=HORIZON_COLUMNS,
    )


def attribution_table(totals: Mapping[str, AttributionTotals], label: str) -> pd.DataFrame:
    """One row per key of ``totals`` (``label`` names the key column and must not be one of
    ``ATTRIBUTION_COLUMNS``), P&L in dollars.
    """
    return pd.DataFrame(
        [
            {
                label: key,
                "Market": fmt_money(part.market),
                "Sector": fmt_money(part.sector),
                "Residual": fmt_money(part.residual),
                "Trading": fmt_money(part.trading),
                "Total": fmt_money(part.total),
            }
            for key, part in totals.items()
        ],
        columns=[label, *ATTRIBUTION_COLUMNS],
    )


def cost_table(view: ResearchView) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Cycle": fmt_date(row.as_of),
                "LLM budget": fmt_money(row.budget),
                "LLM spent": fmt_money(row.spent),
            }
            for row in view.costs
        ],
        columns=COST_COLUMNS,
    )


def gap_message(kind: RootKind | None, what: str, command: str) -> str:
    """Why ``what`` (verification or attribution) is absent, by ``classify_root``'s kind."""
    if kind is None:
        return f"No completed {what} in this root yet; `{command}` writes one."
    return (
        f"No {what}: this is a {kind.value} root. A backtest keeps its fills in memory, so "
        "`hedge-fund settle`, `hedge-fund attribute` and `hedge-fund verify` cannot run on it."
    )
