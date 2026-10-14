"""Quality gates: Piotroski F-score and Altman Z-score from fundamentals and close."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from hedge_fund.engine.factors.fundamental import FundamentalInputs


@dataclass(frozen=True)
class QualityGates:
    """Thresholds. A ticker passes when F >= min_piotroski and Z >= min_altman_z."""

    min_piotroski: int = 4
    min_altman_z: float = 1.81


DEFAULT_GATES = QualityGates()

PIOTROSKI_SIGNALS: tuple[str, ...] = (
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


def _ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """``numerator / denominator``, NaN where the denominator is <= 0 or NaN."""
    return numerator / denominator.where(denominator > 0)


def piotroski_signals(inputs: FundamentalInputs) -> pd.DataFrame:
    """The nine Piotroski signals as 0/1 ints; a signal whose inputs are NaN scores 0.

    Uses end-of-period assets for both years. Leverage is long-term debt only, NaN as 0.
    """
    cur, pri = inputs.current, inputs.prior
    roa = _ratio(cur["net_income"], cur["total_assets"])
    prior_roa = _ratio(pri["net_income"], pri["total_assets"])
    leverage = _ratio(cur["long_term_debt"].fillna(0.0), cur["total_assets"])
    prior_leverage = _ratio(pri["long_term_debt"].fillna(0.0), pri["total_assets"])
    signals = pd.DataFrame(
        {
            "roa_positive": roa > 0,
            "cfo_positive": cur["operating_cash_flow"] > 0,
            "roa_improved": roa > prior_roa,
            "accruals": cur["operating_cash_flow"] > cur["net_income"],
            "leverage_not_up": leverage <= prior_leverage,
            "current_ratio_up": _ratio(cur["current_assets"], cur["current_liabilities"])
            > _ratio(pri["current_assets"], pri["current_liabilities"]),
            "no_dilution": cur["shares_outstanding"] <= pri["shares_outstanding"],
            "gross_margin_up": _ratio(cur["gross_profit"], cur["revenue"])
            > _ratio(pri["gross_profit"], pri["revenue"]),
            "asset_turnover_up": _ratio(cur["revenue"], cur["total_assets"])
            > _ratio(pri["revenue"], pri["total_assets"]),
        },
        columns=list(PIOTROSKI_SIGNALS),
    )
    return signals.astype("int64")


def piotroski(inputs: FundamentalInputs) -> pd.Series:
    """Piotroski F-score, 0..9: the row sum of ``piotroski_signals``."""
    return piotroski_signals(inputs).sum(axis=1)
