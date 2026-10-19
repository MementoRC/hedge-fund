"""Tests for the market-neutral mean-variance optimizer (synthetic sigma and alpha)."""

import dataclasses

import numpy as np
import pandas as pd
import pytest

from hedge_fund.domain.shortlist import Side
from hedge_fund.engine.portfolio.optimize import (
    DEFAULT_OPTIMIZE,
    Allocation,
    Asset,
    OptimizeParams,
    PortfolioInfeasibleError,
    optimize_weights,
)

NAN = float("nan")
INF = float("inf")
LONG = Side.LONG
SHORT = Side.SHORT


def _asset(ticker: str, sector: str, side: Side, alpha: float = 0.0, beta: float = 1.0) -> Asset:
    return Asset(ticker=ticker, sector=sector, side=side, alpha=alpha, beta=beta)


def _sigma(assets: list[Asset], var: float = 0.04) -> pd.DataFrame:
    """Diagonal covariance (20% vol per name) labelled by ticker."""
    tickers = [asset.ticker for asset in assets]
    return pd.DataFrame(var * np.eye(len(tickers)), index=tickers, columns=tickers)


def _paired(n: int, alphas: list[float] | None = None) -> list[Asset]:
    """``n`` longs L1..Ln and ``n`` shorts S1..Sn; Lk and Sk share sector ``Sec<k>``."""
    values = alphas or [0.01] * n
    longs = [_asset(f"L{k}", f"Sec{k}", LONG, values[k - 1]) for k in range(1, n + 1)]
    shorts = [_asset(f"S{k}", f"Sec{k}", SHORT, -values[k - 1]) for k in range(1, n + 1)]
    return longs + shorts


# --- params ----------------------------------------------------------------


def test_default_params() -> None:
    assert (
        OptimizeParams(
            risk_aversion=1.0,
            gross=2.0,
            max_weight=0.10,
            min_names_per_side=3,
            beta_band=0.10,
            sector_net=0.10,
            sector_gross=0.50,
        )
        == DEFAULT_OPTIMIZE
    )


FLOAT_FIELDS = ["risk_aversion", "gross", "max_weight", "beta_band", "sector_net", "sector_gross"]


@pytest.mark.parametrize("field", FLOAT_FIELDS)
@pytest.mark.parametrize("value", [NAN, INF, -INF, True, "x"])
def test_rejects_non_finite_or_non_numeric(field: str, value: object) -> None:
    params = dataclasses.replace(DEFAULT_OPTIMIZE, **{field: value})
    with pytest.raises(ValueError, match=f"{field} must be a finite number"):
        optimize_weights([], pd.DataFrame(), params=params)


@pytest.mark.parametrize("value", [0, -1, 1.5, NAN, True])
def test_rejects_invalid_min_names_per_side(value: object) -> None:
    params = dataclasses.replace(DEFAULT_OPTIMIZE, min_names_per_side=value)
    with pytest.raises(ValueError, match="min_names_per_side must be an int >= 1"):
        optimize_weights([], pd.DataFrame(), params=params)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"risk_aversion": 0.0}, "risk_aversion must be > 0"),
        ({"risk_aversion": -1.0}, "risk_aversion must be > 0"),
        ({"max_weight": 0.0}, r"0 < max_weight <= gross / 2"),
        ({"max_weight": 1.5}, r"0 < max_weight <= gross / 2"),  # gross / 2 = 1.0
        ({"gross": -2.0}, r"0 < max_weight <= gross / 2"),
        ({"max_weight": 0.4}, r"max_weight \* min_names_per_side"),  # 0.4 * 3 = 1.2 > 1.0
        ({"min_names_per_side": 11}, r"max_weight \* min_names_per_side"),  # 1.1 > 1.0
        ({"beta_band": -0.01}, "beta_band must be >= 0"),
        ({"sector_net": 0.0}, r"0 < sector_net <= sector_gross"),
        ({"sector_net": 0.6}, r"0 < sector_net <= sector_gross"),  # sector_gross = 0.5
    ],
)
def test_rejects_out_of_range(overrides: dict[str, float], message: str) -> None:
    params = dataclasses.replace(DEFAULT_OPTIMIZE, **overrides)
    with pytest.raises(ValueError, match=message):
        optimize_weights([], pd.DataFrame(), params=params)


def test_accepts_boundary_params() -> None:
    # max_weight * min_names_per_side == gross / 2 and sector_net == sector_gross are allowed;
    # beta_band 0 too. 0.5 * 2 == 1.0 exactly in floating point.
    params = OptimizeParams(
        max_weight=0.5, min_names_per_side=2, beta_band=0, sector_net=0.5, sector_gross=0.5
    )
    assert optimize_weights([], pd.DataFrame(), params=params) == Allocation({}, {})


def test_portfolio_infeasible_error_is_a_value_error() -> None:
    assert issubclass(PortfolioInfeasibleError, ValueError)


# --- thin book -------------------------------------------------------------


def test_thin_long_side_excludes_every_name() -> None:
    assets = _paired(3)[1:]  # L2, L3 and three shorts: 2 < min_names_per_side 3
    out = optimize_weights(assets, _sigma(assets))
    assert out.weights == {}
    assert out.excluded == dict.fromkeys(["L2", "L3", "S1", "S2", "S3"], "thin-book")


def test_missing_short_side_excludes_every_name() -> None:
    assets = _paired(3)[:3]
    out = optimize_weights(assets, _sigma(assets))
    assert out == Allocation({}, dict.fromkeys(["L1", "L2", "L3"], "thin-book"))


def test_thin_book_respects_min_names_per_side() -> None:
    assets = _paired(2)
    params = OptimizeParams(min_names_per_side=4, max_weight=0.25)  # 0.25 * 4 = 1.0
    out = optimize_weights(assets, _sigma(assets), params=params)
    assert out.excluded == dict.fromkeys(["L1", "L2", "S1", "S2"], "thin-book")
