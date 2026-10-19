"""Tests for the market-neutral mean-variance optimizer (synthetic sigma and alpha)."""

import dataclasses
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from hedge_fund.domain.shortlist import Side
from hedge_fund.engine.portfolio import optimize as optimize_module
from hedge_fund.engine.portfolio.optimize import (
    DEFAULT_OPTIMIZE,
    Allocation,
    Asset,
    OptimizeParams,
    PortfolioInfeasibleError,
    _feasible_start,
    _problem,
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


# --- feasible side gross (LP) ----------------------------------------------


def _start(
    assets: list[Asset], params: OptimizeParams = DEFAULT_OPTIMIZE
) -> tuple[float, np.ndarray]:
    start = _feasible_start(_problem(assets, _sigma(assets), params), params)
    assert start is not None
    return start


def _lopsided() -> list[Asset]:
    """Three longs all in Tech and three shorts all in Energy, every beta 1."""
    longs = [_asset(f"L{k}", "Tech", LONG, 0.01) for k in (1, 2, 3)]
    shorts = [_asset(f"S{k}", "Energy", SHORT, -0.01) for k in (1, 2, 3)]
    return longs + shorts


def _beta_tilted() -> list[Asset]:
    """Paired sectors; long betas 2 and short betas 1, so book beta = 2g - g = g."""
    return [
        dataclasses.replace(asset, beta=2.0 if asset.side is LONG else 1.0) for asset in _paired(3)
    ]


def test_side_gross_reaches_the_cap_with_every_name_at_its_bound() -> None:
    # 3 + 3 names, default params: cap = min(2.0 / 2, 3 * 0.1, 3 * 0.1) = 0.3. Paired
    # sectors have net 0 and beta = sum(w) = 0, so g* = cap and no slack is taken.
    side_gross, start = _start(_paired(3))
    assert side_gross == pytest.approx(0.3, abs=1e-12)
    np.testing.assert_allclose(start, [0.1, 0.1, 0.1, -0.1, -0.1, -0.1], atol=1e-12)


def test_side_gross_is_capped_by_the_thinner_side() -> None:
    # 4 longs, 3 shorts (S4 dropped): cap = min(1.0, 4 * 0.1, 3 * 0.1) = 0.3.
    side_gross, start = _start(_paired(4)[:7])
    assert side_gross == pytest.approx(0.3, abs=1e-12)
    np.testing.assert_allclose(start[4:], [-0.1, -0.1, -0.1], atol=1e-12)


def test_side_gross_is_capped_by_half_the_gross() -> None:
    # gross 0.6, 4 + 4 names: cap = min(0.6 / 2, 0.4, 0.4) = 0.3.
    params = OptimizeParams(gross=0.6, min_names_per_side=2)  # 0.1 * 2 <= 0.3
    side_gross, start = _start(_paired(4), params)
    assert side_gross == pytest.approx(0.3, abs=1e-12)
    assert start[:4].sum() == pytest.approx(0.3, abs=1e-12)
    assert start[4:].sum() == pytest.approx(-0.3, abs=1e-12)


def test_lopsided_sectors_shrink_side_gross_to_the_lp_optimum_with_slack() -> None:
    # Tech net = g <= sector_net 0.10 binds first: sector_gross 0.50 would bind at g = 0.5,
    # the cap is 0.3 and beta = g - g = 0. g* = 0.10 < cap, so side_gross = 0.10 * (1 - 1e-6)
    # and the LP start is scaled down to it.
    side_gross, start = _start(_lopsided())
    assert side_gross == pytest.approx(0.10 * (1 - 1e-6), rel=1e-9)
    assert start[:3].sum() == pytest.approx(side_gross, rel=1e-9)
    assert start[3:].sum() == pytest.approx(-side_gross, rel=1e-9)


def test_beta_band_unreachable_for_any_positive_gross_is_infeasible() -> None:
    # Book beta = g, so beta_band 0 forces the LP optimum g* = 0.
    assets = _beta_tilted()
    out = optimize_weights(assets, _sigma(assets), params=OptimizeParams(beta_band=0.0))
    tickers = ["L1", "L2", "L3", "S1", "S2", "S3"]
    assert out == Allocation({}, dict.fromkeys(tickers, "infeasible"))


def test_side_gross_below_one_millionth_is_infeasible() -> None:
    # Book beta = g, so beta_band 5e-7 gives g* = 5e-7 and side_gross < 1e-6.
    assets = _beta_tilted()
    out = optimize_weights(assets, _sigma(assets), params=OptimizeParams(beta_band=5e-7))
    assert out.weights == {}
    assert set(out.excluded.values()) == {"infeasible"}


def test_lp_solver_failure_is_infeasible(monkeypatch: pytest.MonkeyPatch) -> None:
    # w = 0, g = 0 satisfies every limit (beta_band >= 0, sector_net > 0), so the LP is never
    # infeasible: this branch is reachable only through a solver failure status.
    def failed_linprog(*args: object, **kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(success=False, status=4, message="Numerical difficulties", x=None)

    monkeypatch.setattr(optimize_module, "linprog", failed_linprog)
    assets = _paired(3)
    out = optimize_weights(assets, _sigma(assets))
    assert out.excluded == dict.fromkeys(["L1", "L2", "L3", "S1", "S2", "S3"], "infeasible")


# --- QP solve --------------------------------------------------------------

TOL = 1e-6


def _assert_within_limits(
    assets: list[Asset], out: Allocation, params: OptimizeParams = DEFAULT_OPTIMIZE
) -> None:
    w = {asset.ticker: out.weights.get(asset.ticker, 0.0) for asset in assets}
    for asset in assets:
        low, high = (0.0, params.max_weight) if asset.side is LONG else (-params.max_weight, 0.0)
        assert low - TOL <= w[asset.ticker] <= high + TOL
    longs = sum(w[a.ticker] for a in assets if a.side is LONG)
    shorts = sum(w[a.ticker] for a in assets if a.side is SHORT)
    assert longs == pytest.approx(-shorts, abs=TOL)
    assert longs <= params.gross / 2 + TOL
    assert abs(sum(a.beta * w[a.ticker] for a in assets)) <= params.beta_band + TOL
    for sector in {asset.sector for asset in assets}:
        members = [a.ticker for a in assets if a.sector == sector]
        assert abs(sum(w[t] for t in members)) <= params.sector_net + TOL
        assert sum(abs(w[t]) for t in members) <= params.sector_gross + TOL


def test_weights_tilt_toward_alpha_and_zero_weights_are_dropped() -> None:
    # Diagonal sigma 0.04, risk_aversion 1: per side maximize sum(a_i w_i) - 0.04 sum(w_i^2)
    # with sum(w) = 0.3, so w_i = (a_i - lam) / 0.08, floored at 0. With L4 at 0:
    # (0.030 - 3 lam) / 0.08 = 0.3 -> lam = 0.002 -> w = 0.125, 0.1, 0.075; L4's alpha
    # 0.001 < lam keeps it at 0. Shorts mirror. Paired sectors: every sector net is 0.
    params = OptimizeParams(gross=0.6, max_weight=0.15, min_names_per_side=2)
    assets = _paired(4, [0.012, 0.010, 0.008, 0.001])
    out = optimize_weights(assets, _sigma(assets), params=params)
    expected = {"L1": 0.125, "L2": 0.1, "L3": 0.075, "S1": -0.125, "S2": -0.1, "S3": -0.075}
    assert out.weights == pytest.approx(expected, abs=TOL)
    assert list(out.weights) == ["L1", "L2", "L3", "S1", "S2", "S3"]
    assert out.excluded == {"L4": "optimizer:zero", "S4": "optimizer:zero"}
    _assert_within_limits(assets, out, params)


def test_every_name_at_its_bound_still_solves() -> None:
    # g* = cap = 3 * max_weight = 0.3: every bound is active together with the side sums.
    assets = _paired(3)
    out = optimize_weights(assets, _sigma(assets))
    expected = {"L1": 0.1, "L2": 0.1, "L3": 0.1, "S1": -0.1, "S2": -0.1, "S3": -0.1}
    assert out.weights == pytest.approx(expected, abs=TOL)
    assert out.excluded == {}


def test_thinner_side_sets_the_side_gross() -> None:
    # 4 longs, 3 shorts: side gross 0.3, so every short sits at -0.1. Longs:
    # (0.036 - 4 lam) / 0.08 = 0.3 -> lam = 0.003 -> w = 0.0975, 0.0875, 0.075, 0.04.
    assets = _paired(4, [0.0108, 0.010, 0.009, 0.0062])[:7]
    out = optimize_weights(assets, _sigma(assets))
    expected = {
        "L1": 0.0975,
        "L2": 0.0875,
        "L3": 0.075,
        "L4": 0.04,
        "S1": -0.1,
        "S2": -0.1,
        "S3": -0.1,
    }
    assert out.weights == pytest.approx(expected, abs=TOL)
    _assert_within_limits(assets, out)


def test_lopsided_sectors_size_the_book_to_the_lp_optimum() -> None:
    # Same setup as the LP test: sector_net binds at g = 0.10, side_gross = 0.10 * (1 - 1e-6).
    assets = _lopsided()
    out = optimize_weights(assets, _sigma(assets))
    longs = sum(weight for weight in out.weights.values() if weight > 0)
    shorts = sum(weight for weight in out.weights.values() if weight < 0)
    assert longs == pytest.approx(0.10 * (1 - 1e-6), rel=1e-9)
    assert shorts == pytest.approx(-0.10 * (1 - 1e-6), rel=1e-9)
    assert out.excluded == {}
    _assert_within_limits(assets, out)


def test_mixed_book_respects_every_limit() -> None:
    # Correlated names (0.04 variance, 0.01 covariance), uneven betas, shared sectors.
    params = OptimizeParams(gross=0.6, min_names_per_side=2)
    assets = [
        _asset("L1", "Tech", LONG, 0.012, 1.5),
        _asset("L2", "Tech", LONG, 0.010, 1.2),
        _asset("L3", "Health", LONG, 0.008, 0.8),
        _asset("L4", "Energy", LONG, 0.006, 0.5),
        _asset("S1", "Tech", SHORT, -0.012, 0.6),
        _asset("S2", "Health", SHORT, -0.010, 0.9),
        _asset("S3", "Health", SHORT, -0.008, 1.1),
        _asset("S4", "Energy", SHORT, -0.006, 1.4),
    ]
    tickers = [asset.ticker for asset in assets]
    sigma = pd.DataFrame(0.01 + 0.03 * np.eye(8), index=tickers, columns=tickers)
    out = optimize_weights(assets, sigma, params=params)
    assert out.weights
    _assert_within_limits(assets, out, params)


def test_repeat_runs_give_identical_weights() -> None:
    assets = _paired(4, [0.0108, 0.010, 0.009, 0.0062])[:7]
    assert optimize_weights(assets, _sigma(assets)) == optimize_weights(assets, _sigma(assets))


def _fake_minimize(monkeypatch: pytest.MonkeyPatch, *, success: bool, x: list[float]) -> None:
    def fake(*args: object, **kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(
            success=success, status=0 if success else 9, message="scripted", x=np.array(x)
        )

    monkeypatch.setattr(optimize_module, "minimize", fake)


def test_solver_failure_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_minimize(monkeypatch, success=False, x=[0.1, 0.1, 0.1, -0.1, -0.1, -0.1])
    assets = _paired(3)
    with pytest.raises(PortfolioInfeasibleError, match=r"status 9 \(scripted\): did not converge"):
        optimize_weights(assets, _sigma(assets))


def test_solution_breaking_a_side_sum_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    # All-zero weights pass every bound but miss the long side sum of 0.3 by 0.3.
    _fake_minimize(monkeypatch, success=True, x=[0.0] * 6)
    assets = _paired(3)
    with pytest.raises(PortfolioInfeasibleError, match="long side sum violated by 0.3"):
        optimize_weights(assets, _sigma(assets))


def test_solution_breaking_a_bound_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    # Side sums hold (0.2 + 0.05 + 0.05 = 0.3) but L1 exceeds max_weight 0.1 by 0.1.
    _fake_minimize(monkeypatch, success=True, x=[0.2, 0.05, 0.05, -0.1, -0.1, -0.1])
    assets = _paired(3)
    with pytest.raises(PortfolioInfeasibleError, match="L1 weight bounds violated by 0.1"):
        optimize_weights(assets, _sigma(assets))


def test_solution_breaking_a_sector_limit_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    # 4 + 4 paired names, side gross 0.3. Bounds and side sums hold (0.1 * 3 = 0.3), beta is
    # 0.3 - 0.3 = 0, but Sec1 net = 0.1 + 0.0 = 0.1 exceeds sector_net 0.05 by 0.05.
    _fake_minimize(monkeypatch, success=True, x=[0.1, 0.1, 0.1, 0.0, 0.0, -0.1, -0.1, -0.1])
    params = OptimizeParams(gross=0.6, min_names_per_side=2, sector_net=0.05)
    assets = _paired(4)
    with pytest.raises(PortfolioInfeasibleError, match="sector Sec1 net <= sector_net violated"):
        optimize_weights(assets, _sigma(assets), params=params)


def test_solution_breaking_the_short_side_sum_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    # Bounds and the long sum (0.3) hold; the shorts sum to -0.2, missing -0.3 by 0.1.
    _fake_minimize(monkeypatch, success=True, x=[0.1, 0.1, 0.1, -0.1, -0.1, 0.0])
    assets = _paired(3)
    with pytest.raises(PortfolioInfeasibleError, match="short side sum violated by 0.1"):
        optimize_weights(assets, _sigma(assets))


def test_solution_breaking_the_beta_band_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    # Long betas 3, 1, 1 and short betas 1: book beta = 3 w1 + w2 + w3 - g = 2 w1, so
    # beta_band 0.05 caps w1 at 0.025 and the LP gives g* = 0.225 (side_gross 0.225 minus
    # 1e-6 slack). The fake weights keep bounds, side sums (within 2.25e-7) and sectors
    # (net at most 0.1 - 0.075 = 0.025), but beta = 2 * 0.1 = 0.2 breaks 0.05 by 0.15.
    _fake_minimize(monkeypatch, success=True, x=[0.1, 0.1, 0.025, -0.075, -0.075, -0.075])
    betas = [3.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    assets = [dataclasses.replace(a, beta=b) for a, b in zip(_paired(3), betas, strict=True)]
    params = OptimizeParams(beta_band=0.05)
    with pytest.raises(PortfolioInfeasibleError, match="beta <= beta_band violated by 0.15"):
        optimize_weights(assets, _sigma(assets), params=params)


def test_zero_drops_do_not_rerun_the_thin_book_check() -> None:
    # Lopsided sectors: side_gross = 0.10 * (1 - 1e-6). L1's alpha 0.05 dwarfs the others:
    # if only L1 is held, lam = 0.05 - 0.08 * 0.1 = 0.042 > 0.001, so L2 and L3 stay at 0
    # (S side mirrors). One name per side is left, below min_names_per_side 3, yet the
    # book is returned: the thin-book rule runs once, before the LP.
    alphas = [0.05, 0.001, 0.001, -0.05, -0.001, -0.001]
    assets = [dataclasses.replace(a, alpha=x) for a, x in zip(_lopsided(), alphas, strict=True)]
    out = optimize_weights(assets, _sigma(assets))
    side_gross = 0.10 * (1 - 1e-6)
    assert out.weights == pytest.approx({"L1": side_gross, "S1": -side_gross}, abs=TOL)
    assert out.excluded == dict.fromkeys(["L2", "L3", "S2", "S3"], "optimizer:zero")
