"""Layer 5 risk check: approve or reject a proposed target book as a whole."""

from __future__ import annotations

import math

import pandas as pd

from hedge_fund.domain.portfolio import TargetPortfolio
from hedge_fund.domain.risk import NavHistory, RiskVerdict
from hedge_fund.domain.snapshot import PriceSnapshot
from hedge_fund.engine.portfolio.risk_model import DEFAULT_RISK, RiskParams, validate_risk_params
from hedge_fund.engine.risk.breakers import (
    DEFAULT_BREAKERS,
    BreakerParams,
    check_breakers,
    validate_breakers,
)
from hedge_fund.engine.risk.factor_model import decompose, empty_breakdown, fit_factor_model
from hedge_fund.engine.risk.limits import (
    DEFAULT_LIMITS,
    RiskLimits,
    check_exposures,
    check_risk,
    validate_limits,
)
from hedge_fund.engine.risk.veto import assess
from hedge_fund.services.portfolio import naive_dates, validate_benchmark_end


def assess_portfolio(
    target: TargetPortfolio,
    snapshot: PriceSnapshot,
    benchmark: pd.Series,
    *,
    nav: NavHistory = (),
    limits: RiskLimits = DEFAULT_LIMITS,
    breakers: BreakerParams = DEFAULT_BREAKERS,
    risk: RiskParams = DEFAULT_RISK,
) -> RiskVerdict:
    """Breakers, factor model, limit checks -> one verdict; ``target`` is only read.

    The factor model is fitted on every name in ``snapshot`` (skipped for an empty book,
    which gets a zero breakdown). Raises ValueError when ``target`` and ``snapshot``
    disagree on ``as_of``, when ``benchmark`` (tz-normalised) does not end on the snapshot's
    last date, on duplicate book tickers, on a non-finite position weight or beta, on a
    position sector that differs from the
    snapshot's, on a NAV history ending after ``as_of`` or malformed, or on any invalid
    param set (all checked up front, even for an empty book); factor-model errors (zero
    benchmark variance, too few usable days) propagate.
    """
    benchmark = naive_dates(benchmark)
    validate_book_inputs(target, snapshot, benchmark, nav)
    validate_limits(limits)
    validate_breakers(breakers)
    validate_risk_params(risk)
    state = check_breakers(nav, params=breakers)
    weights = {p.ticker: p.weight for p in target.positions}
    if weights:
        model = fit_factor_model(snapshot.closes, benchmark, snapshot.sectors, params=risk)
        breakdown = decompose(model, weights)
    else:
        breakdown = empty_breakdown()
    violations = check_exposures(target, limits=limits) + check_risk(breakdown, limits=limits)
    return assess(target.as_of, violations, breakdown, state)


def validate_book_inputs(
    target: TargetPortfolio, snapshot: PriceSnapshot, benchmark: pd.Series, nav: NavHistory
) -> None:
    """Cross-input checks for ``assess_portfolio``; ``benchmark`` may be tz-aware."""
    if target.as_of != snapshot.as_of:
        raise ValueError(
            f"target as_of {target.as_of} differs from snapshot as_of {snapshot.as_of}"
        )
    validate_benchmark_end(snapshot, benchmark)
    tickers = [p.ticker for p in target.positions]
    duplicates = sorted({t for t in tickers if tickers.count(t) > 1})
    if duplicates:
        raise ValueError(f"duplicate book tickers: {', '.join(map(repr, duplicates))}")
    for p in target.positions:
        for field in ("weight", "beta"):
            value = getattr(p, field)
            if not math.isfinite(value):
                raise ValueError(f"{p.ticker!r} {field} must be finite, got {value!r}")
        expected = snapshot.sectors.get(p.ticker)
        if expected is not None and expected != p.sector:
            raise ValueError(
                f"{p.ticker!r} sector {p.sector!r} differs from the snapshot's {expected!r}"
            )
    if nav and nav[-1].date > target.as_of:
        raise ValueError(f"NAV history ends on {nav[-1].date}, after as_of {target.as_of}")
