"""Risk limits: L4 exposure re-checks on the recorded book plus L5's own net and risk limits."""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence
from dataclasses import dataclass

from hedge_fund.domain.portfolio import Position, TargetPortfolio
from hedge_fund.domain.risk import RiskBreakdown, Violation

L4_TOLERANCE = 1e-6

# check id, observed, limit, detail, tolerance
_Measure = tuple[str, float, float, str, float]


@dataclass(frozen=True)
class RiskLimits:
    """L4 re-checks (``OptimizeParams`` defaults) and L5's own limits, as fractions.

    Risk defaults, fixed 2026-10 from fixture-book measurements: ``vol`` 0.10 (ex-ante annual
    volatility), ``name_risk`` 0.15 (signed share of total variance; hedges (negative shares) never breach) and
    ``factor_share`` 0.50 (factor share of total variance).
    """

    gross: float = 2.0
    max_weight: float = 0.10
    beta_band: float = 0.10
    sector_net: float = 0.10
    sector_gross: float = 0.50
    net: float = 0.10
    vol: float = 0.10
    name_risk: float = 0.15
    factor_share: float = 0.50


DEFAULT_LIMITS = RiskLimits()


def check_exposures(
    target: TargetPortfolio, *, limits: RiskLimits = DEFAULT_LIMITS
) -> list[Violation]:
    """Re-check the L4 limits (``1e-6`` tolerance) and ``net`` (inclusive, no tolerance).

    Everything is measured from ``target.positions`` (weight, sector, recorded beta); the
    book-level ``gross``/``net``/``beta`` fields are not read. Raises ValueError on invalid
    limits.
    """
    validate_limits(limits)
    positions = target.positions
    measures: list[_Measure] = [
        (
            "gross",
            sum(abs(p.weight) for p in positions),
            limits.gross,
            "sum of |weight|",
            L4_TOLERANCE,
        ),
        (
            "beta",
            abs(sum(p.weight * p.beta for p in positions)),
            limits.beta_band,
            "|sum of weight * beta|",
            L4_TOLERANCE,
        ),
        ("net", abs(sum(p.weight for p in positions)), limits.net, "|sum of weight|", 0.0),
    ]
    measures += [
        (
            f"name:{p.ticker}",
            abs(p.weight),
            limits.max_weight,
            f"|weight| of {p.ticker}",
            L4_TOLERANCE,
        )
        for p in positions
    ]
    measures += _sector_measures(positions, limits)
    return _breaches(measures)


def _sector_measures(positions: Sequence[Position], limits: RiskLimits) -> list[_Measure]:
    out: list[_Measure] = []
    for sector in sorted({p.sector for p in positions}):
        weights = [p.weight for p in positions if p.sector == sector]
        out.append(
            (
                f"sector_net:{sector}",
                abs(sum(weights)),
                limits.sector_net,
                f"|net weight| of sector {sector}",
                L4_TOLERANCE,
            )
        )
        out.append(
            (
                f"sector_gross:{sector}",
                sum(abs(w) for w in weights),
                limits.sector_gross,
                f"gross weight of sector {sector}",
                L4_TOLERANCE,
            )
        )
    return out


def check_risk(breakdown: RiskBreakdown, *, limits: RiskLimits = DEFAULT_LIMITS) -> list[Violation]:
    """``vol``, ``name_risk:<T>`` and ``factor_share``: inclusive, no tolerance.

    An unassessed name gives ``unassessable:<T>`` (observed = limit = nan) and skips the
    three risk checks: the book is rejected anyway. A negative (hedging) share never
    breaches ``name_risk``. Raises ValueError on invalid limits.
    """
    validate_limits(limits)
    if breakdown.unassessed:
        return [
            Violation(
                check=f"unassessable:{t}",
                observed=math.nan,
                limit=math.nan,
                detail=f"no factor-model estimate for {t}",
            )
            for t in breakdown.unassessed
        ]
    measures: list[_Measure] = [
        ("vol", breakdown.total_vol, limits.vol, "ex-ante annual volatility", 0.0),
        (
            "factor_share",
            breakdown.factor_share,
            limits.factor_share,
            "factor share of total variance",
            0.0,
        ),
    ]
    measures += [
        (f"name_risk:{t}", share, limits.name_risk, f"share of total variance from {t}", 0.0)
        for t, share in breakdown.by_name.items()
    ]
    return _breaches(measures)


def _breaches(measures: list[_Measure]) -> list[Violation]:
    return [
        Violation(check=check, observed=observed, limit=limit, detail=detail)
        for check, observed, limit, detail, tolerance in measures
        if not observed <= limit + tolerance  # NaN compares False, so it breaches
    ]


def validate_limits(limits: RiskLimits) -> None:
    """Type check every field first (bool and NaN slip past ``> 0``), then require > 0."""
    values = {field.name: getattr(limits, field.name) for field in dataclasses.fields(limits)}
    for name, value in values.items():
        if (
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(value)
        ):
            raise ValueError(f"{name} must be a finite number, got {value!r}")
    for name, value in values.items():
        if not value > 0:
            raise ValueError(f"{name} must be > 0, got {value!r}")
