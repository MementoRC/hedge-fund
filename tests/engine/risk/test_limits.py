"""Tests for the L5 limits: L4 exposure re-checks, net, and the factor-model risk limits."""

import dataclasses
import math
from collections.abc import Callable
from datetime import date

import pytest

from hedge_fund.domain.portfolio import Position, TargetPortfolio
from hedge_fund.domain.risk import RiskBreakdown, Violation
from hedge_fund.domain.shortlist import Side
from hedge_fund.engine.portfolio.optimize import DEFAULT_OPTIMIZE
from hedge_fund.engine.risk.limits import (
    DEFAULT_LIMITS,
    RiskLimits,
    check_exposures,
    check_risk,
    validate_limits,
)

NAN = math.nan
INF = math.inf
AS_OF = date(2026, 9, 30)
FIELDS = [field.name for field in dataclasses.fields(RiskLimits)]
LOOSE = RiskLimits(**dict.fromkeys(FIELDS, 100.0))  # tighten one field per test
INSIDE = 0.9e-6  # within the 1e-6 L4 tolerance
OUTSIDE = 1.1e-6  # just past it
JUST_OVER = math.nextafter(0.1, 1.0)  # smallest float above 0.1


def _pos(ticker: str, weight: float, *, sector: str = "Tech", beta: float = 0.0) -> Position:
    side = Side.LONG if weight > 0 else Side.SHORT
    return Position(ticker=ticker, sector=sector, side=side, weight=weight, alpha=0.0, beta=beta)


def _book(*positions: Position, recorded: float = 0.0) -> TargetPortfolio:
    """The checks read positions only; ``recorded`` fills the book-level statistics."""
    return TargetPortfolio(
        as_of=AS_OF,
        positions=positions,
        excluded={},
        gross=recorded,
        net=recorded,
        beta=recorded,
        expected_vol=recorded,
    )


def _breakdown(
    *,
    total_vol: float = 0.05,
    factor_share: float = 0.2,
    by_name: dict[str, float] | None = None,
    unassessed: tuple[str, ...] = (),
) -> RiskBreakdown:
    return RiskBreakdown(
        total_vol=total_vol,
        factor_vol=0.0,
        specific_vol=0.0,
        factor_share=factor_share,
        by_factor={},
        by_name={"A": 1.0} if by_name is None else by_name,
        unassessed=unassessed,
    )


# --- params ----------------------------------------------------------------


def test_l4_and_net_defaults() -> None:
    # vol, name_risk and factor_share defaults are provisional and deliberately not pinned
    # until they are fixed from fixture-book measurements.
    opt = DEFAULT_OPTIMIZE
    l4 = (opt.gross, opt.max_weight, opt.beta_band, opt.sector_net, opt.sector_gross)
    limits = DEFAULT_LIMITS
    assert (
        limits.gross,
        limits.max_weight,
        limits.beta_band,
        limits.sector_net,
        limits.sector_gross,
    ) == l4
    assert limits.net == 0.10


@pytest.mark.parametrize("field", FIELDS)
@pytest.mark.parametrize("value", [NAN, INF, True, "x"])
def test_rejects_non_finite_or_non_numeric_limits(field: str, value: object) -> None:
    with pytest.raises(ValueError, match=f"{field} must be a finite number"):
        validate_limits(dataclasses.replace(DEFAULT_LIMITS, **{field: value}))


@pytest.mark.parametrize("field", FIELDS)
@pytest.mark.parametrize("value", [0, 0.0, -0.1])
def test_rejects_non_positive_limits(field: str, value: float) -> None:
    with pytest.raises(ValueError, match=f"{field} must be > 0"):
        validate_limits(dataclasses.replace(DEFAULT_LIMITS, **{field: value}))


def test_type_checks_come_before_range_checks() -> None:
    limits = dataclasses.replace(DEFAULT_LIMITS, gross=-1.0, factor_share=NAN)
    with pytest.raises(ValueError, match="factor_share must be a finite number"):
        validate_limits(limits)


def test_both_checks_validate_their_limits() -> None:
    bad = dataclasses.replace(DEFAULT_LIMITS, net=0)
    with pytest.raises(ValueError, match="net must be > 0"):
        check_exposures(_book(), limits=bad)
    with pytest.raises(ValueError, match="net must be > 0"):
        check_risk(_breakdown(), limits=bad)


# --- exposures -------------------------------------------------------------

Build = Callable[[float], tuple[Position, ...]]
L4_CASES: list[tuple[str, str, Build]] = [
    ("gross", "gross", lambda v: (_pos("A", v / 2), _pos("B", -v / 2))),
    ("max_weight", "name:A", lambda v: (_pos("A", v),)),
    ("max_weight", "name:A", lambda v: (_pos("A", -v),)),
    ("beta_band", "beta", lambda v: (_pos("A", 0.5, beta=2 * v),)),
    ("beta_band", "beta", lambda v: (_pos("A", -0.5, beta=2 * v),)),
    ("sector_net", "sector_net:Tech", lambda v: (_pos("A", -v),)),
    ("sector_gross", "sector_gross:Tech", lambda v: (_pos("A", v / 2), _pos("B", -v / 2))),
]


@pytest.mark.parametrize(
    ("field", "check", "build"),
    L4_CASES,
    ids=["gross", "name-long", "name-short", "beta-up", "beta-down", "sector-net", "sector-gross"],
)
def test_l4_rechecks_allow_1e6_of_rounding(field: str, check: str, build: Build) -> None:
    limits = dataclasses.replace(LOOSE, **{field: 0.1})
    assert check_exposures(_book(*build(0.1 + INSIDE)), limits=limits) == []
    (violation,) = check_exposures(_book(*build(0.1 + OUTSIDE)), limits=limits)
    assert violation.check == check
    assert violation.observed == pytest.approx(0.1 + OUTSIDE, abs=1e-12)
    assert violation.limit == 0.1


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_net_is_inclusive_with_no_tolerance(sign: float) -> None:
    limits = dataclasses.replace(LOOSE, net=0.1)
    assert check_exposures(_book(_pos("A", sign * 0.1)), limits=limits) == []
    assert check_exposures(_book(_pos("A", sign * JUST_OVER)), limits=limits) == [
        Violation(check="net", observed=JUST_OVER, limit=0.1, detail="|sum of weight|")
    ]
    (inside,) = check_exposures(_book(_pos("A", sign * (0.1 + INSIDE))), limits=limits)
    assert inside.check == "net"  # the L4 tolerance does not apply to net


def test_a_market_neutral_book_passes_the_defaults() -> None:
    book = _book(
        _pos("A", 0.1, beta=1.2),
        _pos("B", -0.1, beta=1.1),
        _pos("C", 0.1, sector="Energy", beta=0.8),
        _pos("D", -0.1, sector="Energy", beta=0.9),
    )
    assert check_exposures(book) == []


def test_every_exposure_violation_is_reported() -> None:
    # One 0.3 Tech long: name, sector net and net all breach; gross 0.3 and sector gross
    # 0.3 do not.
    out = check_exposures(_book(_pos("A", 0.3)))
    assert sorted(v.check for v in out) == ["name:A", "net", "sector_net:Tech"]


def test_nan_weight_breaches_every_exposure_check() -> None:
    # NaN fails every ordered comparison, so "not observed <= limit" counts it as a breach.
    out = check_exposures(_book(_pos("A", NAN)))
    assert sorted(v.check for v in out) == [
        "beta",
        "gross",
        "name:A",
        "net",
        "sector_gross:Tech",
        "sector_net:Tech",
    ]
    assert all(math.isnan(v.observed) for v in out)


def test_nan_beta_breaches_the_beta_check() -> None:
    (violation,) = check_exposures(_book(_pos("A", 0.1, beta=NAN)))
    assert violation.check == "beta"
    assert math.isnan(violation.observed)


def test_book_level_statistics_are_not_read() -> None:
    assert check_exposures(_book(recorded=5.0)) == []


# --- risk ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("field", "check", "argument"),
    [("vol", "vol", "total_vol"), ("factor_share", "factor_share", "factor_share")],
)
def test_risk_limits_are_inclusive_with_no_tolerance(field: str, check: str, argument: str) -> None:
    limits = dataclasses.replace(LOOSE, **{field: 0.1})
    assert check_risk(_breakdown(**{argument: 0.1}), limits=limits) == []
    (violation,) = check_risk(_breakdown(**{argument: JUST_OVER}), limits=limits)
    assert (violation.check, violation.observed, violation.limit) == (check, JUST_OVER, 0.1)


def test_name_risk_is_inclusive_and_ignores_hedges() -> None:
    limits = dataclasses.replace(LOOSE, name_risk=0.1)
    by_name = {"A": 0.1, "B": JUST_OVER, "C": -5.0}  # C hedges: a negative share
    assert check_risk(_breakdown(by_name=by_name), limits=limits) == [
        Violation(
            check="name_risk:B",
            observed=JUST_OVER,
            limit=0.1,
            detail="share of total variance from B",
        )
    ]


def test_nan_risk_values_breach() -> None:
    breakdown = _breakdown(total_vol=NAN, factor_share=NAN, by_name={"A": NAN})
    out = check_risk(breakdown)
    assert sorted(v.check for v in out) == ["factor_share", "name_risk:A", "vol"]
    assert all(math.isnan(v.observed) for v in out)


def test_unassessable_names_skip_the_risk_checks() -> None:
    # Every risk value is far over its limit, yet only the unassessable violations appear.
    # NaN != NaN, and Python 3.13 dataclass equality compares fields with ==, so the NaN
    # fields are checked with math.isnan instead of comparing Violation objects.
    breakdown = _breakdown(
        total_vol=9.0, factor_share=9.0, by_name={"A": 9.0}, unassessed=("Q", "R")
    )
    out = check_risk(breakdown)
    assert [(v.check, v.detail) for v in out] == [
        ("unassessable:Q", "no factor-model estimate for Q"),
        ("unassessable:R", "no factor-model estimate for R"),
    ]
    assert all(math.isnan(v.observed) and math.isnan(v.limit) for v in out)
