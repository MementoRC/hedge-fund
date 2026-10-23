"""Tests for the risk service: input checks and the verdict on a real Layer 4 book."""

import dataclasses
import functools
import math
from datetime import timedelta

import pandas as pd
import pytest

from hedge_fund.domain.portfolio import TargetPortfolio
from hedge_fund.domain.research import NameResearch, ResearchResult, Verdict
from hedge_fund.domain.risk import BreakerState, NavPoint, RiskVerdict
from hedge_fund.domain.shortlist import Candidate, Side
from hedge_fund.engine.portfolio.risk_model import RiskParams
from hedge_fund.engine.risk.breakers import BreakerParams
from hedge_fund.engine.risk.factor_model import empty_breakdown
from hedge_fund.engine.risk.limits import RiskLimits
from hedge_fund.services.portfolio import build_portfolio
from hedge_fund.services.risk import assess_portfolio
from tests.factories import make_closes, snapshot_from

# Three sectors of four random-walk names; within each sector the alphabetically 1st and
# 3rd are researched long, the 2nd and 4th short, all CONFIRM with score 1. SPY is an
# independent random walk, so every beta is small but non-zero (the Walsh-pattern fixture
# in tests/services/test_portfolio.py has every beta exactly 0, which leaves rank(X_t) < K
# on every day and cannot be fitted).
SECTORS = {
    f"{prefix}{suffix}": sector
    for prefix, sector in (("A", "Tech"), ("B", "Health"), ("C", "Energy"))
    for suffix in ("AA", "BB", "CC", "DD")
}
SNAPSHOT = snapshot_from(make_closes(list(SECTORS), seed=3), SECTORS)
SPY = make_closes(["SPY"], seed=11)["SPY"]
AS_OF = SNAPSHOT.as_of  # 2026-09-30
L4_CHECKS = ("gross", "name:", "beta", "sector_net:", "sector_gross:", "net")
QUIET = BreakerState(daily_loss=0.0, drawdown=0.0, daily_tripped=False, drawdown_tripped=False)


def _research() -> ResearchResult:
    names = []
    for ticker, sector in SECTORS.items():
        side = Side.LONG if ticker[1:] in ("AA", "CC") else Side.SHORT
        candidate = Candidate(ticker=ticker, sector=sector, side=side, depth=1, composite=1.0)
        names.append(
            NameResearch(
                candidate=candidate,
                runs=(),
                valid=3,
                mean=2.0,
                std=0.0,
                agreement=1.0,
                verdict=Verdict.CONFIRM,
            )
        )
    return ResearchResult(as_of=AS_OF, names=tuple(names))


@functools.cache
def _book() -> TargetPortfolio:
    """The real Layer 4 book on default params: all 12 names at +-0.10."""
    return build_portfolio(_research(), SNAPSHOT, SPY)


def _assess(book: TargetPortfolio, **kwargs: object) -> RiskVerdict:
    return assess_portfolio(book, SNAPSHOT, SPY, **kwargs)  # type: ignore[arg-type]


def _empty() -> TargetPortfolio:
    return TargetPortfolio(
        as_of=AS_OF, positions=(), excluded={}, gross=0.0, net=0.0, beta=0.0, expected_vol=0.0
    )


def test_layer4_book_passes_the_l4_rechecks_and_net() -> None:
    book = _book()
    assert len(book.positions) == 12
    verdict = _assess(book)
    assert [v.check for v in verdict.violations if v.check.startswith(L4_CHECKS)] == []
    assert verdict.halt is False
    breakdown = verdict.breakdown
    assert breakdown.unassessed == ()
    assert set(breakdown.by_name) == set(SECTORS)
    values = [breakdown.total_vol, breakdown.factor_share, *breakdown.by_name.values()]
    assert all(math.isfinite(v) for v in values)
    assert sum(breakdown.by_name.values()) == pytest.approx(1.0, rel=1e-10)


def test_layer4_book_is_vetoed_on_name_risk_only() -> None:
    """Strict veto by design: the equal-weight fixture book concentrates variance in a few names."""
    verdict = _assess(_book())
    assert verdict.approved is False
    assert {v.check.split(":")[0] for v in verdict.violations} == {"name_risk"}


def test_hand_edited_breaching_book_is_rejected() -> None:
    book = _book()
    first = book.positions[0]  # the largest long, +0.10
    edited = dataclasses.replace(
        book, positions=(dataclasses.replace(first, weight=0.3), *book.positions[1:])
    )
    verdict = _assess(edited)
    assert (verdict.approved, verdict.halt) == (False, False)
    checks = {v.check for v in verdict.violations}
    assert {f"name:{first.ticker}", f"sector_net:{first.sector}", "net"} <= checks
    name = next(v for v in verdict.violations if v.check == f"name:{first.ticker}")
    assert (name.observed, name.limit) == (pytest.approx(0.3), 0.10)


def test_tz_aware_benchmark_is_accepted() -> None:
    aware = SPY.copy()
    aware.index = aware.index.tz_localize("America/New_York")
    assert assess_portfolio(_book(), SNAPSHOT, aware) == _assess(_book())


def test_empty_book_is_approved_with_a_zero_breakdown() -> None:
    assert _assess(_empty()) == RiskVerdict(
        as_of=AS_OF,
        approved=True,
        halt=False,
        violations=(),
        breakdown=empty_breakdown(),
        breakers=QUIET,
    )


def test_tripped_breaker_halts_even_an_empty_book() -> None:
    nav = (NavPoint(AS_OF - timedelta(days=1), 100.0), NavPoint(AS_OF, 97.0))
    verdict = _assess(_empty(), nav=nav)
    assert (verdict.approved, verdict.halt) == (False, True)
    assert verdict.breakers.daily_tripped is True


def test_nav_may_end_before_as_of() -> None:
    nav = (NavPoint(AS_OF - timedelta(days=3), 100.0), NavPoint(AS_OF - timedelta(days=1), 99.0))
    verdict = _assess(_empty(), nav=nav)
    assert verdict.breakers.daily_loss == pytest.approx(-0.01)
    assert verdict.approved is True


def test_unassessable_name_gives_a_partial_breakdown_and_skips_risk_checks() -> None:
    book = _book()
    stranger = dataclasses.replace(book.positions[0], ticker="QQQ", weight=0.01)
    verdict = _assess(dataclasses.replace(book, positions=(*book.positions, stranger)))
    assert verdict.approved is False
    assert verdict.breakdown.unassessed == ("QQQ",)
    assert set(verdict.breakdown.by_name) == set(SECTORS)
    checks = [v.check for v in verdict.violations]
    assert "unassessable:QQQ" in checks
    assert not [c for c in checks if c.startswith(("vol", "name_risk:", "factor_share"))]


def test_rejects_as_of_mismatch() -> None:
    with pytest.raises(ValueError, match="as_of"):
        _assess(dataclasses.replace(_book(), as_of=AS_OF - timedelta(days=1)))


def test_rejects_duplicate_book_tickers() -> None:
    book = _book()
    with pytest.raises(ValueError, match="duplicate book tickers: 'AAA'"):
        _assess(dataclasses.replace(book, positions=(*book.positions, book.positions[0])))


def test_rejects_sector_mismatch() -> None:
    book = _book()
    moved = dataclasses.replace(book.positions[0], sector="Energy")  # AAA is Tech
    with pytest.raises(ValueError, match="'AAA' sector 'Energy' differs from the snapshot's"):
        _assess(dataclasses.replace(book, positions=(moved, *book.positions[1:])))


@pytest.mark.parametrize("field", ["weight", "beta"])
@pytest.mark.parametrize("value", [math.nan, math.inf])
def test_rejects_non_finite_weight_or_beta(field: str, value: float) -> None:
    book = _book()
    broken = dataclasses.replace(book.positions[0], **{field: value})  # AAA
    with pytest.raises(ValueError, match=f"'AAA' {field} must be finite"):
        _assess(dataclasses.replace(book, positions=(broken, *book.positions[1:])))


def test_rejects_benchmark_not_ending_with_the_snapshot() -> None:
    with pytest.raises(ValueError, match="benchmark must end on"):
        assess_portfolio(_book(), SNAPSHOT, SPY.iloc[:-1])


@pytest.mark.parametrize(
    ("nav", "message"),
    [
        ((NavPoint(AS_OF + timedelta(days=1), 100.0),), "after as_of"),
        ((NavPoint(AS_OF, 100.0), NavPoint(AS_OF, 101.0)), "strictly ascending"),
    ],
    ids=["after-as_of", "malformed"],
)
def test_rejects_bad_nav_history(nav: tuple[NavPoint, ...], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _assess(_empty(), nav=nav)


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"limits": RiskLimits(net=0)}, "net must be > 0"),
        ({"breakers": BreakerParams(drawdown=1.0)}, "0 < drawdown < 1"),
        ({"risk": RiskParams(lookback=1, min_obs=1)}, "2 <= min_obs <= lookback"),
    ],
    ids=["limits", "breakers", "risk"],
)
def test_invalid_params_are_rejected_even_for_an_empty_book(
    params: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _assess(_empty(), **params)


def test_flat_benchmark_propagates_the_factor_model_error() -> None:
    flat = pd.Series(400.0, index=SNAPSHOT.closes.index)
    with pytest.raises(ValueError, match="zero variance"):
        assess_portfolio(_book(), SNAPSHOT, flat)
