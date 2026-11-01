"""Tests for the budget plan: K cut first, then N from the bottom of the shortlist."""

from __future__ import annotations

import math

import pytest

from hedge_fund.domain.cost import BudgetError, BudgetPlan, Estimate, EstimateSource
from hedge_fund.engine.cost import check_budget, plan
from hedge_fund.engine.cost.budget import _largest

EIGHTY_CENTS = Estimate(cost=0.8, source=EstimateSource.HISTORY, n_samples=30)


def _plan(
    budget: float, *, n: int = 40, k: int = 3, min_k: int = 2, cost: Estimate = EIGHTY_CENTS
) -> BudgetPlan:
    return plan(budget, cost, n=n, k=k, min_k=min_k)


def test_a_plan_that_fits_is_unchanged() -> None:
    out = _plan(100.0)
    assert (out.budget, out.estimate) == (100.0, EIGHTY_CENTS)
    assert (out.requested_n, out.requested_k, out.n, out.k) == (40, 3, 40, 3)
    assert out.projected == pytest.approx(96.0)
    assert (out.skipped, out.reasons) == (False, ())


def test_k_is_cut_first() -> None:
    out = _plan(80.0)
    assert (out.n, out.k, out.skipped) == (40, 2, False)
    assert out.projected == pytest.approx(64.0)
    assert out.reasons == ("k 3→2: 40×3×$0.80=$96.00 > $80.00",)


def test_k_may_go_below_two_when_min_k_allows() -> None:
    out = _plan(40.0, min_k=1)
    assert (out.n, out.k) == (40, 1)
    assert out.reasons == ("k 3→1: 40×3×$0.80=$96.00 > $40.00",)


def test_then_n_is_cut() -> None:
    out = _plan(40.0)
    assert (out.n, out.k, out.skipped) == (25, 2, False)
    assert out.projected == pytest.approx(40.0)
    assert out.reasons == (
        "k 3→2: 40×3×$0.80=$96.00 > $40.00",
        "n 40→25: 40×2×$0.80=$64.00 > $40.00",
    )


def test_no_name_fitting_skips_research() -> None:
    out = _plan(1.0)
    assert (out.n, out.k, out.skipped, out.projected) == (0, 2, True, 0.0)
    assert out.reasons == (
        "k 3→2: 40×3×$0.80=$96.00 > $1.00",
        "n 40→0: 40×2×$0.80=$64.00 > $1.00",
    )


def test_min_k_is_capped_by_the_requested_k() -> None:
    out = _plan(10.0, k=1)
    assert (out.n, out.k) == (12, 1)
    assert out.reasons == ("n 40→12: 40×1×$0.80=$32.00 > $10.00",)


def test_min_k_comes_from_the_caller() -> None:
    out = _plan(80.0, min_k=3)
    assert (out.n, out.k) == (33, 3)
    assert out.projected == pytest.approx(79.2)
    assert out.reasons == ("n 40→33: 40×3×$0.80=$96.00 > $80.00",)


def test_a_free_estimate_changes_nothing() -> None:
    free = Estimate(cost=0.0, source=EstimateSource.FALLBACK, n_samples=0)
    out = _plan(1.0, cost=free)
    assert (out.n, out.k, out.projected, out.skipped, out.reasons) == (40, 3, 0.0, False, ())


def test_an_empty_shortlist_is_skipped() -> None:
    out = _plan(80.0, n=0)
    assert (out.n, out.k, out.projected, out.skipped) == (0, 3, 0.0, True)
    assert out.reasons == ("empty shortlist",)


@pytest.mark.parametrize(
    "budget",
    [
        pytest.param(0.0, id="zero"),
        pytest.param(-1.0, id="negative"),
        pytest.param(math.nan, id="nan"),
        pytest.param(math.inf, id="inf"),
        pytest.param(-math.inf, id="-inf"),
    ],
)
def test_a_budget_must_be_finite_and_positive(budget: float) -> None:
    with pytest.raises(BudgetError, match="budget must be a finite number > 0"):
        _plan(budget)


def test_the_budget_is_checked_before_the_shortlist() -> None:
    with pytest.raises(BudgetError, match="budget must be a finite number > 0"):
        _plan(0.0, n=0)


@pytest.mark.parametrize("budget", [pytest.param(True, id="bool"), pytest.param(-0.01, id="neg")])
def test_check_budget_rejects_on_its_own(budget: float) -> None:
    with pytest.raises(BudgetError, match="budget must be a finite number > 0"):
        check_budget(budget)


def test_check_budget_accepts_a_positive_number() -> None:
    check_budget(0.01)
    check_budget(5)


def test_a_budget_equal_to_the_product_fits() -> None:
    full = _plan(96.0)
    assert (full.n, full.k) == (40, 3)
    dime = Estimate(cost=0.1, source=EstimateSource.HISTORY, n_samples=1)
    assert 1 * 3 * 0.1 > 0.3  # float rounding puts the product just above the budget
    out = plan(0.3, dime, n=1, k=3, min_k=2)
    assert (out.n, out.k, out.reasons) == (1, 3, ())


def test_largest_steps_down_from_the_floor_until_it_fits() -> None:
    assert _largest(5, 9.7, lambda value: value <= 3) == 3
    assert _largest(5, 2.9, lambda value: True) == 2
    assert _largest(4, 3.0, lambda value: False) == 0
