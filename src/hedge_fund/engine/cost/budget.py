"""The pre-research budget plan: cut K, then N, until N × K runs fit the budget."""

from __future__ import annotations

import math
from collections.abc import Callable

from hedge_fund.domain.cost import BudgetError, BudgetPlan, Estimate

TOLERANCE = 1e-9  # relative slack: a budget equal to n × k × cost fits despite float rounding


def plan(budget: float, estimate: Estimate, n: int, k: int, min_k: int) -> BudgetPlan:
    """Fit ``n`` names × ``k`` runs × ``estimate.cost`` into ``budget``, in this order.

    (a) BudgetError unless ``budget`` is a finite number > 0. (b) ``n < 1``: skipped,
    reason ``"empty shortlist"``. (c) A zero cost, or a plan that fits, is unchanged.
    (d) Otherwise K drops to the largest value in ``[min(min_k, k), k]`` that fits; if
    none fits, K is that floor and N drops to the largest value in ``[0, n]`` that fits
    (the shortlist's last names go first). N < 1 is skipped. ``min_k`` is the conviction
    ``min_valid``; it is capped by ``k`` here.
    """
    check_budget(budget)
    cost = estimate.cost
    if n < 1:
        return _plan(budget, estimate, n, k, n=n, k_planned=k, reasons=("empty shortlist",))
    limit = budget * (1 + TOLERANCE)

    def fits(names: int, runs: int) -> bool:
        return names * runs * cost <= limit

    if cost == 0 or fits(n, k):
        return _plan(budget, estimate, n, k, n=n, k_planned=k, reasons=())
    floor_k = max(1, min(min_k, k))
    new_k = _largest(k, limit / (n * cost), lambda runs: fits(n, runs))
    new_n = n
    if new_k < floor_k:
        new_k = floor_k
        new_n = _largest(n, limit / (floor_k * cost), lambda names: fits(names, floor_k))
    reasons: list[str] = []
    if new_k != k:
        reasons.append(f"k {k}→{new_k}: {_product(n, k, cost)} > {_usd(budget)}")
    if new_n != n:
        reasons.append(f"n {n}→{new_n}: {_product(n, new_k, cost)} > {_usd(budget)}")
    return _plan(budget, estimate, n, k, n=new_n, k_planned=new_k, reasons=tuple(reasons))


def check_budget(budget: float) -> None:
    """BudgetError unless ``budget`` is a finite number > 0.

    The cycle calls it up front (stage ``inputs``), so a bad budget fails even when stored
    research is reused; ``plan`` checks again. Type check first: bool and NaN would slip
    past a plain ``budget > 0``.
    """
    if (
        isinstance(budget, bool)
        or not isinstance(budget, int | float)
        or not math.isfinite(budget)
        or budget <= 0
    ):
        raise BudgetError(f"budget must be a finite number > 0, got {budget!r}")


def _largest(high: int, quotient: float, fits: Callable[[int], bool]) -> int:
    """The largest ``value <= high`` with ``fits(value)``: ``floor(quotient)`` capped at
    ``high``, then stepped down while a float product still rounds above the budget.

    ``fits(0)`` always holds, so the result is >= 0.
    """
    value = math.floor(min(float(high), quotient))
    while value > 0 and not fits(value):
        value -= 1
    return value


def _plan(
    budget: float,
    estimate: Estimate,
    requested_n: int,
    requested_k: int,
    *,
    n: int,
    k_planned: int,
    reasons: tuple[str, ...],
) -> BudgetPlan:
    return BudgetPlan(
        budget=budget,
        estimate=estimate,
        requested_n=requested_n,
        requested_k=requested_k,
        n=n,
        k=k_planned,
        projected=n * k_planned * estimate.cost,
        skipped=n < 1,
        reasons=reasons,
    )


def _product(n: int, k: int, cost: float) -> str:
    return f"{n}×{k}×{_usd(cost)}={_usd(n * k * cost)}"


def _usd(value: float) -> str:
    return f"${value:,.2f}"
