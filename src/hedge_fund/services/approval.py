"""Layer 6c approval: show a risk-assessed order plan to a human and record the answer.

Calling execute_plan on APPROVED and persisting decisions are 6d's job.
"""

from __future__ import annotations

from hedge_fund.domain.approval import ApprovalDecision, ApprovalStatus
from hedge_fund.domain.orders import BLOCKED_HALT, BLOCKED_VETO, OrderPlan
from hedge_fund.domain.risk import RiskVerdict
from hedge_fund.engine.approval import plan_fingerprint, render_proposal
from hedge_fund.ports.approval import ApprovalGate


def request_approval(verdict: RiskVerdict, plan: OrderPlan, gate: ApprovalGate) -> ApprovalDecision:
    """Offer ``plan`` to ``gate`` unless it is blocked or empty; return the decision.

    Raises ValueError, before the gate is called, when the verdict and plan disagree on
    ``as_of`` or on ``blocked`` (which must be what ``generate_orders`` derives from the
    verdict), or when ``prepare_submission`` rejects the plan. A blocked plan is BLOCKED and
    an empty one NOTHING_TO_DO, without calling the gate: the risk veto stands. Otherwise
    the gate decides: only ``True`` is APPROVED, anything else REJECTED. Exceptions from the
    gate propagate.
    """
    if verdict.as_of != plan.as_of:
        raise ValueError(f"verdict as_of {verdict.as_of} does not match plan as_of {plan.as_of}")
    expected = _blocked(verdict)
    if plan.blocked != expected:
        raise ValueError(f"plan blocked {plan.blocked!r} does not match the verdict ({expected!r})")
    summary = render_proposal(verdict, plan)
    fingerprint = plan_fingerprint(plan)
    if plan.blocked is not None:
        status = ApprovalStatus.BLOCKED
    elif not plan.orders:
        status = ApprovalStatus.NOTHING_TO_DO
    elif gate.confirm(summary) is True:
        status = ApprovalStatus.APPROVED
    else:
        status = ApprovalStatus.REJECTED
    return ApprovalDecision(
        as_of=plan.as_of, status=status, plan_fingerprint=fingerprint, summary=summary
    )


def _blocked(verdict: RiskVerdict) -> str | None:
    """The ``blocked`` value generate_orders derives: halt outranks a plain veto."""
    if verdict.halt:
        return BLOCKED_HALT
    if not verdict.approved:
        return BLOCKED_VETO
    return None
