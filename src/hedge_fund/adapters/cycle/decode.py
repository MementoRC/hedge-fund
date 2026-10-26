"""Cycle reports back from JSON: ``decode_report`` is the inverse of ``jsonable``.

Every float-typed field goes through ``decode_float``, so the ``"nan"`` / ``"inf"`` /
``"-inf"`` strings ``jsonable`` writes come back as floats. Research names reuse the
research store's decoders. A malformed body raises KeyError, TypeError or ValueError.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import date, datetime
from typing import Any, TypeVar

from hedge_fund.adapters._json import decode_float
from hedge_fund.adapters.research.store import decode_candidate, decode_name
from hedge_fund.domain.account import AccountSnapshot
from hedge_fund.domain.approval import ApprovalDecision, ApprovalStatus
from hedge_fund.domain.cycle import CycleOutcome, CycleReport
from hedge_fund.domain.execution import ExecutionReport, OrderRequest, SubmitResult, SubmitStatus
from hedge_fund.domain.orders import Holding, Order, OrderAction, OrderPlan
from hedge_fund.domain.portfolio import Position, TargetPortfolio
from hedge_fund.domain.research import ResearchResult
from hedge_fund.domain.risk import BreakerState, RiskBreakdown, RiskVerdict, Violation
from hedge_fund.domain.shortlist import Shortlist, Side

_T = TypeVar("_T")


def decode_report(obj: Mapping[str, Any]) -> CycleReport:
    """The CycleReport whose ``jsonable`` form is ``obj``; ``started_at`` must carry an offset."""
    return CycleReport(
        as_of=date.fromisoformat(obj["as_of"]),
        started_at=_instant(obj["started_at"]),
        dry_run=obj["dry_run"],
        outcome=CycleOutcome(obj["outcome"]),
        failed_stage=obj["failed_stage"],
        error=obj["error"],
        account=_optional(obj["account"], _account),
        price_unresolved=dict(obj["price_unresolved"]),
        fundamentals_unresolved=dict(obj["fundamentals_unresolved"]),
        shortlist=_optional(obj["shortlist"], _shortlist),
        research=_optional(obj["research"], _research),
        research_reused=obj["research_reused"],
        target=_optional(obj["target"], _target),
        verdict=_optional(obj["verdict"], _verdict),
        order_prices=_optional(obj["order_prices"], _order_prices),
        plan=_optional(obj["plan"], _plan),
        decision=_optional(obj["decision"], _decision),
        execution=_optional(obj["execution"], _execution),
        execution_interrupted=obj["execution_interrupted"],
        planned_client_order_ids=tuple(obj["planned_client_order_ids"]),
    )


def _optional(value: Any, decode: Callable[[Any], _T]) -> _T | None:
    return None if value is None else decode(value)


def _instant(text: str) -> datetime:
    instant = datetime.fromisoformat(text)
    if instant.utcoffset() is None:
        raise ValueError(f"{text!r} has no UTC offset")
    return instant


def _floats(obj: Mapping[str, Any]) -> dict[str, float]:
    return {key: decode_float(value) for key, value in obj.items()}


def _account(obj: Mapping[str, Any]) -> AccountSnapshot:
    return AccountSnapshot(
        as_of=date.fromisoformat(obj["as_of"]),
        observed_at=_instant(obj["observed_at"]),
        equity=decode_float(obj["equity"]),
        holdings=tuple(
            Holding(ticker=item["ticker"], shares=item["shares"]) for item in obj["holdings"]
        ),
    )


def _shortlist(obj: Mapping[str, Any]) -> Shortlist:
    return Shortlist(
        as_of=date.fromisoformat(obj["as_of"]),
        candidates=tuple(decode_candidate(item) for item in obj["candidates"]),
        trimmed=dict(obj["trimmed"]),
    )


def _research(obj: Mapping[str, Any]) -> ResearchResult:
    return ResearchResult(
        as_of=date.fromisoformat(obj["as_of"]),
        names=tuple(decode_name(item) for item in obj["names"]),
    )


def _target(obj: Mapping[str, Any]) -> TargetPortfolio:
    return TargetPortfolio(
        as_of=date.fromisoformat(obj["as_of"]),
        positions=tuple(
            Position(
                ticker=item["ticker"],
                sector=item["sector"],
                side=Side(item["side"]),
                weight=decode_float(item["weight"]),
                alpha=decode_float(item["alpha"]),
                beta=decode_float(item["beta"]),
            )
            for item in obj["positions"]
        ),
        excluded=dict(obj["excluded"]),
        gross=decode_float(obj["gross"]),
        net=decode_float(obj["net"]),
        beta=decode_float(obj["beta"]),
        expected_vol=decode_float(obj["expected_vol"]),
    )


def _verdict(obj: Mapping[str, Any]) -> RiskVerdict:
    breakdown, breakers = obj["breakdown"], obj["breakers"]
    return RiskVerdict(
        as_of=date.fromisoformat(obj["as_of"]),
        approved=obj["approved"],
        halt=obj["halt"],
        violations=tuple(
            Violation(
                check=item["check"],
                observed=decode_float(item["observed"]),
                limit=decode_float(item["limit"]),
                detail=item["detail"],
            )
            for item in obj["violations"]
        ),
        breakdown=RiskBreakdown(
            total_vol=decode_float(breakdown["total_vol"]),
            factor_vol=decode_float(breakdown["factor_vol"]),
            specific_vol=decode_float(breakdown["specific_vol"]),
            factor_share=decode_float(breakdown["factor_share"]),
            by_factor=_floats(breakdown["by_factor"]),
            by_name=_floats(breakdown["by_name"]),
            unassessed=tuple(breakdown["unassessed"]),
        ),
        breakers=BreakerState(
            daily_loss=decode_float(breakers["daily_loss"]),
            drawdown=decode_float(breakers["drawdown"]),
            daily_tripped=breakers["daily_tripped"],
            drawdown_tripped=breakers["drawdown_tripped"],
        ),
    )


def _order_prices(obj: Mapping[str, Any]) -> dict[str, tuple[float, date]]:
    return {
        ticker: (decode_float(price), date.fromisoformat(day))
        for ticker, (price, day) in obj.items()
    }


def _order(obj: Mapping[str, Any]) -> Order:
    return Order(
        ticker=obj["ticker"],
        action=OrderAction(obj["action"]),
        shares=obj["shares"],
        ref_price=decode_float(obj["ref_price"]),
        notional=decode_float(obj["notional"]),
    )


def _plan(obj: Mapping[str, Any]) -> OrderPlan:
    return OrderPlan(
        as_of=date.fromisoformat(obj["as_of"]),
        orders=tuple(_order(item) for item in obj["orders"]),
        skipped=dict(obj["skipped"]),
        residual_weight=_floats(obj["residual_weight"]),
        blocked=obj["blocked"],
    )


def _decision(obj: Mapping[str, Any]) -> ApprovalDecision:
    return ApprovalDecision(
        as_of=date.fromisoformat(obj["as_of"]),
        status=ApprovalStatus(obj["status"]),
        plan_fingerprint=obj["plan_fingerprint"],
        summary=obj["summary"],
    )


def _execution(obj: Mapping[str, Any]) -> ExecutionReport:
    return ExecutionReport(
        as_of=date.fromisoformat(obj["as_of"]),
        blocked=obj["blocked"],
        results=tuple(
            SubmitResult(
                request=OrderRequest(
                    order=_order(item["request"]["order"]),
                    client_order_id=item["request"]["client_order_id"],
                ),
                status=SubmitStatus(item["status"]),
                broker_order_id=item["broker_order_id"],
                reason=item["reason"],
            )
            for item in obj["results"]
        ),
        opening_halted=obj["opening_halted"],
    )
