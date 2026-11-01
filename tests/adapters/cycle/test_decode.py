"""decode_report: the inverse of jsonable for a CycleReport."""

from __future__ import annotations

import dataclasses
import json
import math
from collections.abc import Iterator
from datetime import UTC, date, datetime
from typing import Any

import pytest

from hedge_fund.adapters._json import jsonable
from hedge_fund.adapters.cycle.decode import decode_report
from hedge_fund.domain.account import AccountSnapshot
from hedge_fund.domain.approval import ApprovalStatus
from hedge_fund.domain.cost import BudgetPlan, Estimate, EstimateSource, Spend
from hedge_fund.domain.cycle import CycleOutcome, CycleReport
from hedge_fund.domain.execution import (
    ExecutionReport,
    OrderRequest,
    SubmitResult,
    SubmitStatus,
)
from hedge_fund.domain.orders import Holding, Order, OrderAction, OrderPlan
from hedge_fund.domain.portfolio import Position, TargetPortfolio
from hedge_fund.domain.research import NameResearch, ResearchResult, RunRecord, Verdict
from hedge_fund.domain.risk import BreakerState, RiskBreakdown, RiskVerdict, Violation
from hedge_fund.domain.shortlist import Candidate, Shortlist, Side
from tests.adapters.cycle.test_json_report import AS_OF, REPORT

assert REPORT.verdict is not None
FINITE = dataclasses.replace(
    REPORT,
    verdict=dataclasses.replace(
        REPORT.verdict, violations=(Violation(check="vol", observed=0.3, limit=0.2, detail="d"),)
    ),
)
EMPTY = dataclasses.replace(
    REPORT,
    account=None,
    shortlist=None,
    research=None,
    target=None,
    verdict=None,
    order_prices=None,
    plan=None,
    decision=None,
    execution=None,
    failed_stage=None,
    error=None,
    planned_client_order_ids=(),
    budget=None,
    spend=None,
)


def _through_json(report: CycleReport) -> dict[str, Any]:
    return json.loads(json.dumps(jsonable(report), allow_nan=False))


def _floats(value: Any) -> Iterator[float]:
    """Every float inside ``value`` (dataclass fields, mapping values, sequence items)."""
    if isinstance(value, float):
        yield value
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):
        for field in dataclasses.fields(value):
            yield from _floats(getattr(value, field.name))
    elif isinstance(value, dict):
        for item in value.values():
            yield from _floats(item)
    elif isinstance(value, tuple):
        for item in value:
            yield from _floats(item)


def _report_with(x: float) -> CycleReport:
    """REPORT's shape with every float-typed field set to ``x``."""
    candidate = Candidate(ticker="AAPL", sector="Tech", side=Side.LONG, depth=1, composite=x)
    order = Order(ticker="AAPL", action=OrderAction.BUY, shares=10, ref_price=x, notional=x)
    run = RunRecord("AAPL", AS_OF, 0, None, "", {}, x, "RuntimeError: x")
    return dataclasses.replace(
        REPORT,
        account=AccountSnapshot(AS_OF, REPORT.started_at, x, (Holding("XOM", -5),)),
        shortlist=Shortlist(AS_OF, (candidate,), {}),
        research=ResearchResult(
            AS_OF, (NameResearch(candidate, (run,), 0, x, x, x, Verdict.INSUFFICIENT),)
        ),
        target=TargetPortfolio(
            AS_OF, (Position("AAPL", "Tech", Side.LONG, x, x, x),), {}, x, x, x, x
        ),
        verdict=RiskVerdict(
            AS_OF,
            False,
            False,
            (Violation("vol", x, x, "d"),),
            RiskBreakdown(x, x, x, x, {"market": x}, {"AAPL": x}, ()),
            BreakerState(x, x, False, False),
        ),
        order_prices={"AAPL": (x, AS_OF)},
        plan=OrderPlan(AS_OF, (order,), {}, {"AAPL": x}, None),
        execution=ExecutionReport(
            AS_OF,
            None,
            (SubmitResult(OrderRequest(order, "hf-x"), SubmitStatus.ERROR, None, "boom"),),
            True,
        ),
        budget=BudgetPlan(x, Estimate(x, EstimateSource.HISTORY, 1), 1, 3, 1, 2, x, False, ()),
        spend=Spend(x, 1, 0, (), None),
    )


def test_the_pinned_report_round_trips() -> None:
    decoded = decode_report(_through_json(REPORT))
    assert jsonable(decoded) == jsonable(REPORT)
    assert decode_report(_through_json(FINITE)) == FINITE


def test_a_report_with_no_sections_round_trips() -> None:
    assert decode_report(_through_json(EMPTY)) == EMPTY


def test_decoded_values_have_their_domain_types() -> None:
    decoded = decode_report(_through_json(FINITE))
    assert decoded.outcome is CycleOutcome.SUBMITTED
    assert decoded.started_at == datetime(2026, 9, 30, 22, 5, 9, tzinfo=UTC)
    assert decoded.started_at.utcoffset() is not None
    assert decoded.account is not None
    assert decoded.account.holdings == (Holding(ticker="XOM", shares=-5),)
    assert decoded.order_prices == {"AAPL": (100.5, AS_OF), "XOM": (110.25, date(2026, 9, 29))}
    assert decoded.plan is not None
    assert decoded.plan.orders[0].action is OrderAction.BUY
    assert decoded.decision is not None
    assert decoded.decision.status is ApprovalStatus.APPROVED
    assert decoded.execution is not None
    assert decoded.execution.results[0].status is SubmitStatus.ACCEPTED
    assert decoded.shortlist is not None
    assert decoded.shortlist.candidates[0].side is Side.LONG
    assert decoded.planned_client_order_ids == ("hf-2026-09-30-AAPL-buy-10",)


@pytest.mark.parametrize("x", [math.nan, math.inf, -math.inf], ids=["nan", "inf", "-inf"])
def test_non_finite_strings_decode_back_for_every_float_field(x: float) -> None:
    original = _report_with(x)
    decoded = decode_report(_through_json(original))
    expected = list(_floats(original))
    found = list(_floats(decoded))
    assert len(expected) == 34  # every float-typed field of the report types, once each
    assert len(found) == len(expected)
    if math.isnan(x):
        assert all(math.isnan(value) for value in found)
    else:
        assert found == expected


def test_a_report_written_before_budgets_loads_with_none() -> None:
    body = _through_json(FINITE)
    del body["budget"], body["spend"]
    decoded = decode_report(body)
    assert (decoded.budget, decoded.spend) == (None, None)
    assert decoded == dataclasses.replace(FINITE, budget=None, spend=None)


def test_the_budget_and_spend_decode_to_their_types() -> None:
    decoded = decode_report(_through_json(FINITE))
    assert decoded.budget is not None
    assert decoded.budget.estimate.source is EstimateSource.HISTORY
    assert decoded.budget.reasons == ("k 3→2: 40×3×$0.80=$96.00 > $80.00",)
    assert decoded.spend is not None
    assert FINITE.spend is not None
    assert decoded.spend.usage
    assert decoded.spend.usage == FINITE.spend.usage


@pytest.mark.parametrize(
    ("change", "error"),
    [
        pytest.param(lambda body: body.pop("outcome"), KeyError, id="missing-key"),
        pytest.param(lambda body: body.update(outcome="bogus"), ValueError, id="bad-enum"),
        pytest.param(
            lambda body: body.update(started_at="2026-09-30T22:05:09"), ValueError, id="naive"
        ),
        pytest.param(lambda body: body.update(started_at=None), TypeError, id="null-instant"),
        pytest.param(
            lambda body: body["account"].update(equity="lots"), ValueError, id="bad-float"
        ),
        pytest.param(
            lambda body: body["budget"].pop("estimate"), KeyError, id="budget-missing-estimate"
        ),
        pytest.param(
            lambda body: body["budget"]["estimate"].update(source="bogus"),
            ValueError,
            id="budget-bad-source",
        ),
        pytest.param(
            lambda body: body["budget"].update(reasons=7),
            TypeError,
            id="budget-reasons-not-list",
        ),
    ],
)
def test_a_malformed_body_raises(change: Any, error: type[Exception]) -> None:
    body = _through_json(FINITE)
    change(body)
    with pytest.raises(error):
        decode_report(body)
