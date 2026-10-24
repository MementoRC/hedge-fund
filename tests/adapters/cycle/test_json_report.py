"""Tests for the JSON cycle report store."""

from __future__ import annotations

import dataclasses
import json
import math
from datetime import UTC, date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from hedge_fund.adapters.cycle.json_report import JsonCycleReportStore
from hedge_fund.domain.account import AccountSnapshot
from hedge_fund.domain.approval import ApprovalDecision, ApprovalStatus
from hedge_fund.domain.cycle import CycleOutcome, CycleReport
from hedge_fund.domain.execution import ExecutionReport, OrderRequest, SubmitResult, SubmitStatus
from hedge_fund.domain.orders import Holding, Order, OrderAction, OrderPlan
from hedge_fund.domain.portfolio import Position, TargetPortfolio
from hedge_fund.domain.research import NameResearch, Rating, ResearchResult, RunRecord, Verdict
from hedge_fund.domain.risk import BreakerState, RiskBreakdown, RiskVerdict, Violation
from hedge_fund.domain.shortlist import Candidate, Shortlist, Side
from hedge_fund.ports.cycle import CycleReportStore

AS_OF = date(2026, 9, 30)
EASTERN = timezone(timedelta(hours=-4))
CANDIDATE = Candidate(ticker="AAPL", sector="Tech", side=Side.LONG, depth=1, composite=1.25)
ORDER = Order(ticker="AAPL", action=OrderAction.BUY, shares=10, ref_price=100.5, notional=1005.0)
CLIENT_ID = "hf-2026-09-30-AAPL-buy-10"

# Every field populated (execution and execution_interrupted together only pin the shape).
REPORT = CycleReport(
    as_of=AS_OF,
    started_at=datetime(2026, 9, 30, 18, 5, 9, tzinfo=EASTERN),  # 22:05:09 UTC
    dry_run=False,
    outcome=CycleOutcome.SUBMITTED,
    failed_stage="execution",
    error="RuntimeError: boom",
    account=AccountSnapshot(
        as_of=AS_OF,
        observed_at=datetime(2026, 9, 30, 22, 5, 10, tzinfo=UTC),
        equity=1_000_000.0,
        holdings=(Holding(ticker="XOM", shares=-5),),
    ),
    price_unresolved={"BRK.B": "no_prices"},
    fundamentals_unresolved={"XOM": "no_cik"},
    shortlist=Shortlist(as_of=AS_OF, candidates=(CANDIDATE,), trimmed={"MSFT": "cap"}),
    research=ResearchResult(
        as_of=AS_OF,
        names=(
            NameResearch(
                candidate=CANDIDATE,
                runs=(
                    RunRecord(
                        ticker="AAPL",
                        as_of=AS_OF,
                        run_index=0,
                        rating=Rating.BUY,
                        decision="buy",
                        reports={"market_report": "m"},
                        seconds=1.5,
                        error=None,
                    ),
                ),
                valid=1,
                mean=2.0,
                std=0.0,
                agreement=1.0,
                verdict=Verdict.CONFIRM,
            ),
        ),
    ),
    research_reused=True,
    target=TargetPortfolio(
        as_of=AS_OF,
        positions=(
            Position(
                ticker="AAPL", sector="Tech", side=Side.LONG, weight=0.1, alpha=0.002, beta=1.1
            ),
        ),
        excluded={"XOM": "verdict:neutral"},
        gross=0.1,
        net=0.1,
        beta=0.11,
        expected_vol=0.02,
    ),
    verdict=RiskVerdict(
        as_of=AS_OF,
        approved=True,
        halt=False,
        violations=(
            Violation(check="unassessable:ZZZ", observed=math.nan, limit=math.nan, detail="none"),
            Violation(check="vol", observed=math.inf, limit=-math.inf, detail="pinned"),
        ),
        breakdown=RiskBreakdown(
            total_vol=0.02,
            factor_vol=0.01,
            specific_vol=0.015,
            factor_share=0.25,
            by_factor={"market": 0.0001},
            by_name={"AAPL": 1.0},
            unassessed=("ZZZ",),
        ),
        breakers=BreakerState(
            daily_loss=-0.01, drawdown=0.02, daily_tripped=False, drawdown_tripped=False
        ),
    ),
    order_prices={"AAPL": (100.5, AS_OF), "XOM": (110.25, date(2026, 9, 29))},
    plan=OrderPlan(
        as_of=AS_OF,
        orders=(ORDER,),
        skipped={"XOM": "band"},
        residual_weight={"AAPL": -0.001, "XOM": 0.0},
        blocked=None,
    ),
    decision=ApprovalDecision(
        as_of=AS_OF, status=ApprovalStatus.APPROVED, plan_fingerprint="abc123", summary="1 order"
    ),
    execution=ExecutionReport(
        as_of=AS_OF,
        blocked=None,
        results=(
            SubmitResult(
                request=OrderRequest(order=ORDER, client_order_id=CLIENT_ID),
                status=SubmitStatus.ACCEPTED,
                broker_order_id="b-1",
                reason=None,
            ),
        ),
        opening_halted=False,
    ),
    execution_interrupted=True,
    planned_client_order_ids=(CLIENT_ID,),
)

ORDER_JSON = {
    "ticker": "AAPL",
    "action": "buy",
    "shares": 10,
    "ref_price": 100.5,
    "notional": 1005.0,
}
CANDIDATE_JSON = {
    "ticker": "AAPL",
    "sector": "Tech",
    "side": "long",
    "depth": 1,
    "composite": 1.25,
}
EXPECTED: dict[str, Any] = {
    "as_of": "2026-09-30",
    "started_at": "2026-09-30T18:05:09-04:00",
    "dry_run": False,
    "outcome": "submitted",
    "failed_stage": "execution",
    "error": "RuntimeError: boom",
    "account": {
        "as_of": "2026-09-30",
        "observed_at": "2026-09-30T22:05:10+00:00",
        "equity": 1000000.0,
        "holdings": [{"ticker": "XOM", "shares": -5}],
    },
    "price_unresolved": {"BRK.B": "no_prices"},
    "fundamentals_unresolved": {"XOM": "no_cik"},
    "shortlist": {
        "as_of": "2026-09-30",
        "candidates": [CANDIDATE_JSON],
        "trimmed": {"MSFT": "cap"},
    },
    "research": {
        "as_of": "2026-09-30",
        "names": [
            {
                "candidate": CANDIDATE_JSON,
                "runs": [
                    {
                        "ticker": "AAPL",
                        "as_of": "2026-09-30",
                        "run_index": 0,
                        "rating": "Buy",
                        "decision": "buy",
                        "reports": {"market_report": "m"},
                        "seconds": 1.5,
                        "error": None,
                    }
                ],
                "valid": 1,
                "mean": 2.0,
                "std": 0.0,
                "agreement": 1.0,
                "verdict": "confirm",
            }
        ],
    },
    "research_reused": True,
    "target": {
        "as_of": "2026-09-30",
        "positions": [
            {
                "ticker": "AAPL",
                "sector": "Tech",
                "side": "long",
                "weight": 0.1,
                "alpha": 0.002,
                "beta": 1.1,
            }
        ],
        "excluded": {"XOM": "verdict:neutral"},
        "gross": 0.1,
        "net": 0.1,
        "beta": 0.11,
        "expected_vol": 0.02,
    },
    "verdict": {
        "as_of": "2026-09-30",
        "approved": True,
        "halt": False,
        "violations": [
            {"check": "unassessable:ZZZ", "observed": "nan", "limit": "nan", "detail": "none"},
            {"check": "vol", "observed": "inf", "limit": "-inf", "detail": "pinned"},
        ],
        "breakdown": {
            "total_vol": 0.02,
            "factor_vol": 0.01,
            "specific_vol": 0.015,
            "factor_share": 0.25,
            "by_factor": {"market": 0.0001},
            "by_name": {"AAPL": 1.0},
            "unassessed": ["ZZZ"],
        },
        "breakers": {
            "daily_loss": -0.01,
            "drawdown": 0.02,
            "daily_tripped": False,
            "drawdown_tripped": False,
        },
    },
    "order_prices": {"AAPL": [100.5, "2026-09-30"], "XOM": [110.25, "2026-09-29"]},
    "plan": {
        "as_of": "2026-09-30",
        "orders": [ORDER_JSON],
        "skipped": {"XOM": "band"},
        "residual_weight": {"AAPL": -0.001, "XOM": 0.0},
        "blocked": None,
    },
    "decision": {
        "as_of": "2026-09-30",
        "status": "approved",
        "plan_fingerprint": "abc123",
        "summary": "1 order",
    },
    "execution": {
        "as_of": "2026-09-30",
        "blocked": None,
        "results": [
            {
                "request": {"order": ORDER_JSON, "client_order_id": CLIENT_ID},
                "status": "accepted",
                "broker_order_id": "b-1",
                "reason": None,
            }
        ],
        "opening_halted": False,
    },
    "execution_interrupted": True,
    "planned_client_order_ids": [CLIENT_ID],
}


def _runs(root: Path) -> list[Path]:
    return sorted((root / "runs").iterdir())


def test_satisfies_the_port_and_writes_the_pinned_shape(tmp_path: Path) -> None:
    store: CycleReportStore = JsonCycleReportStore(tmp_path)
    store.save(REPORT)
    [path] = _runs(tmp_path)
    text = path.read_text(encoding="utf-8")
    assert json.loads(text) == EXPECTED
    assert text == json.dumps(EXPECTED, indent=2)  # key order and indent pinned too


def test_keys_are_the_report_fields_in_order(tmp_path: Path) -> None:
    JsonCycleReportStore(tmp_path).save(REPORT)
    [path] = _runs(tmp_path)
    body = json.loads(path.read_text(encoding="utf-8"))
    assert list(body) == [field.name for field in dataclasses.fields(CycleReport)]


@pytest.mark.parametrize(
    ("started_at", "name"),
    [
        pytest.param(REPORT.started_at, "2026-09-30_220509.json", id="eastern-offset"),
        pytest.param(
            datetime(2026, 10, 1, 1, 2, 3, tzinfo=UTC), "2026-09-30_010203.json", id="utc"
        ),
    ],
)
def test_file_name_is_as_of_and_utc_start_time(
    tmp_path: Path, started_at: datetime, name: str
) -> None:
    JsonCycleReportStore(tmp_path).save(dataclasses.replace(REPORT, started_at=started_at))
    assert [path.name for path in _runs(tmp_path)] == [name]


def test_an_existing_report_is_never_overwritten(tmp_path: Path) -> None:
    store = JsonCycleReportStore(tmp_path)
    store.save(REPORT)
    [path] = _runs(tmp_path)
    first = path.read_text(encoding="utf-8")
    with pytest.raises(FileExistsError, match="2026-09-30_220509.json"):
        store.save(dataclasses.replace(REPORT, outcome=CycleOutcome.ERROR))
    assert path.read_text(encoding="utf-8") == first
    assert _runs(tmp_path) == [path]


def test_a_failed_write_leaves_no_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write_text = Path.write_text

    def partial_then_fail(self: Path, data: str, *args: Any, **kwargs: Any) -> int:
        write_text(self, data[:10], *args, **kwargs)
        raise OSError("disk full")

    monkeypatch.setattr(Path, "write_text", partial_then_fail)
    with pytest.raises(OSError, match="disk full"):
        JsonCycleReportStore(tmp_path).save(REPORT)
    assert _runs(tmp_path) == []


def test_package_exports() -> None:
    import hedge_fund.adapters.cycle as cycle

    assert cycle.__all__ == ["JsonCycleReportStore"]
    assert cycle.JsonCycleReportStore is JsonCycleReportStore
