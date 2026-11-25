"""Pipeline cycle types: run options, the outcome, and the audit report of one cycle."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime
from enum import StrEnum

from hedge_fund.domain.account import AccountSnapshot
from hedge_fund.domain.approval import ApprovalDecision
from hedge_fund.domain.cost import BudgetPlan, Spend
from hedge_fund.domain.execution import ExecutionReport
from hedge_fund.domain.orders import OrderPlan
from hedge_fund.domain.portfolio import TargetPortfolio
from hedge_fund.domain.research import ResearchResult
from hedge_fund.domain.risk import RiskVerdict
from hedge_fund.domain.shortlist import Shortlist


@dataclass(frozen=True)
class CycleOptions:
    """Per-run switches; ``any_time`` skips the submission-window check.

    ``budget`` (USD) caps this cycle's planned research spend: K, then N, are cut before any
    LLM call (7c). None researches the whole shortlist ``k`` times.
    """

    dry_run: bool = False
    fresh_research: bool = False
    k: int = 3
    max_workers: int = 4
    any_time: bool = False
    budget: float | None = None


class CycleOutcome(StrEnum):
    """How one cycle ended.

    ERROR and INTERRUPTED are partial reports of a cycle that raised (an ``Exception`` or a
    ``KeyboardInterrupt``); every other outcome is a normal completion.
    """

    SUBMITTED = "submitted"
    NOTHING_TO_DO = "nothing-to-do"
    REJECTED = "rejected"
    BLOCKED = "blocked"
    HALTED_PREFLIGHT = "halted-preflight"
    DRY_RUN = "dry-run"
    ERROR = "error"
    INTERRUPTED = "interrupted"


@dataclass(frozen=True)
class CycleReport:
    """Audit record of one cycle; fields not reached are None, ``{}``, False or ``()``.

    ``started_at`` is the tz-aware ``now()`` at the start. ``failed_stage`` and ``error``
    (``"<ExceptionType>: <message>"``) are set only for ERROR and INTERRUPTED.
    ``order_prices`` maps ticker -> (price, close date). ``execution_interrupted`` is True
    when ``execute_plan`` raised; ``execution`` is then None and
    ``planned_client_order_ids`` holds the plan's ids in submission order. ``budget`` is the
    pre-research plan (None without a budget, or when stored research was reused); ``spend``
    covers the research runs this cycle made (None until research finished). Not validated.
    """

    as_of: date
    started_at: datetime
    dry_run: bool
    outcome: CycleOutcome
    failed_stage: str | None
    error: str | None
    account: AccountSnapshot | None
    price_unresolved: Mapping[str, str]
    fundamentals_unresolved: Mapping[str, str]
    shortlist: Shortlist | None
    research: ResearchResult | None
    research_reused: bool
    target: TargetPortfolio | None
    verdict: RiskVerdict | None
    order_prices: Mapping[str, tuple[float, date]] | None
    plan: OrderPlan | None
    decision: ApprovalDecision | None
    execution: ExecutionReport | None
    execution_interrupted: bool
    planned_client_order_ids: tuple[str, ...]
    budget: BudgetPlan | None = None
    spend: Spend | None = None
