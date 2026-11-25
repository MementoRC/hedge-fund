"""Tests for the pipeline cycle domain types."""

from __future__ import annotations

import dataclasses
from datetime import UTC, date, datetime

import pytest

from hedge_fund.domain.cycle import CycleOptions, CycleOutcome, CycleReport

REPORT = CycleReport(
    as_of=date(2026, 9, 30),
    started_at=datetime(2026, 9, 30, 22, 0, tzinfo=UTC),
    dry_run=True,
    outcome=CycleOutcome.DRY_RUN,
    failed_stage=None,
    error=None,
    account=None,
    price_unresolved={},
    fundamentals_unresolved={"XOM": "no_cik"},
    shortlist=None,
    research=None,
    research_reused=False,
    target=None,
    verdict=None,
    order_prices={"AAPL": (101.5, date(2026, 9, 30))},
    plan=None,
    decision=None,
    execution=None,
    execution_interrupted=False,
    planned_client_order_ids=(),
)


def test_options_defaults() -> None:
    assert CycleOptions() == CycleOptions(
        dry_run=False, fresh_research=False, k=3, max_workers=4, any_time=False, budget=None
    )
    assert REPORT.budget is None
    assert REPORT.spend is None


def test_outcome_values() -> None:
    assert [outcome.value for outcome in CycleOutcome] == [
        "submitted",
        "nothing-to-do",
        "rejected",
        "blocked",
        "halted-preflight",
        "dry-run",
        "error",
        "interrupted",
    ]


def test_report_compares_by_value() -> None:
    assert dataclasses.replace(REPORT) == REPORT
    assert dataclasses.replace(REPORT, failed_stage="data") != REPORT


def test_report_field_order() -> None:
    assert [field.name for field in dataclasses.fields(CycleReport)] == [
        "as_of",
        "started_at",
        "dry_run",
        "outcome",
        "failed_stage",
        "error",
        "account",
        "price_unresolved",
        "fundamentals_unresolved",
        "shortlist",
        "research",
        "research_reused",
        "target",
        "verdict",
        "order_prices",
        "plan",
        "decision",
        "execution",
        "execution_interrupted",
        "planned_client_order_ids",
        "budget",
        "spend",
    ]


@pytest.mark.parametrize(
    ("instance", "field"),
    [
        pytest.param(CycleOptions(), "k", id="options"),
        pytest.param(REPORT, "outcome", id="report"),
    ],
)
def test_types_are_frozen(instance: object, field: str) -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(instance, field, None)
