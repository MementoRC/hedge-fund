"""Tests for the CycleReportStore port, exercised through the in-memory fake."""

from __future__ import annotations

import dataclasses
from datetime import UTC, date, datetime

import pytest

from hedge_fund.domain.cycle import CycleOutcome, CycleReport
from hedge_fund.domain.ledger import ReportLoad
from hedge_fund.ports.cycle import CycleReportStore
from tests.fakes import InMemoryReportStore

REPORT = CycleReport(
    as_of=date(2026, 9, 30),
    started_at=datetime(2026, 9, 30, 22, 0, tzinfo=UTC),
    dry_run=True,
    outcome=CycleOutcome.DRY_RUN,
    failed_stage=None,
    error=None,
    account=None,
    price_unresolved={},
    fundamentals_unresolved={},
    shortlist=None,
    research=None,
    research_reused=False,
    target=None,
    verdict=None,
    order_prices=None,
    plan=None,
    decision=None,
    execution=None,
    execution_interrupted=False,
    planned_client_order_ids=(),
)


def test_in_memory_store_satisfies_the_port() -> None:
    fake = InMemoryReportStore()
    store: CycleReportStore = fake
    store.save(REPORT)
    store.save(REPORT)
    assert fake.saved == [REPORT, REPORT]


def test_in_memory_store_fails_when_told_to_and_keeps_nothing() -> None:
    fake = InMemoryReportStore(fail=OSError("disk full"))
    with pytest.raises(OSError, match="disk full"):
        fake.save(REPORT)
    assert fake.saved == []


def test_in_memory_store_loads_its_reports_by_started_at() -> None:
    later = dataclasses.replace(REPORT, started_at=datetime(2026, 9, 30, 23, 0, tzinfo=UTC))
    fake = InMemoryReportStore([later])
    store: CycleReportStore = fake
    store.save(REPORT)
    fake.unreadable = {"bad.json": "ValueError: x"}
    assert store.load_all() == ReportLoad(
        reports=(REPORT, later), unreadable={"bad.json": "ValueError: x"}
    )
    assert InMemoryReportStore().load_all() == ReportLoad(reports=(), unreadable={})
