"""Tests for the attribution port, exercised through the in-memory fake."""

from __future__ import annotations

import dataclasses
from datetime import UTC, date, datetime, timedelta

import pytest

from hedge_fund.domain.attribution import NO_PNL, AttributeOutcome, AttributionReport
from hedge_fund.ports.attribution import AttributionReportStore
from tests.fakes import InMemoryAttributionStore

AT = datetime(2026, 9, 30, 22, 0, tzinfo=UTC)
REPORT = AttributionReport(
    start=date(2026, 9, 25),
    end=date(2026, 9, 30),
    started_at=AT,
    outcome=AttributeOutcome.CLEAN,
    failed_stage=None,
    error=None,
    days=(),
    by_ticker={},
    by_sector={},
    total=NO_PNL,
    breaks=(),
    no_beta={},
    unregressed_days=(),
    excluded={},
)


def test_in_memory_attribution_store_satisfies_the_port() -> None:
    fake = InMemoryAttributionStore()
    store: AttributionReportStore = fake
    store.save(REPORT)
    assert fake.saved == [REPORT]


def test_in_memory_attribution_store_refuses_a_repeat_window_and_second() -> None:
    fake = InMemoryAttributionStore()
    fake.save(REPORT)
    with pytest.raises(FileExistsError, match="2026-09-25_2026-09-30_220000"):
        fake.save(dataclasses.replace(REPORT, started_at=AT + timedelta(microseconds=5)))
    other = dataclasses.replace(REPORT, start=date(2026, 9, 28))
    fake.save(other)
    assert fake.saved == [REPORT, other]


def test_in_memory_attribution_store_fails_when_told_to_and_keeps_nothing() -> None:
    fake = InMemoryAttributionStore(fail=OSError("disk full"))
    with pytest.raises(OSError, match="disk full"):
        fake.save(REPORT)
    assert fake.saved == []
