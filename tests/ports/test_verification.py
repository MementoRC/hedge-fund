"""Tests for the verification port, exercised through the in-memory fake."""

from __future__ import annotations

import dataclasses
from datetime import UTC, date, datetime, timedelta

import pytest

from hedge_fund.domain.verification import VerificationReport, VerifyOutcome
from hedge_fund.ports.verification import VerificationReportStore
from tests.fakes import InMemoryVerificationStore

AT = datetime(2026, 9, 30, 22, 0, tzinfo=UTC)
REPORT = VerificationReport(
    start=date(2026, 9, 1),
    end=date(2026, 9, 30),
    started_at=AT,
    outcome=VerifyOutcome.CLEAN,
    failed_stage=None,
    error=None,
    observations=(),
    summaries=(),
)


def test_in_memory_verification_store_satisfies_the_port() -> None:
    fake = InMemoryVerificationStore()
    store: VerificationReportStore = fake
    store.save(REPORT)
    assert fake.saved == [REPORT]


def test_in_memory_verification_store_refuses_a_repeat_window_and_second() -> None:
    fake = InMemoryVerificationStore()
    fake.save(REPORT)
    with pytest.raises(FileExistsError, match="2026-09-01_2026-09-30_220000"):
        fake.save(dataclasses.replace(REPORT, started_at=AT + timedelta(microseconds=5)))
    other = dataclasses.replace(REPORT, start=date(2026, 9, 2))
    fake.save(other)
    assert fake.saved == [REPORT, other]


def test_in_memory_verification_store_fails_when_told_to_and_keeps_nothing() -> None:
    fake = InMemoryVerificationStore(fail=OSError("disk full"))
    with pytest.raises(OSError, match="disk full"):
        fake.save(REPORT)
    assert fake.saved == []
