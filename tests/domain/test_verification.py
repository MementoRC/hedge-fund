"""Tests for the thesis and rating verification domain types."""

from __future__ import annotations

import dataclasses
from datetime import UTC, date, datetime

import pytest

from hedge_fund.domain.verification import (
    HORIZONS,
    Bucket,
    ExclusionReason,
    HorizonSummary,
    Observation,
    ObservationStatus,
    VerificationReport,
    VerifyOutcome,
)
from tests.factories import make_observation

AT = datetime(2026, 9, 30, 22, 0, tzinfo=UTC)
ROW = make_observation()
BUCKET = Bucket(key="all", n=1, hits=1, hit_rate=1.0, mean_excess=0.05, median_excess=0.05)
ZERO = Bucket(key="all", n=0, hits=0, hit_rate=None, mean_excess=None, median_excess=None)
SUMMARY = HorizonSummary(
    horizon=5,
    scored=1,
    pending=0,
    excluded=0,
    overall=BUCKET,
    by_verdict={"confirm": BUCKET},
    by_score={"1": BUCKET},
    by_sector={"Tech": BUCKET},
    executed=ZERO,
    ic=None,
    no_call_mean_abs_excess=None,
)
REPORT = VerificationReport(
    start=date(2026, 9, 1),
    end=date(2026, 9, 30),
    started_at=AT,
    outcome=VerifyOutcome.CLEAN,
    failed_stage=None,
    error=None,
    observations=(ROW,),
    summaries=(SUMMARY,),
)


def test_horizons() -> None:
    assert HORIZONS == (5, 21, 63)


def test_enum_values() -> None:
    assert [status.value for status in ObservationStatus] == ["scored", "pending", "excluded"]
    assert [reason.value for reason in ExclusionReason] == ["no_price", "insufficient_research"]
    assert [outcome.value for outcome in VerifyOutcome] == ["clean", "gaps", "failed"]


def test_observation_field_order() -> None:
    assert [field.name for field in dataclasses.fields(Observation)] == [
        "as_of",
        "ticker",
        "sector",
        "side",
        "verdict",
        "mean",
        "std",
        "agreement",
        "valid",
        "horizon",
        "base_date",
        "end_date",
        "name_return",
        "spy_return",
        "aligned_excess",
        "hit",
        "executed",
        "status",
        "reason",
    ]


def test_bucket_field_order() -> None:
    assert [field.name for field in dataclasses.fields(Bucket)] == [
        "key",
        "n",
        "hits",
        "hit_rate",
        "mean_excess",
        "median_excess",
    ]


def test_summary_field_order() -> None:
    assert [field.name for field in dataclasses.fields(HorizonSummary)] == [
        "horizon",
        "scored",
        "pending",
        "excluded",
        "overall",
        "by_verdict",
        "by_score",
        "by_sector",
        "executed",
        "ic",
        "no_call_mean_abs_excess",
    ]


def test_report_field_order() -> None:
    assert [field.name for field in dataclasses.fields(VerificationReport)] == [
        "start",
        "end",
        "started_at",
        "outcome",
        "failed_stage",
        "error",
        "observations",
        "summaries",
    ]


def test_types_compare_by_value() -> None:
    assert dataclasses.replace(REPORT) == REPORT
    assert dataclasses.replace(REPORT, outcome=VerifyOutcome.GAPS) != REPORT


@pytest.mark.parametrize(
    ("instance", "field"),
    [
        pytest.param(ROW, "hit", id="observation"),
        pytest.param(BUCKET, "n", id="bucket"),
        pytest.param(SUMMARY, "scored", id="summary"),
        pytest.param(REPORT, "outcome", id="report"),
    ],
)
def test_types_are_frozen(instance: object, field: str) -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(instance, field, None)
