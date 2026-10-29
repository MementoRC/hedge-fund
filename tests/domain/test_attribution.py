"""Tests for the realised attribution domain types."""

from __future__ import annotations

import dataclasses
from datetime import UTC, date, datetime

import pytest

from hedge_fund.domain.attribution import (
    DIVIDEND_NOTE,
    NO_PNL,
    UNKNOWN_SECTOR,
    AttributeOutcome,
    AttributionReport,
    AttributionTotals,
    DayAttribution,
)

AT = datetime(2026, 9, 30, 22, 0, tzinfo=UTC)
ROW = DayAttribution(
    day=date(2026, 9, 29),
    ticker="AAA",
    sector="Tech",
    prior_shares=10,
    market=15.0,
    sector_pnl=4.0,
    residual=1.0,
    trading=2.0,
    regressed=True,
)
REPORT = AttributionReport(
    start=date(2026, 9, 25),
    end=date(2026, 9, 30),
    started_at=AT,
    outcome=AttributeOutcome.CLEAN,
    failed_stage=None,
    error=None,
    days=(ROW,),
    by_ticker={},
    by_sector={},
    total=NO_PNL,
    breaks=(),
    no_beta={},
    unregressed_days=(),
    excluded={},
)


def test_outcome_values() -> None:
    assert [outcome.value for outcome in AttributeOutcome] == ["clean", "break", "failed"]


def test_constants() -> None:
    assert UNKNOWN_SECTOR == "Unknown"
    assert AttributionTotals(market=0.0, sector=0.0, residual=0.0, trading=0.0, total=0.0) == NO_PNL
    assert "dividend-adjusted" in DIVIDEND_NOTE


def test_day_attribution_field_order() -> None:
    assert [field.name for field in dataclasses.fields(DayAttribution)] == [
        "day",
        "ticker",
        "sector",
        "prior_shares",
        "market",
        "sector_pnl",
        "residual",
        "trading",
        "regressed",
    ]


def test_totals_field_order() -> None:
    assert [field.name for field in dataclasses.fields(AttributionTotals)] == [
        "market",
        "sector",
        "residual",
        "trading",
        "total",
    ]


def test_report_field_order() -> None:
    assert [field.name for field in dataclasses.fields(AttributionReport)] == [
        "start",
        "end",
        "started_at",
        "outcome",
        "failed_stage",
        "error",
        "days",
        "by_ticker",
        "by_sector",
        "total",
        "breaks",
        "no_beta",
        "unregressed_days",
        "excluded",
    ]


def test_types_compare_by_value() -> None:
    assert dataclasses.replace(REPORT) == REPORT
    assert dataclasses.replace(REPORT, outcome=AttributeOutcome.BREAK) != REPORT


@pytest.mark.parametrize(
    ("instance", "field"),
    [
        pytest.param(ROW, "market", id="day"),
        pytest.param(NO_PNL, "total", id="totals"),
        pytest.param(REPORT, "outcome", id="report"),
    ],
)
def test_types_are_frozen(instance: object, field: str) -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(instance, field, None)
