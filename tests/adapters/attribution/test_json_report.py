"""Tests for the JSON attribution report store."""

from __future__ import annotations

import dataclasses
import json
import math
from datetime import UTC, date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from hedge_fund.adapters.attribution.json_report import JsonAttributionReportStore
from hedge_fund.domain.attribution import (
    AttributeOutcome,
    AttributionReport,
    AttributionTotals,
    DayAttribution,
)
from hedge_fund.domain.ledger import ReconciliationBreak
from hedge_fund.ports.attribution import AttributionReportStore

EASTERN = timezone(timedelta(hours=-4))
TUE, WED = date(2026, 9, 29), date(2026, 9, 30)
TECH = AttributionTotals(market=15.0, sector=4.0, residual=1.0, trading=2.0, total=22.0)

# Every field populated; the overall total pins the non-finite strings.
REPORT = AttributionReport(
    start=TUE,
    end=WED,
    started_at=datetime(2026, 9, 30, 18, 0, 5, tzinfo=EASTERN),  # 22:00:05 UTC
    outcome=AttributeOutcome.BREAK,
    failed_stage=None,
    error=None,
    days=(DayAttribution(TUE, "AAA", "Tech", 10, 15.0, 4.0, 1.0, 2.0, True),),
    by_ticker={"AAA": TECH},
    by_sector={"Tech": TECH},
    total=AttributionTotals(
        market=15.0, sector=4.0, residual=math.inf, trading=2.0, total=math.nan
    ),
    breaks=(ReconciliationBreak(ticker="KKK", ledger_shares=5, broker_shares=7),),
    no_beta={"ZZZ": (TUE, WED)},
    unregressed_days=(WED,),
    excluded={"GAP": "no close on 2026-09-30"},
)
TECH_JSON = {"market": 15.0, "sector": 4.0, "residual": 1.0, "trading": 2.0, "total": 22.0}
EXPECTED: dict[str, Any] = {
    "start": "2026-09-29",
    "end": "2026-09-30",
    "started_at": "2026-09-30T18:00:05-04:00",
    "outcome": "break",
    "failed_stage": None,
    "error": None,
    "days": [
        {
            "day": "2026-09-29",
            "ticker": "AAA",
            "sector": "Tech",
            "prior_shares": 10,
            "market": 15.0,
            "sector_pnl": 4.0,
            "residual": 1.0,
            "trading": 2.0,
            "regressed": True,
        }
    ],
    "by_ticker": {"AAA": TECH_JSON},
    "by_sector": {"Tech": TECH_JSON},
    "total": {"market": 15.0, "sector": 4.0, "residual": "inf", "trading": 2.0, "total": "nan"},
    "breaks": [{"ticker": "KKK", "ledger_shares": 5, "broker_shares": 7}],
    "no_beta": {"ZZZ": ["2026-09-29", "2026-09-30"]},
    "unregressed_days": ["2026-09-30"],
    "excluded": {"GAP": "no close on 2026-09-30"},
}


def _files(root: Path) -> list[Path]:
    return sorted((root / "attributions").iterdir())


def test_satisfies_the_port_and_writes_the_pinned_shape(tmp_path: Path) -> None:
    store: AttributionReportStore = JsonAttributionReportStore(tmp_path)
    store.save(REPORT)
    [path] = _files(tmp_path)
    text = path.read_text(encoding="utf-8")
    assert json.loads(text) == EXPECTED
    assert text == json.dumps(EXPECTED, indent=2)


def test_keys_are_the_report_fields_in_order(tmp_path: Path) -> None:
    JsonAttributionReportStore(tmp_path).save(REPORT)
    [path] = _files(tmp_path)
    keys = list(json.loads(path.read_text(encoding="utf-8")))
    assert keys == [field.name for field in dataclasses.fields(AttributionReport)]


@pytest.mark.parametrize(
    ("changes", "name"),
    [
        pytest.param({}, "2026-09-29_2026-09-30_220005.json", id="eastern-offset"),
        pytest.param(
            {"started_at": datetime(2026, 10, 1, 1, 2, 3, tzinfo=UTC)},
            "2026-09-29_2026-09-30_010203.json",
            id="utc",
        ),
        pytest.param({"start": None, "end": None}, "none_none_220005.json", id="no-window"),
    ],
)
def test_file_name_is_the_window_and_utc_start_time(
    tmp_path: Path, changes: dict[str, Any], name: str
) -> None:
    JsonAttributionReportStore(tmp_path).save(dataclasses.replace(REPORT, **changes))
    assert [path.name for path in _files(tmp_path)] == [name]


def test_an_existing_report_is_never_overwritten(tmp_path: Path) -> None:
    store = JsonAttributionReportStore(tmp_path)
    store.save(REPORT)
    [path] = _files(tmp_path)
    first = path.read_text(encoding="utf-8")
    with pytest.raises(FileExistsError, match="2026-09-29_2026-09-30_220005.json"):
        store.save(dataclasses.replace(REPORT, outcome=AttributeOutcome.CLEAN))
    assert path.read_text(encoding="utf-8") == first
    assert _files(tmp_path) == [path]


def test_a_failed_write_leaves_no_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write_text = Path.write_text

    def partial_then_fail(self: Path, data: str, *args: Any, **kwargs: Any) -> int:
        write_text(self, data[:10], *args, **kwargs)
        raise OSError("disk full")

    monkeypatch.setattr(Path, "write_text", partial_then_fail)
    with pytest.raises(OSError, match="disk full"):
        JsonAttributionReportStore(tmp_path).save(REPORT)
    assert _files(tmp_path) == []


def test_package_exports() -> None:
    import hedge_fund.adapters.attribution as attribution

    assert attribution.__all__ == ["JsonAttributionReportStore"]
    assert attribution.JsonAttributionReportStore is JsonAttributionReportStore
