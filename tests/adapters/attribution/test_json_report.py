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
    NO_PNL,
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


def test_load_all_round_trips_a_saved_report(tmp_path: Path) -> None:
    store = JsonAttributionReportStore(tmp_path)
    store.save(REPORT)

    loaded = store.load_all()

    # NaN never compares equal (dataclass eq is field-wise on 3.13), so pin it explicitly.
    [got] = loaded.items
    assert math.isnan(got.total.total)
    assert got.started_at.utcoffset() == REPORT.started_at.utcoffset() == timedelta(hours=-4)
    no_nan = AttributionTotals(market=15.0, sector=4.0, residual=math.inf, trading=2.0, total=0.0)
    assert dataclasses.replace(got, total=no_nan) == dataclasses.replace(REPORT, total=no_nan)
    assert loaded.unreadable == {}


def test_load_all_round_trips_a_failed_run_with_no_window(tmp_path: Path) -> None:
    failed = dataclasses.replace(
        REPORT,
        start=None,
        end=None,
        outcome=AttributeOutcome.FAILED,
        failed_stage="load",
        error="ValueError: x",
        days=(),
        by_ticker={},
        by_sector={},
        total=NO_PNL,
        breaks=(),
        no_beta={},
        unregressed_days=(),
        excluded={},
    )
    store = JsonAttributionReportStore(tmp_path)
    store.save(failed)

    assert store.load_all().items == (failed,)


@pytest.mark.parametrize(
    ("field", "bad"),
    [
        pytest.param("prior_shares", True, id="prior_shares-bool"),
        pytest.param("prior_shares", 1.5, id="prior_shares-float"),
        pytest.param("regressed", 1, id="regressed-int"),
        pytest.param("regressed", "true", id="regressed-str"),
    ],
)
def test_a_day_with_a_mistyped_field_is_unreadable_and_the_good_report_still_loads(
    tmp_path: Path, field: str, bad: object
) -> None:
    store = JsonAttributionReportStore(tmp_path)
    store.save(REPORT)
    [good] = _files(tmp_path)
    body = json.loads(good.read_text(encoding="utf-8"))
    body["days"][0][field] = bad
    (tmp_path / "attributions" / "bad.json").write_text(json.dumps(body), encoding="utf-8")

    loaded = store.load_all()

    assert len(loaded.items) == 1
    assert list(loaded.unreadable) == ["bad.json"]


def test_a_truncated_file_is_unreadable_and_the_good_report_still_loads(tmp_path: Path) -> None:
    store = JsonAttributionReportStore(tmp_path)
    store.save(REPORT)
    [good] = _files(tmp_path)
    text = good.read_text(encoding="utf-8")
    (tmp_path / "attributions" / "cut.json").write_text(text[: len(text) // 2], encoding="utf-8")

    loaded = store.load_all()

    assert loaded.items[0].outcome is AttributeOutcome.BREAK
    assert len(loaded.items) == 1
    assert list(loaded.unreadable) == ["cut.json"]


@pytest.mark.parametrize("key", ["started_at", "total", "excluded"])
def test_a_file_missing_a_key_is_unreadable_and_the_good_report_still_loads(
    tmp_path: Path, key: str
) -> None:
    store = JsonAttributionReportStore(tmp_path)
    store.save(REPORT)
    [good] = _files(tmp_path)
    body = json.loads(good.read_text(encoding="utf-8"))
    del body[key]
    (tmp_path / "attributions" / "gap.json").write_text(json.dumps(body), encoding="utf-8")

    loaded = store.load_all()

    assert len(loaded.items) == 1
    assert list(loaded.unreadable) == ["gap.json"]
    assert key in loaded.unreadable["gap.json"]


def test_load_all_of_a_root_without_reports_is_empty(tmp_path: Path) -> None:
    assert JsonAttributionReportStore(tmp_path).load_all().items == ()
