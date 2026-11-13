"""Tests for the JSON verification report store."""

from __future__ import annotations

import dataclasses
import json
import math
from datetime import UTC, date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from hedge_fund.adapters._json import jsonable
from hedge_fund.adapters.verification.json_report import JsonVerificationReportStore
from hedge_fund.domain.verification import (
    Bucket,
    ExclusionReason,
    HorizonSummary,
    ObservationStatus,
    VerificationReport,
    VerifyOutcome,
)
from hedge_fund.ports.verification import VerificationReportStore
from tests.factories import make_observation

EASTERN = timezone(timedelta(hours=-4))
BUCKET = Bucket(key="all", n=1, hits=1, hit_rate=1.0, mean_excess=0.05, median_excess=0.05)
SUMMARY = HorizonSummary(
    horizon=5,
    scored=1,
    pending=0,
    excluded=0,
    overall=BUCKET,
    by_verdict={"confirm": BUCKET},
    by_score={"1": BUCKET},
    by_sector={"Tech": BUCKET},
    executed=Bucket("executed", 0, 0, None, None, None),
    ic=math.nan,  # pins the non-finite string
    no_call_mean_abs_excess=None,
)

# Every field populated.
REPORT = VerificationReport(
    start=date(2026, 9, 1),
    end=date(2026, 9, 30),
    started_at=datetime(2026, 9, 30, 18, 0, 5, tzinfo=EASTERN),  # 22:00:05 UTC
    outcome=VerifyOutcome.GAPS,
    failed_stage=None,
    error=None,
    observations=(make_observation(),),
    summaries=(SUMMARY,),
)


def _files(root: Path) -> list[Path]:
    return sorted((root / "verifications").iterdir())


def test_satisfies_the_port_and_writes_the_report_as_jsonable_json(tmp_path: Path) -> None:
    store: VerificationReportStore = JsonVerificationReportStore(tmp_path)
    store.save(REPORT)
    [path] = _files(tmp_path)
    text = path.read_text(encoding="utf-8")
    assert json.loads(text) == jsonable(REPORT)
    assert text == json.dumps(jsonable(REPORT), indent=2)
    body = json.loads(text)
    assert body["started_at"] == "2026-09-30T18:00:05-04:00"
    assert body["outcome"] == "gaps"
    assert body["observations"][0]["status"] == "scored"
    assert body["observations"][0]["side"] == "long"
    assert body["summaries"][0]["ic"] == "nan"
    assert body["summaries"][0]["by_score"]["1"]["hit_rate"] == 1.0


def test_keys_are_the_report_fields_in_order(tmp_path: Path) -> None:
    JsonVerificationReportStore(tmp_path).save(REPORT)
    [path] = _files(tmp_path)
    keys = list(json.loads(path.read_text(encoding="utf-8")))
    assert keys == [field.name for field in dataclasses.fields(VerificationReport)]


@pytest.mark.parametrize(
    ("changes", "name"),
    [
        pytest.param({}, "2026-09-01_2026-09-30_220005.json", id="eastern-offset"),
        pytest.param(
            {"started_at": datetime(2026, 10, 1, 1, 2, 3, tzinfo=UTC)},
            "2026-09-01_2026-09-30_010203.json",
            id="utc",
        ),
        pytest.param({"start": None, "end": None}, "none_none_220005.json", id="no-window"),
    ],
)
def test_file_name_is_the_window_and_utc_start_time(
    tmp_path: Path, changes: dict[str, Any], name: str
) -> None:
    JsonVerificationReportStore(tmp_path).save(dataclasses.replace(REPORT, **changes))
    assert [path.name for path in _files(tmp_path)] == [name]


def test_an_existing_report_is_never_overwritten(tmp_path: Path) -> None:
    store = JsonVerificationReportStore(tmp_path)
    store.save(REPORT)
    [path] = _files(tmp_path)
    first = path.read_text(encoding="utf-8")
    with pytest.raises(FileExistsError, match="2026-09-01_2026-09-30_220005.json"):
        store.save(dataclasses.replace(REPORT, outcome=VerifyOutcome.CLEAN))
    assert path.read_text(encoding="utf-8") == first
    assert _files(tmp_path) == [path]


def test_a_failed_write_leaves_no_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write_text = Path.write_text

    def partial_then_fail(self: Path, data: str, *args: Any, **kwargs: Any) -> int:
        write_text(self, data[:10], *args, **kwargs)
        raise OSError("disk full")

    monkeypatch.setattr(Path, "write_text", partial_then_fail)
    with pytest.raises(OSError, match="disk full"):
        JsonVerificationReportStore(tmp_path).save(REPORT)
    assert _files(tmp_path) == []


def _without_nan_ic(report: VerificationReport) -> VerificationReport:
    summaries = tuple(dataclasses.replace(s, ic=None) for s in report.summaries)
    return dataclasses.replace(report, summaries=summaries)


def test_load_all_round_trips_a_saved_report(tmp_path: Path) -> None:
    store = JsonVerificationReportStore(tmp_path)
    store.save(REPORT)

    loaded = store.load_all()

    [item] = loaded.items
    assert math.isnan(item.summaries[0].ic)  # type: ignore[arg-type]
    assert item.started_at.utcoffset() == REPORT.started_at.utcoffset() == timedelta(hours=-4)
    assert _without_nan_ic(item) == _without_nan_ic(REPORT)
    assert loaded.unreadable == {}


def test_load_all_round_trips_an_excluded_observation_and_no_window(tmp_path: Path) -> None:
    excluded = make_observation(
        mean=None,
        std=None,
        agreement=None,
        base_date=None,
        end_date=None,
        name_return=None,
        spy_return=None,
        aligned_excess=None,
        hit=None,
        status=ObservationStatus.EXCLUDED,
        reason=ExclusionReason.INSUFFICIENT_RESEARCH,
    )
    bare = dataclasses.replace(
        REPORT,
        start=None,
        end=None,
        outcome=VerifyOutcome.FAILED,
        failed_stage="load",
        error="ValueError: x",
        observations=(excluded,),
        summaries=(),
    )
    store = JsonVerificationReportStore(tmp_path)
    store.save(bare)

    assert store.load_all().items == (bare,)


INT_FIELDS = [
    ("observations", 0, "valid"),
    ("observations", 0, "horizon"),
    ("summaries", 0, "horizon"),
    ("summaries", 0, "scored"),
    ("summaries", 0, "pending"),
    ("summaries", 0, "excluded"),
    ("summaries", 0, "overall", "n"),
    ("summaries", 0, "overall", "hits"),
]
BOOL_FIELDS = [("observations", 0, "hit"), ("observations", 0, "executed")]


def _with_tampered_copy(tmp_path: Path, path: tuple[str | int, ...], bad: object) -> Any:
    store = JsonVerificationReportStore(tmp_path)
    store.save(REPORT)
    [good] = _files(tmp_path)
    body = json.loads(good.read_text(encoding="utf-8"))
    node = body
    for step in path[:-1]:
        node = node[step]
    node[path[-1]] = bad
    (tmp_path / "verifications" / "bad.json").write_text(json.dumps(body), encoding="utf-8")
    return store.load_all()


@pytest.mark.parametrize("bad", [True, 1.5], ids=["bool", "float"])
@pytest.mark.parametrize("path", INT_FIELDS, ids=lambda p: ".".join(map(str, p)))
def test_an_int_field_that_is_not_an_int_makes_the_file_unreadable(
    tmp_path: Path, path: tuple[str | int, ...], bad: object
) -> None:
    loaded = _with_tampered_copy(tmp_path, path, bad)

    assert len(loaded.items) == 1
    assert list(loaded.unreadable) == ["bad.json"]


@pytest.mark.parametrize("bad", [1, "yes"], ids=["int", "str"])
@pytest.mark.parametrize("path", BOOL_FIELDS, ids=lambda p: ".".join(map(str, p)))
def test_a_bool_field_that_is_not_a_bool_makes_the_file_unreadable(
    tmp_path: Path, path: tuple[str | int, ...], bad: object
) -> None:
    loaded = _with_tampered_copy(tmp_path, path, bad)

    assert len(loaded.items) == 1
    assert list(loaded.unreadable) == ["bad.json"]


def test_a_truncated_file_is_unreadable_and_the_good_report_still_loads(tmp_path: Path) -> None:
    store = JsonVerificationReportStore(tmp_path)
    store.save(REPORT)
    [good] = _files(tmp_path)
    text = good.read_text(encoding="utf-8")
    (tmp_path / "verifications" / "cut.json").write_text(text[: len(text) // 2], encoding="utf-8")

    loaded = store.load_all()

    assert len(loaded.items) == 1
    assert loaded.items[0].outcome is VerifyOutcome.GAPS
    assert list(loaded.unreadable) == ["cut.json"]


@pytest.mark.parametrize("key", ["started_at", "outcome", "summaries"])
def test_a_file_missing_a_key_is_unreadable_and_the_good_report_still_loads(
    tmp_path: Path, key: str
) -> None:
    store = JsonVerificationReportStore(tmp_path)
    store.save(REPORT)
    [good] = _files(tmp_path)
    body = json.loads(good.read_text(encoding="utf-8"))
    del body[key]
    (tmp_path / "verifications" / "gap.json").write_text(json.dumps(body), encoding="utf-8")

    loaded = store.load_all()

    assert len(loaded.items) == 1
    assert list(loaded.unreadable) == ["gap.json"]
    assert key in loaded.unreadable["gap.json"]


def test_load_all_of_a_root_without_reports_is_empty(tmp_path: Path) -> None:
    assert JsonVerificationReportStore(tmp_path).load_all().items == ()


def test_package_exports() -> None:
    import hedge_fund.adapters.verification as verification

    assert verification.__all__ == ["JsonVerificationReportStore"]
    assert verification.JsonVerificationReportStore is JsonVerificationReportStore
