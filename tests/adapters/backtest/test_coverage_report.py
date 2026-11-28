"""The membership coverage report writer."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from hedge_fund.adapters.backtest.coverage_report import (
    decode_coverage_report,
    save_coverage_report,
)
from hedge_fund.engine.backtest.coverage import CoverageReport, SpanCoverage

RUN_ID = "2024-01-01_2024-02-09_0123456789ab"


def _report() -> CoverageReport:
    rows = (
        SpanCoverage("AAA", date(2020, 1, 2), None, 30, 0, 29, date(2024, 1, 2)),
        SpanCoverage("BBB", date(2020, 1, 2), date(2024, 1, 29), 20, 0, 0, None),
        SpanCoverage("CCC", date(2024, 1, 29), date(2024, 2, 5), 0, 3, 0, None),
    )
    return CoverageReport(
        start=date(2024, 1, 1),
        end=date(2024, 2, 9),
        spans=rows,
        member_days=50,
        priced_member_days=29,
        flagged_member_days=3,
        zero_spans=1,
        partial_spans=1,
        full_spans=0,
    )


def test_save_writes_the_documented_shape_under_coverage(tmp_path: Path) -> None:
    path = save_coverage_report(tmp_path, RUN_ID, _report())

    assert path == tmp_path / "coverage" / f"{RUN_ID}.json"
    body = json.loads(path.read_text(encoding="utf-8"))
    assert body["run_id"] == RUN_ID
    assert body["window"] == {"start": "2024-01-01", "end": "2024-02-09"}
    assert body["member_days"] == 50
    assert body["priced_member_days"] == 29
    assert body["flagged_member_days"] == 3
    assert body["coverage"] == pytest.approx(29 / 50)
    assert body["spans"] == {"zero": 1, "partial": 1, "full": 0}
    assert body["rows"][0] == {
        "ticker": "AAA",
        "start": "2020-01-02",
        "end": None,
        "window_days": 30,
        "flagged_days": 0,
        "priced_days": 29,
        "first_priced": "2024-01-02",
        "coverage": pytest.approx(29 / 30),
    }
    assert body["rows"][2]["coverage"] is None


def test_the_report_round_trips(tmp_path: Path) -> None:
    path = save_coverage_report(tmp_path, RUN_ID, _report())

    decoded = decode_coverage_report(json.loads(path.read_text(encoding="utf-8")))

    assert decoded == _report()


def test_an_empty_report_has_no_coverage_fraction(tmp_path: Path) -> None:
    empty = CoverageReport(date(2024, 1, 1), date(2024, 1, 2), (), 0, 0, 0, 0, 0, 0)
    path = save_coverage_report(tmp_path, RUN_ID, empty)

    body = json.loads(path.read_text(encoding="utf-8"))

    assert body["coverage"] is None
    assert decode_coverage_report(body) == empty


def test_a_second_save_of_the_same_run_raises(tmp_path: Path) -> None:
    save_coverage_report(tmp_path, RUN_ID, _report())

    with pytest.raises(FileExistsError, match=RUN_ID):
        save_coverage_report(tmp_path, RUN_ID, _report())
