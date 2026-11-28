"""Write the membership coverage of a run to ``<root>/coverage/<run-id>.json`` (write-once).

It lives beside, not inside, ``<root>/backtest/``: that directory is decoded strictly as
BacktestReports. The JSON object holds ``run_id``; ``window`` (``start``, ``end``, inclusive);
``member_days`` (member-days with no flagged span over them), ``priced_member_days`` (those
with a stored close) and ``flagged_member_days``; ``coverage`` (priced / member days, null
when there are none); ``spans`` (counts of spans with ``zero``, ``partial`` and ``full``
priced days); and ``rows``, one per membership span: ``ticker``, ``start``, ``end`` (null
when open), ``window_days``, ``flagged_days``, ``priced_days``, ``first_priced`` (null when
none) and ``coverage`` (null when every day is flagged).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import date
from pathlib import Path
from typing import Any

from hedge_fund.adapters._json import decode_int, decode_optional
from hedge_fund.adapters.data._atomic import atomic_create
from hedge_fund.engine.backtest.coverage import CoverageReport, SpanCoverage

COVERAGE_DIR = "coverage"


def save_coverage_report(root: Path, run_id: str, report: CoverageReport) -> Path:
    """Write ``report`` atomically; raises FileExistsError if this run id already exists."""
    path = root / COVERAGE_DIR / f"{run_id}.json"
    body = json.dumps(_encode(run_id, report), indent=2, allow_nan=False)
    try:
        atomic_create(path, lambda tmp: tmp.write_text(body, encoding="utf-8"))
    except FileExistsError as exc:
        raise FileExistsError(f"{path}: a coverage report for this run id exists") from exc
    return path


def decode_coverage_report(obj: Mapping[str, Any]) -> CoverageReport:
    """The CoverageReport whose saved form is ``obj``."""
    window, counts = obj["window"], obj["spans"]
    return CoverageReport(
        start=date.fromisoformat(window["start"]),
        end=date.fromisoformat(window["end"]),
        spans=tuple(_decode_row(row) for row in obj["rows"]),
        member_days=decode_int(obj["member_days"]),
        priced_member_days=decode_int(obj["priced_member_days"]),
        flagged_member_days=decode_int(obj["flagged_member_days"]),
        zero_spans=decode_int(counts["zero"]),
        partial_spans=decode_int(counts["partial"]),
        full_spans=decode_int(counts["full"]),
    )


def _encode(run_id: str, report: CoverageReport) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "window": {"start": report.start.isoformat(), "end": report.end.isoformat()},
        "member_days": report.member_days,
        "priced_member_days": report.priced_member_days,
        "flagged_member_days": report.flagged_member_days,
        "coverage": report.coverage,
        "spans": {
            "zero": report.zero_spans,
            "partial": report.partial_spans,
            "full": report.full_spans,
        },
        "rows": [_encode_row(row) for row in report.spans],
    }


def _encode_row(row: SpanCoverage) -> dict[str, Any]:
    return {
        "ticker": row.ticker,
        "start": row.start.isoformat(),
        "end": None if row.end is None else row.end.isoformat(),
        "window_days": row.window_days,
        "flagged_days": row.flagged_days,
        "priced_days": row.priced_days,
        "first_priced": None if row.first_priced is None else row.first_priced.isoformat(),
        "coverage": row.coverage,
    }


def _decode_row(item: Mapping[str, Any]) -> SpanCoverage:
    return SpanCoverage(
        ticker=item["ticker"],
        start=date.fromisoformat(item["start"]),
        end=decode_optional(item["end"], date.fromisoformat),
        window_days=decode_int(item["window_days"]),
        flagged_days=decode_int(item["flagged_days"]),
        priced_days=decode_int(item["priced_days"]),
        first_priced=decode_optional(item["first_priced"], date.fromisoformat),
    )
