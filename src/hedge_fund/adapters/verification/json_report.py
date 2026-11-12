"""Verification reports as JSON files, one per verify run."""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, date
from pathlib import Path
from typing import Any

from hedge_fund.adapters._json import decode_float, decode_instant, decode_optional, jsonable
from hedge_fund.adapters._load import load_dir
from hedge_fund.adapters.data._atomic import atomic_create
from hedge_fund.domain.loaded import Loaded
from hedge_fund.domain.research import Verdict
from hedge_fund.domain.shortlist import Side
from hedge_fund.domain.verification import (
    Bucket,
    ExclusionReason,
    HorizonSummary,
    Observation,
    ObservationStatus,
    VerificationReport,
    VerifyOutcome,
)

VERIFICATIONS_DIR = "verifications"


class JsonVerificationReportStore:
    """VerificationReportStore writing ``<root>/verifications/<start>_<end>_<HHMMSS>.json``.

    ``HHMMSS`` is ``started_at`` converted to UTC; a missing bound (a run that failed before
    resolving it, or has no research and no ``--start``) is ``none``. The body is one object
    keyed by the ``VerificationReport`` fields in field order (``adapters._json.jsonable``),
    ``indent=2``.
    """

    def __init__(self, root: Path) -> None:
        self._dir = Path(root) / VERIFICATIONS_DIR

    def save(self, report: VerificationReport) -> None:
        """Write atomically; FileExistsError, writing nothing, if the name is taken."""
        stamp = report.started_at.astimezone(UTC).strftime("%H%M%S")
        path = self._dir / f"{_bound(report.start)}_{_bound(report.end)}_{stamp}.json"
        body = json.dumps(jsonable(report), indent=2, allow_nan=False)
        try:
            atomic_create(path, lambda tmp: tmp.write_text(body, encoding="utf-8"))
        except FileExistsError as exc:
            raise FileExistsError(f"{path}: a report for this window and second exists") from exc

    def load_all(self) -> Loaded[VerificationReport]:
        """Decode every ``verifications/*.json``; a file that does not decode is in ``unreadable``."""
        return load_dir(self._dir, decode_verification)


def _bound(day: date | None) -> str:
    return "none" if day is None else day.isoformat()


def decode_verification(obj: Mapping[str, Any]) -> VerificationReport:
    """The VerificationReport whose ``jsonable`` form is ``obj``."""
    return VerificationReport(
        start=decode_optional(obj["start"], date.fromisoformat),
        end=decode_optional(obj["end"], date.fromisoformat),
        started_at=decode_instant(obj["started_at"]),
        outcome=VerifyOutcome(obj["outcome"]),
        failed_stage=obj["failed_stage"],
        error=obj["error"],
        observations=tuple(_observation(item) for item in obj["observations"]),
        summaries=tuple(_summary(item) for item in obj["summaries"]),
    )


def _optional_float(value: Any) -> float | None:
    return decode_optional(value, decode_float)


def _optional_day(value: Any) -> date | None:
    return decode_optional(value, date.fromisoformat)


def _observation(item: Mapping[str, Any]) -> Observation:
    return Observation(
        as_of=date.fromisoformat(item["as_of"]),
        ticker=item["ticker"],
        sector=item["sector"],
        side=Side(item["side"]),
        verdict=Verdict(item["verdict"]),
        mean=_optional_float(item["mean"]),
        std=_optional_float(item["std"]),
        agreement=_optional_float(item["agreement"]),
        valid=item["valid"],
        horizon=item["horizon"],
        base_date=_optional_day(item["base_date"]),
        end_date=_optional_day(item["end_date"]),
        name_return=_optional_float(item["name_return"]),
        spy_return=_optional_float(item["spy_return"]),
        aligned_excess=_optional_float(item["aligned_excess"]),
        hit=item["hit"],
        executed=item["executed"],
        status=ObservationStatus(item["status"]),
        reason=decode_optional(item["reason"], ExclusionReason),
    )


def _bucket(item: Mapping[str, Any]) -> Bucket:
    return Bucket(
        key=item["key"],
        n=item["n"],
        hits=item["hits"],
        hit_rate=_optional_float(item["hit_rate"]),
        mean_excess=_optional_float(item["mean_excess"]),
        median_excess=_optional_float(item["median_excess"]),
    )


def _buckets(obj: Mapping[str, Any]) -> dict[str, Bucket]:
    return {key: _bucket(item) for key, item in obj.items()}


def _summary(item: Mapping[str, Any]) -> HorizonSummary:
    return HorizonSummary(
        horizon=item["horizon"],
        scored=item["scored"],
        pending=item["pending"],
        excluded=item["excluded"],
        overall=_bucket(item["overall"]),
        by_verdict=_buckets(item["by_verdict"]),
        by_score=_buckets(item["by_score"]),
        by_sector=_buckets(item["by_sector"]),
        executed=_bucket(item["executed"]),
        ic=_optional_float(item["ic"]),
        no_call_mean_abs_excess=_optional_float(item["no_call_mean_abs_excess"]),
    )
