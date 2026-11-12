"""Attribution reports as JSON files, one per attribute run."""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, date
from pathlib import Path
from typing import Any

from hedge_fund.adapters._json import decode_float, decode_instant, decode_optional, jsonable
from hedge_fund.adapters._load import load_dir
from hedge_fund.adapters.data._atomic import atomic_create
from hedge_fund.adapters.ledger.settlement_report import decode_break
from hedge_fund.domain.attribution import (
    AttributeOutcome,
    AttributionReport,
    AttributionTotals,
    DayAttribution,
)
from hedge_fund.domain.loaded import Loaded

ATTRIBUTIONS_DIR = "attributions"


class JsonAttributionReportStore:
    """AttributionReportStore writing ``<root>/attributions/<start>_<end>_<HHMMSS>.json``.

    ``HHMMSS`` is ``started_at`` converted to UTC; a missing bound (a run that failed before
    resolving it) is ``none``. The body is one object keyed by the ``AttributionReport``
    fields in field order (``adapters._json.jsonable``), ``indent=2``.
    """

    def __init__(self, root: Path) -> None:
        self._dir = Path(root) / ATTRIBUTIONS_DIR

    def save(self, report: AttributionReport) -> None:
        """Write atomically; FileExistsError, writing nothing, if the name is taken."""
        stamp = report.started_at.astimezone(UTC).strftime("%H%M%S")
        path = self._dir / f"{_bound(report.start)}_{_bound(report.end)}_{stamp}.json"
        body = json.dumps(jsonable(report), indent=2, allow_nan=False)
        try:
            atomic_create(path, lambda tmp: tmp.write_text(body, encoding="utf-8"))
        except FileExistsError as exc:
            raise FileExistsError(f"{path}: a report for this window and second exists") from exc

    def load_all(self) -> Loaded[AttributionReport]:
        """Decode every ``attributions/*.json``; a file that does not decode is in ``unreadable``."""
        return load_dir(self._dir, decode_attribution)


def _bound(day: date | None) -> str:
    return "none" if day is None else day.isoformat()


def decode_attribution(obj: Mapping[str, Any]) -> AttributionReport:
    """The AttributionReport whose ``jsonable`` form is ``obj``."""
    return AttributionReport(
        start=decode_optional(obj["start"], date.fromisoformat),
        end=decode_optional(obj["end"], date.fromisoformat),
        started_at=decode_instant(obj["started_at"]),
        outcome=AttributeOutcome(obj["outcome"]),
        failed_stage=obj["failed_stage"],
        error=obj["error"],
        days=tuple(_day(item) for item in obj["days"]),
        by_ticker={key: _totals(item) for key, item in obj["by_ticker"].items()},
        by_sector={key: _totals(item) for key, item in obj["by_sector"].items()},
        total=_totals(obj["total"]),
        breaks=tuple(decode_break(item) for item in obj["breaks"]),
        no_beta={
            ticker: tuple(date.fromisoformat(day) for day in days)
            for ticker, days in obj["no_beta"].items()
        },
        unregressed_days=tuple(date.fromisoformat(day) for day in obj["unregressed_days"]),
        excluded=dict(obj["excluded"]),
    )


def _day(item: Mapping[str, Any]) -> DayAttribution:
    return DayAttribution(
        day=date.fromisoformat(item["day"]),
        ticker=item["ticker"],
        sector=item["sector"],
        prior_shares=item["prior_shares"],
        market=decode_float(item["market"]),
        sector_pnl=decode_float(item["sector_pnl"]),
        residual=decode_float(item["residual"]),
        trading=decode_float(item["trading"]),
        regressed=item["regressed"],
    )


def _totals(item: Mapping[str, Any]) -> AttributionTotals:
    return AttributionTotals(
        market=decode_float(item["market"]),
        sector=decode_float(item["sector"]),
        residual=decode_float(item["residual"]),
        trading=decode_float(item["trading"]),
        total=decode_float(item["total"]),
    )
