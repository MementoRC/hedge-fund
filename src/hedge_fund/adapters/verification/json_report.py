"""Verification reports as JSON files, one per verify run."""

from __future__ import annotations

import json
from datetime import UTC, date
from pathlib import Path

from hedge_fund.adapters._json import jsonable
from hedge_fund.adapters.data._atomic import atomic_write
from hedge_fund.domain.verification import VerificationReport

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
        if path.exists():
            raise FileExistsError(f"{path}: a report for this window and second exists")
        body = json.dumps(jsonable(report), indent=2, allow_nan=False)
        atomic_write(path, lambda tmp: tmp.write_text(body, encoding="utf-8"))


def _bound(day: date | None) -> str:
    return "none" if day is None else day.isoformat()
