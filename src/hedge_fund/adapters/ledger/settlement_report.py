"""Settlement reports as JSON files, one per settle run."""

from __future__ import annotations

import json
from datetime import UTC
from pathlib import Path

from hedge_fund.adapters._json import jsonable
from hedge_fund.adapters.data._atomic import atomic_write
from hedge_fund.domain.ledger import SettlementReport

SETTLEMENTS_DIR = "settlements"


class JsonSettlementReportStore:
    """SettlementReportStore writing ``<root>/settlements/<as_of>_<HHMMSS>.json``.

    ``HHMMSS`` is ``started_at`` converted to UTC. The body is one object keyed by the
    ``SettlementReport`` fields in field order (``adapters._json.jsonable``), ``indent=2``.
    """

    def __init__(self, root: Path) -> None:
        self._dir = Path(root) / SETTLEMENTS_DIR

    def save(self, report: SettlementReport) -> None:
        """Write atomically; FileExistsError, writing nothing, if the name is taken."""
        stamp = report.started_at.astimezone(UTC).strftime("%H%M%S")
        path = self._dir / f"{report.as_of.isoformat()}_{stamp}.json"
        if path.exists():
            raise FileExistsError(f"{path}: a report for a settle started in this second exists")
        body = json.dumps(jsonable(report), indent=2, allow_nan=False)
        atomic_write(path, lambda tmp: tmp.write_text(body, encoding="utf-8"))
