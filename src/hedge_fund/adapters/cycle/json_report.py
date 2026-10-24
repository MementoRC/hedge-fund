"""Cycle reports as JSON files, one per run: an audit trail, not the Layer 7 ledger."""

from __future__ import annotations

import dataclasses
import json
import math
from collections.abc import Mapping
from datetime import UTC, date
from enum import Enum
from pathlib import Path
from typing import Any

from hedge_fund.adapters.data._atomic import atomic_write
from hedge_fund.domain.cycle import CycleReport

RUNS_DIR = "runs"


class JsonCycleReportStore:
    """CycleReportStore writing ``<root>/runs/<as_of>_<HHMMSS>.json``.

    ``HHMMSS`` is ``started_at`` converted to UTC. The body is one object keyed by the
    ``CycleReport`` fields in field order (see ``_jsonable``), written with ``indent=2``.
    """

    def __init__(self, root: Path) -> None:
        self._dir = Path(root) / RUNS_DIR

    def save(self, report: CycleReport) -> None:
        """Write atomically; FileExistsError, writing nothing, if the name is taken."""
        stamp = report.started_at.astimezone(UTC).strftime("%H%M%S")
        path = self._dir / f"{report.as_of.isoformat()}_{stamp}.json"
        if path.exists():
            raise FileExistsError(f"{path}: a report for a run started in this second exists")
        body = json.dumps(_jsonable(report), indent=2, allow_nan=False)
        atomic_write(path, lambda tmp: tmp.write_text(body, encoding="utf-8"))


def _jsonable(value: Any) -> Any:
    """Dataclasses as objects in field order, enums as values, dates and datetimes as ISO
    strings, mappings as objects, tuples and lists as lists, non-finite floats as
    ``"nan"`` / ``"inf"`` / ``"-inf"``.
    """
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _jsonable(getattr(value, field.name)) for field in dataclasses.fields(value)
        }
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    return value
