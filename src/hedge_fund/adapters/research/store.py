"""Content-addressed JSON files for ResearchResults, so a research run can be replayed."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Mapping
from datetime import UTC, date, datetime
from enum import Enum
from pathlib import Path
from typing import Any, cast

from hedge_fund.adapters.data._atomic import atomic_write
from hedge_fund.domain.research import NameResearch, Rating, ResearchResult, RunRecord, Verdict
from hedge_fund.domain.shortlist import Candidate, Side

RESEARCH_DIR = "research"


def save_research(result: ResearchResult, root: Path) -> Path:
    """Write research/<as_of>_<sha12>.json under root and return its path.

    The name is derived from the content (``created_at`` excluded), so an existing file
    is returned as-is and never overwritten. Raises ValueError on non-finite floats.
    """
    payload = _payload(result)
    path = Path(root) / RESEARCH_DIR / f"{result.as_of.isoformat()}_{_digest(payload)}.json"
    if path.is_file():
        return path
    body = json.dumps(
        {"created_at": datetime.now(UTC).isoformat(), "result": payload},
        indent=2,
        allow_nan=False,
    )
    atomic_write(path, lambda tmp: tmp.write_text(body, encoding="utf-8"))
    return path


def load_research(path: Path) -> ResearchResult:
    """Read a saved result, verifying as_of and content digest against the file name."""
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))["result"]
    name_as_of, expected = path.stem.rsplit("_", 1)
    if name_as_of != payload["as_of"]:
        raise ValueError(
            f"{path.name}: as_of mismatch, name has {name_as_of}, file has {payload['as_of']}"
        )
    actual = _digest(payload)
    if actual != expected:
        raise ValueError(f"{path.name}: digest mismatch, name has {expected}, content is {actual}")
    return ResearchResult(
        as_of=date.fromisoformat(payload["as_of"]),
        names=tuple(_name(item) for item in payload["names"]),
    )


def _payload(result: ResearchResult) -> dict[str, Any]:
    return cast(dict[str, Any], _jsonable(dataclasses.asdict(result)))


def _jsonable(value: Any) -> Any:
    """Enums as values, dates as ISO strings, mappings as dicts, tuples as lists."""
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    return value


def _digest(payload: Mapping[str, Any]) -> str:
    """First 12 hex chars of SHA-256 over the canonical JSON of ``payload``."""
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


def _name(item: Mapping[str, Any]) -> NameResearch:
    candidate = item["candidate"]
    return NameResearch(
        candidate=Candidate(
            ticker=candidate["ticker"],
            sector=candidate["sector"],
            side=Side(candidate["side"]),
            depth=candidate["depth"],
            composite=candidate["composite"],
        ),
        runs=tuple(_run(run) for run in item["runs"]),
        valid=item["valid"],
        mean=item["mean"],
        std=item["std"],
        agreement=item["agreement"],
        verdict=Verdict(item["verdict"]),
    )


def _run(item: Mapping[str, Any]) -> RunRecord:
    rating = item["rating"]
    return RunRecord(
        ticker=item["ticker"],
        as_of=date.fromisoformat(item["as_of"]),
        run_index=item["run_index"],
        rating=None if rating is None else Rating(rating),
        decision=item["decision"],
        reports=dict(item["reports"]),
        seconds=item["seconds"],
        error=item["error"],
    )
