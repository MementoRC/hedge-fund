"""Account snapshots as content-addressed JSON files, one per broker read.

Checks structure and types only on load; ``engine.account.validate_snapshot`` checks
values (the service before saving, ``nav_history`` after loading).
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from hedge_fund.adapters.data._atomic import atomic_write
from hedge_fund.domain.account import AccountSnapshot

ACCOUNT_DIR = "account"


class JsonSnapshotStore:
    """SnapshotStore writing ``<root>/account/<as_of>_<sha12>.json``.

    ``sha12`` is the first 12 hex chars of the sha256 of the snapshot payload's canonical
    JSON. ``observed_at`` is in the payload, so every read gets its own file; a name that
    already exists holds the same snapshot and is never overwritten.
    """

    def __init__(self, root: Path) -> None:
        self._dir = Path(root) / ACCOUNT_DIR

    def save(self, snapshot: AccountSnapshot) -> None:
        """Write atomically unless the file exists; ValueError on a non-finite equity."""
        payload = _payload(snapshot)
        path = self._dir / f"{snapshot.as_of.isoformat()}_{_digest(payload)}.json"
        if path.is_file():
            return
        body = json.dumps(
            {"created_at": datetime.now(UTC).isoformat(), "snapshot": payload},
            indent=2,
            allow_nan=False,
        )
        atomic_write(path, lambda tmp: tmp.write_text(body, encoding="utf-8"))


def _payload(snapshot: AccountSnapshot) -> dict[str, Any]:
    return {
        "as_of": snapshot.as_of.isoformat(),
        "observed_at": snapshot.observed_at.isoformat(),
        "equity": snapshot.equity,
        "holdings": [[holding.ticker, holding.shares] for holding in snapshot.holdings],
    }


def _digest(payload: Any) -> str:
    """First 12 hex chars of SHA-256 over the canonical JSON of ``payload``."""
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]
