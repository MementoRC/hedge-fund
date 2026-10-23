"""Account snapshots as content-addressed JSON files, one per broker read.

Checks structure and types only on load; ``engine.account.validate_snapshot`` checks
values (the service before saving, ``nav_history`` after loading).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from hedge_fund.adapters.data._atomic import atomic_write
from hedge_fund.domain.account import AccountSnapshot
from hedge_fund.domain.orders import Holding
from hedge_fund.ports.account import SnapshotStoreError

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

    def snapshots(self) -> tuple[AccountSnapshot, ...]:
        """Every stored snapshot in file-name order; ``()`` when the directory is missing.

        Raises SnapshotStoreError naming the file when it cannot be read or parsed, its
        digest or as_of does not match its name, or a field has the wrong type
        (``observed_at`` must carry an offset).
        """
        if not self._dir.is_dir():
            return ()
        return tuple(_load(path) for path in sorted(self._dir.glob("*.json")))


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


def _load(path: Path) -> AccountSnapshot:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))["snapshot"]
        digest = _digest(payload)
        snapshot = _snapshot(payload)
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise SnapshotStoreError(f"{path.name}: malformed snapshot file: {exc}") from exc
    name_as_of, _, name_digest = path.stem.partition("_")
    if name_digest != digest:
        raise SnapshotStoreError(
            f"{path.name}: digest mismatch, name has {name_digest!r}, content is {digest!r}"
        )
    if name_as_of != payload["as_of"]:
        raise SnapshotStoreError(
            f"{path.name}: as_of mismatch, name has {name_as_of!r}, file has {payload['as_of']!r}"
        )
    return snapshot


def _snapshot(payload: Mapping[str, Any]) -> AccountSnapshot:
    """Typed fields, or KeyError / TypeError / ValueError; values are not range-checked."""
    equity = payload["equity"]
    if isinstance(equity, bool) or not isinstance(equity, int | float):
        raise TypeError(f"equity must be a JSON number, got {equity!r}")
    observed_at = datetime.fromisoformat(payload["observed_at"])
    if observed_at.tzinfo is None:
        raise ValueError(f"observed_at must carry a UTC offset, got {payload['observed_at']!r}")
    holdings = payload["holdings"]
    if not isinstance(holdings, list):
        raise TypeError(f"holdings must be a list, got {holdings!r}")
    return AccountSnapshot(
        as_of=date.fromisoformat(payload["as_of"]),
        observed_at=observed_at,
        equity=float(equity),
        holdings=tuple(_holding(item) for item in holdings),
    )


def _holding(item: Any) -> Holding:
    """``[ticker, shares]`` with a str ticker and a non-bool int ``shares``."""
    if (
        not isinstance(item, list)
        or len(item) != 2
        or not isinstance(item[0], str)
        or isinstance(item[1], bool)
        or not isinstance(item[1], int)
    ):
        raise TypeError(f"holding must be [ticker, shares], got {item!r}")
    return Holding(ticker=item[0], shares=item[1])
