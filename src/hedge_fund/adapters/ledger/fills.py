"""Fills as write-once JSON files: ``<root>/ledger/fills/<client_order_id>.json``."""

from __future__ import annotations

import dataclasses
import json
import math
import re
from datetime import date, datetime
from pathlib import Path
from typing import Any

from hedge_fund.adapters._json import jsonable
from hedge_fund.adapters.data._atomic import atomic_write
from hedge_fund.domain.ledger import Fill, FillStatus
from hedge_fund.domain.orders import OrderAction

FILLS_DIR = Path("ledger", "fills")

_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_KEYS = frozenset(field.name for field in dataclasses.fields(Fill))


class JsonFillStore:
    """FillStore keeping one ``jsonable`` file per client_order_id under ``root``.

    Ids must match ``^[A-Za-z0-9][A-Za-z0-9._-]*$`` (no path separators, no leading dot),
    in ``save`` and as file names in ``load_all``; ValueError otherwise.
    """

    def __init__(self, root: Path) -> None:
        self._dir = Path(root) / FILLS_DIR

    def save(self, fill: Fill) -> None:
        """Write atomically; FileExistsError, leaving the first file intact, if the id is taken."""
        path = self._dir / f"{_checked(fill.client_order_id)}.json"
        if path.exists():
            raise FileExistsError(f"{path}: fill {fill.client_order_id!r} is already recorded")
        body = json.dumps(jsonable(fill), indent=2, allow_nan=False)
        atomic_write(path, lambda tmp: tmp.write_text(body, encoding="utf-8"))

    def load_all(self) -> tuple[Fill, ...]:
        """Every ``*.json`` fill, by file name (temp files end in ``.tmp`` and never match).

        Raises ValueError naming the file for a bad file name, a malformed body, or a body
        whose ``client_order_id`` is not the file name.
        """
        return tuple(_load(path) for path in sorted(self._dir.glob("*.json")))


def _checked(client_order_id: str) -> str:
    if not _ID.fullmatch(client_order_id):
        raise ValueError(f"{client_order_id!r} is not a valid client_order_id")
    return client_order_id


def _load(path: Path) -> Fill:
    _checked(path.name.removesuffix(".json"))
    try:
        fill = _decode(json.loads(path.read_text(encoding="utf-8")))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{path.name}: malformed fill record: {exc}") from exc
    if f"{fill.client_order_id}.json" != path.name:
        raise ValueError(f"{path.name}: holds fill {fill.client_order_id!r}")
    return fill


def _decode(obj: Any) -> Fill:
    if not isinstance(obj, dict) or set(obj) != _KEYS:
        raise ValueError(f"expected an object with keys {', '.join(sorted(_KEYS))}")
    avg_price, filled_at = obj["avg_price"], obj["filled_at"]
    return Fill(
        client_order_id=_text(obj["client_order_id"]),
        broker_order_id=None if obj["broker_order_id"] is None else _text(obj["broker_order_id"]),
        cycle_as_of=date.fromisoformat(obj["cycle_as_of"]),
        ticker=_text(obj["ticker"]),
        action=OrderAction(obj["action"]),
        requested_shares=_count(obj["requested_shares"]),
        filled_shares=_count(obj["filled_shares"]),
        avg_price=None if avg_price is None else _number(avg_price),
        filled_at=None if filled_at is None else _instant(filled_at),
        status=FillStatus(obj["status"]),
        recorded_at=_instant(obj["recorded_at"]),
    )


def _text(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{value!r} is not a non-empty string")
    return value


def _count(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TypeError(f"{value!r} is not a share count")
    return value


def _number(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise TypeError(f"{value!r} is not a finite number")
    return float(value)


def _instant(value: Any) -> datetime:
    instant = datetime.fromisoformat(value)
    if instant.utcoffset() is None:
        raise ValueError(f"{value!r} has no UTC offset")
    return instant
