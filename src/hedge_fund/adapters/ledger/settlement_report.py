"""Settlement reports as JSON files, one per settle run."""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, date
from pathlib import Path
from typing import Any

from hedge_fund.adapters._json import (
    decode_float,
    decode_instant,
    decode_int,
    jsonable,
)
from hedge_fund.adapters._load import load_dir
from hedge_fund.adapters.data._atomic import atomic_create
from hedge_fund.adapters.ledger.fills import decode_fill
from hedge_fund.domain.ledger import (
    ClosedLot,
    Lot,
    MarkedLot,
    PendingOrder,
    ReconciliationBreak,
    SettlementReport,
    SettleOutcome,
    UnsettledOrder,
)
from hedge_fund.domain.loaded import Loaded
from hedge_fund.domain.orders import Holding

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
        body = json.dumps(jsonable(report), indent=2, allow_nan=False)
        try:
            atomic_create(path, lambda tmp: tmp.write_text(body, encoding="utf-8"))
        except FileExistsError as exc:
            raise FileExistsError(
                f"{path}: a report for a settle started in this second exists"
            ) from exc

    def load_all(self) -> Loaded[SettlementReport]:
        """Decode every ``settlements/*.json``; a file that does not decode is in ``unreadable``."""
        return load_dir(self._dir, decode_settlement)


def decode_settlement(obj: Mapping[str, Any]) -> SettlementReport:
    """The SettlementReport whose ``jsonable`` form is ``obj``; KeyError, TypeError or
    ValueError if it is malformed.
    """
    positions = obj["broker_positions"]
    unreadable = obj["unreadable_reports"]
    if not isinstance(unreadable, Mapping):
        raise TypeError(f"unreadable_reports must be an object, got {type(unreadable).__name__}")
    return SettlementReport(
        as_of=date.fromisoformat(obj["as_of"]),
        started_at=decode_instant(obj["started_at"]),
        outcome=SettleOutcome(obj["outcome"]),
        failed_stage=obj["failed_stage"],
        error=obj["error"],
        new_fills=tuple(decode_fill(item) for item in obj["new_fills"]),
        pending=tuple(
            PendingOrder(item["client_order_id"], item["ticker"], item["reason"])
            for item in obj["pending"]
        ),
        unreadable_reports=dict(unreadable),
        unsettled=tuple(
            UnsettledOrder(item["client_order_id"], item["ticker"], item["reason"])
            for item in obj["unsettled"]
        ),
        broker_positions=(
            None
            if positions is None
            else tuple(Holding(item["ticker"], decode_int(item["shares"])) for item in positions)
        ),
        open_lots=tuple(_marked_lot(item) for item in obj["open_lots"]),
        unmarked=tuple(_lot(item) for item in obj["unmarked"]),
        closed_lots=tuple(_closed_lot(item) for item in obj["closed_lots"]),
        breaks=tuple(decode_break(item) for item in obj["breaks"]),
        realised_total=decode_float(obj["realised_total"]),
        unrealised_total=decode_float(obj["unrealised_total"]),
    )


def decode_break(item: Mapping[str, Any]) -> ReconciliationBreak:
    """A ReconciliationBreak from its JSON object (also used by the attribution reader)."""
    return ReconciliationBreak(
        ticker=item["ticker"],
        ledger_shares=decode_int(item["ledger_shares"]),
        broker_shares=decode_int(item["broker_shares"]),
    )


def _lot(item: Mapping[str, Any]) -> Lot:
    return Lot(
        ticker=item["ticker"],
        shares=decode_int(item["shares"]),
        entry_price=decode_float(item["entry_price"]),
        entered_at=decode_instant(item["entered_at"]),
        source=item["source"],
    )


def _marked_lot(item: Mapping[str, Any]) -> MarkedLot:
    return MarkedLot(
        lot=_lot(item["lot"]),
        price=decode_float(item["price"]),
        close_date=date.fromisoformat(item["close_date"]),
        unrealised_pnl=decode_float(item["unrealised_pnl"]),
    )


def _closed_lot(item: Mapping[str, Any]) -> ClosedLot:
    return ClosedLot(
        ticker=item["ticker"],
        shares=decode_int(item["shares"]),
        entry_price=decode_float(item["entry_price"]),
        entered_at=decode_instant(item["entered_at"]),
        source=item["source"],
        exit_price=decode_float(item["exit_price"]),
        exited_at=decode_instant(item["exited_at"]),
        exit_client_order_id=item["exit_client_order_id"],
        realised_pnl=decode_float(item["realised_pnl"]),
    )
