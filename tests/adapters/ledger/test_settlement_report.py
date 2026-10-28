"""Tests for the JSON settlement report store."""

from __future__ import annotations

import dataclasses
import json
import math
from datetime import UTC, date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from hedge_fund.adapters.ledger.settlement_report import JsonSettlementReportStore
from hedge_fund.domain.ledger import (
    ClosedLot,
    Fill,
    FillStatus,
    Lot,
    MarkedLot,
    PendingOrder,
    ReconciliationBreak,
    SettlementReport,
    SettleOutcome,
    UnsettledOrder,
)
from hedge_fund.domain.orders import Holding, OrderAction
from hedge_fund.ports.ledger import SettlementReportStore

EASTERN = timezone(timedelta(hours=-4))
OPEN = datetime(2026, 10, 1, 13, 30, tzinfo=UTC)
MIDNIGHT = datetime(2026, 9, 1, tzinfo=UTC)
CID = "hf-2026-09-30-AAA-buy-10"
LOT = Lot(ticker="AAA", shares=10, entry_price=101.0, entered_at=OPEN, source=CID)

# Every field populated; the totals pin the non-finite strings.
REPORT = SettlementReport(
    as_of=date(2026, 10, 1),
    started_at=datetime(2026, 10, 1, 10, 0, 5, tzinfo=EASTERN),  # 14:00:05 UTC
    outcome=SettleOutcome.BREAK,
    failed_stage="lookup",
    error="BrokerError: down",
    new_fills=(
        Fill(
            client_order_id=CID,
            broker_order_id="b-1",
            cycle_as_of=date(2026, 9, 30),
            ticker="AAA",
            action=OrderAction.BUY,
            requested_shares=10,
            filled_shares=10,
            avg_price=101.0,
            filled_at=OPEN,
            status=FillStatus.FILLED,
            recorded_at=datetime(2026, 10, 1, 14, 0, 6, tzinfo=UTC),
        ),
    ),
    pending=(PendingOrder(client_order_id="hf-x", ticker="BBB", reason="open at broker: new"),),
    unreadable_reports={"bad.json": "ValueError: x"},
    unsettled=(
        UnsettledOrder(client_order_id="hf-y", ticker="", reason="no planned order for id"),
    ),
    broker_positions=(Holding(ticker="BRK.B", shares=3),),
    open_lots=(MarkedLot(lot=LOT, price=102.0, close_date=date(2026, 10, 1), unrealised_pnl=10.0),),
    unmarked=(Lot("CCC", -2, 30.0, MIDNIGHT, "opening"),),
    closed_lots=(ClosedLot("DDD", 5, 20.0, MIDNIGHT, "opening", 22.0, OPEN, "hf-z", 10.0),),
    breaks=(ReconciliationBreak(ticker="BRK-B", ledger_shares=0, broker_shares=3),),
    realised_total=math.inf,
    unrealised_total=math.nan,
)
LOT_JSON = {
    "ticker": "AAA",
    "shares": 10,
    "entry_price": 101.0,
    "entered_at": "2026-10-01T13:30:00+00:00",
    "source": CID,
}
EXPECTED: dict[str, Any] = {
    "as_of": "2026-10-01",
    "started_at": "2026-10-01T10:00:05-04:00",
    "outcome": "break",
    "failed_stage": "lookup",
    "error": "BrokerError: down",
    "new_fills": [
        {
            "client_order_id": CID,
            "broker_order_id": "b-1",
            "cycle_as_of": "2026-09-30",
            "ticker": "AAA",
            "action": "buy",
            "requested_shares": 10,
            "filled_shares": 10,
            "avg_price": 101.0,
            "filled_at": "2026-10-01T13:30:00+00:00",
            "status": "filled",
            "recorded_at": "2026-10-01T14:00:06+00:00",
        }
    ],
    "pending": [{"client_order_id": "hf-x", "ticker": "BBB", "reason": "open at broker: new"}],
    "unreadable_reports": {"bad.json": "ValueError: x"},
    "unsettled": [{"client_order_id": "hf-y", "ticker": "", "reason": "no planned order for id"}],
    "broker_positions": [{"ticker": "BRK.B", "shares": 3}],
    "open_lots": [
        {"lot": LOT_JSON, "price": 102.0, "close_date": "2026-10-01", "unrealised_pnl": 10.0}
    ],
    "unmarked": [
        {
            "ticker": "CCC",
            "shares": -2,
            "entry_price": 30.0,
            "entered_at": "2026-09-01T00:00:00+00:00",
            "source": "opening",
        }
    ],
    "closed_lots": [
        {
            "ticker": "DDD",
            "shares": 5,
            "entry_price": 20.0,
            "entered_at": "2026-09-01T00:00:00+00:00",
            "source": "opening",
            "exit_price": 22.0,
            "exited_at": "2026-10-01T13:30:00+00:00",
            "exit_client_order_id": "hf-z",
            "realised_pnl": 10.0,
        }
    ],
    "breaks": [{"ticker": "BRK-B", "ledger_shares": 0, "broker_shares": 3}],
    "realised_total": "inf",
    "unrealised_total": "nan",
}


def _files(root: Path) -> list[Path]:
    return sorted((root / "settlements").iterdir())


def test_satisfies_the_port_and_writes_the_pinned_shape(tmp_path: Path) -> None:
    store: SettlementReportStore = JsonSettlementReportStore(tmp_path)
    store.save(REPORT)
    [path] = _files(tmp_path)
    text = path.read_text(encoding="utf-8")
    assert json.loads(text) == EXPECTED
    assert text == json.dumps(EXPECTED, indent=2)


def test_keys_are_the_report_fields_in_order(tmp_path: Path) -> None:
    JsonSettlementReportStore(tmp_path).save(REPORT)
    [path] = _files(tmp_path)
    keys = list(json.loads(path.read_text(encoding="utf-8")))
    assert keys == [field.name for field in dataclasses.fields(SettlementReport)]
    assert {"unreadable_reports", "unsettled"} <= set(keys)


@pytest.mark.parametrize(
    ("started_at", "name"),
    [
        pytest.param(REPORT.started_at, "2026-10-01_140005.json", id="eastern-offset"),
        pytest.param(
            datetime(2026, 10, 2, 1, 2, 3, tzinfo=UTC), "2026-10-01_010203.json", id="utc"
        ),
    ],
)
def test_file_name_is_as_of_and_utc_start_time(
    tmp_path: Path, started_at: datetime, name: str
) -> None:
    JsonSettlementReportStore(tmp_path).save(dataclasses.replace(REPORT, started_at=started_at))
    assert [path.name for path in _files(tmp_path)] == [name]


def test_an_existing_report_is_never_overwritten(tmp_path: Path) -> None:
    store = JsonSettlementReportStore(tmp_path)
    store.save(REPORT)
    [path] = _files(tmp_path)
    first = path.read_text(encoding="utf-8")
    with pytest.raises(FileExistsError, match="2026-10-01_140005.json"):
        store.save(dataclasses.replace(REPORT, outcome=SettleOutcome.CLEAN))
    assert path.read_text(encoding="utf-8") == first
    assert _files(tmp_path) == [path]


def test_a_failed_write_leaves_no_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write_text = Path.write_text

    def partial_then_fail(self: Path, data: str, *args: Any, **kwargs: Any) -> int:
        write_text(self, data[:10], *args, **kwargs)
        raise OSError("disk full")

    monkeypatch.setattr(Path, "write_text", partial_then_fail)
    with pytest.raises(OSError, match="disk full"):
        JsonSettlementReportStore(tmp_path).save(REPORT)
    assert _files(tmp_path) == []


def test_package_exports() -> None:
    import hedge_fund.adapters.ledger as ledger
    from hedge_fund.adapters.ledger.fills import JsonFillStore
    from hedge_fund.adapters.ledger.opening import JsonOpeningBalance

    assert ledger.__all__ == ["JsonFillStore", "JsonOpeningBalance", "JsonSettlementReportStore"]
    assert ledger.JsonFillStore is JsonFillStore
    assert ledger.JsonOpeningBalance is JsonOpeningBalance
    assert ledger.JsonSettlementReportStore is JsonSettlementReportStore
