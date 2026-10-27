"""Tests for the ledger ports, exercised through the in-memory fakes."""

from __future__ import annotations

import dataclasses
from datetime import UTC, date, datetime, timedelta

import pytest

from hedge_fund.domain.ledger import (
    Fill,
    FillStatus,
    OpeningEntry,
    SettlementReport,
    SettleOutcome,
)
from hedge_fund.domain.orders import OrderAction
from hedge_fund.ports.ledger import FillStore, OpeningBalance, SettlementReportStore
from tests.fakes import InMemoryFillStore, InMemorySettlementStore, StaticOpeningBalance

AT = datetime(2026, 10, 1, 14, 0, tzinfo=UTC)
FILL = Fill(
    client_order_id="hf-2026-09-30-AAA-buy-10",
    broker_order_id="b-1",
    cycle_as_of=date(2026, 9, 30),
    ticker="AAA",
    action=OrderAction.BUY,
    requested_shares=10,
    filled_shares=10,
    avg_price=101.0,
    filled_at=AT,
    status=FillStatus.FILLED,
    recorded_at=AT,
)
REPORT = SettlementReport(
    as_of=date(2026, 10, 1),
    started_at=AT,
    outcome=SettleOutcome.CLEAN,
    failed_stage=None,
    error=None,
    new_fills=(),
    pending=(),
    unreadable_reports={},
    unsettled=(),
    broker_positions=(),
    open_lots=(),
    unmarked=(),
    closed_lots=(),
    breaks=(),
    realised_total=0.0,
    unrealised_total=0.0,
)


def test_in_memory_fill_store_satisfies_the_port_and_writes_once() -> None:
    fake = InMemoryFillStore()
    store: FillStore = fake
    store.save(FILL)
    with pytest.raises(FileExistsError, match=FILL.client_order_id):
        store.save(dataclasses.replace(FILL, filled_shares=3))
    assert store.load_all() == (FILL,)
    assert fake.saved == [FILL]


def test_in_memory_fill_store_starts_from_given_fills_and_can_fail_to_load() -> None:
    assert InMemoryFillStore([FILL]).load_all() == (FILL,)
    with pytest.raises(ValueError, match="malformed"):
        InMemoryFillStore(load_error=ValueError("malformed")).load_all()


def test_static_opening_balance_satisfies_the_port() -> None:
    entry = OpeningEntry(ticker="AAA", shares=10, price=90.0, on=date(2026, 9, 1))
    opening: OpeningBalance = StaticOpeningBalance([entry])
    assert opening.load() == (entry,)
    assert StaticOpeningBalance().load() == ()
    with pytest.raises(ValueError, match="bad entry"):
        StaticOpeningBalance(error=ValueError("bad entry")).load()


def test_in_memory_settlement_store_satisfies_the_port() -> None:
    fake = InMemorySettlementStore()
    store: SettlementReportStore = fake
    store.save(REPORT)
    assert fake.saved == [REPORT]


def test_in_memory_settlement_store_refuses_a_repeat_as_of_and_second() -> None:
    fake = InMemorySettlementStore()
    fake.save(REPORT)
    with pytest.raises(FileExistsError, match="2026-10-01"):
        fake.save(dataclasses.replace(REPORT, started_at=AT + timedelta(microseconds=5)))
    later = dataclasses.replace(REPORT, started_at=AT + timedelta(seconds=1))
    fake.save(later)
    assert fake.saved == [REPORT, later]


def test_in_memory_settlement_store_fails_when_told_to_and_keeps_nothing() -> None:
    fake = InMemorySettlementStore(fail=OSError("disk full"))
    with pytest.raises(OSError, match="disk full"):
        fake.save(REPORT)
    assert fake.saved == []
