"""Layer 7a settlement over fakes: fills, lots, reconciliation, marks and the report."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from typing import Any, NoReturn

import pandas as pd
import pytest

from hedge_fund.domain.cycle import CycleOutcome, CycleReport
from hedge_fund.domain.execution import ExecutionReport, OrderRequest, SubmitResult, SubmitStatus
from hedge_fund.domain.ledger import (
    ClosedLot,
    Fill,
    FillStatus,
    Lot,
    MarkedLot,
    OpeningEntry,
    OrderState,
    OrderStatus,
    PendingOrder,
    ReconciliationBreak,
    SettlementReport,
    SettleOutcome,
    UnsettledOrder,
)
from hedge_fund.domain.orders import Holding, Order, OrderAction, OrderPlan
from hedge_fund.engine.ledger import NO_PLANNED_ORDER
from hedge_fund.ports.broker import BrokerError, OrderDataError
from hedge_fund.ports.data import DataProviderError
from hedge_fund.services.settle import (
    ACKNOWLEDGED_NOT_FOUND,
    SettleDeps,
    SettlementSaveError,
    run_settle,
)
from tests.fakes import (
    FakeProvider,
    InMemoryFillStore,
    InMemoryReportStore,
    InMemorySettlementStore,
    ScriptedBroker,
    StaticOpeningBalance,
)

AS_OF = date(2026, 10, 1)  # Thursday: the settle day
CYCLE_DAY = date(2026, 9, 30)  # Wednesday: the cycle that submitted
STARTED = datetime(2026, 9, 30, 22, 0, tzinfo=UTC)  # the cycle's start, 18:00 EDT
OPEN = datetime(2026, 10, 1, 13, 30, tzinfo=UTC)  # 09:30 EDT: the opg orders fill
MORNING = datetime(2026, 10, 1, 14, 0, tzinfo=UTC)  # the settle run, before the deadline
DEADLINE = datetime(2026, 10, 1, 20, 0, tzinfo=UTC)  # Thursday 16:00 EDT
OPENED = date(2026, 9, 1)
MIDNIGHT = datetime(2026, 9, 1, tzinfo=UTC)

SELL_CCC = Order(ticker="CCC", action=OrderAction.SELL, shares=20, ref_price=30.0, notional=600.0)
BUY_AAA = Order(ticker="AAA", action=OrderAction.BUY, shares=10, ref_price=100.0, notional=1000.0)
SHORT_BBB = Order(
    ticker="BBB", action=OrderAction.SELL_SHORT, shares=5, ref_price=50.0, notional=250.0
)
FILL_PRICES = {"CCC": 33.0, "AAA": 101.0, "BBB": 50.0}
HOLDINGS = (Holding(ticker="AAA", shares=10), Holding(ticker="BBB", shares=-5))
OPENING = (OpeningEntry(ticker="CCC", shares=20, price=30.0, on=OPENED),)
CLOSES = pd.DataFrame(
    {"AAA": [100.0, 102.0], "BBB": [51.0, 49.0]},
    index=pd.to_datetime(["2026-09-30", "2026-10-01"]),
)


def _cid(order: Order) -> str:
    return f"hf-{CYCLE_DAY}-{order.ticker}-{order.action.value}-{order.shares}"


def _request(order: Order) -> OrderRequest:
    return OrderRequest(order=order, client_order_id=_cid(order))


def _cycle(*legs: tuple[Order, SubmitStatus], started_at: datetime = STARTED) -> CycleReport:
    """A submitted cycle whose execution holds ``legs``."""
    results = tuple(
        SubmitResult(_request(order), status, f"b-{_cid(order)}", None) for order, status in legs
    )
    return CycleReport(
        as_of=CYCLE_DAY,
        started_at=started_at,
        dry_run=False,
        outcome=CycleOutcome.SUBMITTED,
        failed_stage=None,
        error=None,
        account=None,
        price_unresolved={},
        fundamentals_unresolved={},
        shortlist=None,
        research=None,
        research_reused=False,
        target=None,
        verdict=None,
        order_prices=None,
        plan=OrderPlan(CYCLE_DAY, tuple(order for order, _ in legs), {}, {}, None),
        decision=None,
        execution=ExecutionReport(CYCLE_DAY, None, results, opening_halted=False),
        execution_interrupted=False,
        planned_client_order_ids=(),
    )


CYCLE = _cycle(
    (SELL_CCC, SubmitStatus.ACCEPTED),
    (BUY_AAA, SubmitStatus.ACCEPTED),
    (SHORT_BBB, SubmitStatus.ACCEPTED),
)


def _side(order: Order) -> str:
    return "buy" if order.action in {OrderAction.BUY, OrderAction.BUY_TO_COVER} else "sell"


def _filled(order: Order) -> OrderStatus:
    return OrderStatus(
        client_order_id=_cid(order),
        state=OrderState.FILLED,
        broker_order_id=f"b-{_cid(order)}",
        ticker=order.ticker,
        side=_side(order),
        filled_shares=Decimal(order.shares),
        avg_price=FILL_PRICES.get(order.ticker, 10.0),
        filled_at=OPEN,
        broker_state="filled",
    )


def _fill(order: Order, recorded_at: datetime = MORNING) -> Fill:
    return Fill(
        client_order_id=_cid(order),
        broker_order_id=f"b-{_cid(order)}",
        cycle_as_of=CYCLE_DAY,
        ticker=order.ticker,
        action=order.action,
        requested_shares=order.shares,
        filled_shares=order.shares,
        avg_price=FILL_PRICES.get(order.ticker, 10.0),
        filled_at=OPEN,
        status=FillStatus.FILLED,
        recorded_at=recorded_at,
    )


ALL_FILLED = {_cid(order): _filled(order) for order in (SELL_CCC, BUY_AAA, SHORT_BBB)}
AAA_LOT = Lot(ticker="AAA", shares=10, entry_price=101.0, entered_at=OPEN, source=_cid(BUY_AAA))
BBB_LOT = Lot(ticker="BBB", shares=-5, entry_price=50.0, entered_at=OPEN, source=_cid(SHORT_BBB))
CCC_CLOSED = ClosedLot(
    ticker="CCC",
    shares=20,
    entry_price=30.0,
    entered_at=MIDNIGHT,
    source="opening",
    exit_price=33.0,
    exited_at=OPEN,
    exit_client_order_id=_cid(SELL_CCC),
    realised_pnl=60.0,
)


@dataclass
class World:
    """The fakes one settle runs over; every field may be replaced by a test."""

    broker: ScriptedBroker = field(
        default_factory=lambda: ScriptedBroker(orders=ALL_FILLED, holdings=HOLDINGS)
    )
    prices: Any = field(default_factory=lambda: FakeProvider(CLOSES, {}))
    cycle_reports: Any = field(default_factory=lambda: InMemoryReportStore([CYCLE]))
    fills: Any = field(default_factory=InMemoryFillStore)
    opening: Any = field(default_factory=lambda: StaticOpeningBalance(OPENING))
    report_store: InMemorySettlementStore = field(default_factory=InMemorySettlementStore)

    @property
    def deps(self) -> SettleDeps:
        return SettleDeps(
            broker=self.broker,
            prices=self.prices,
            cycle_reports=self.cycle_reports,
            fills=self.fills,
            opening=self.opening,
            report_store=self.report_store,
        )

    def script(self, order: Order, outcome: OrderStatus | BaseException) -> None:
        self.broker.orders[_cid(order)] = outcome


class Boom:
    """A port whose every method raises ``error``."""

    def __init__(self, error: BaseException) -> None:
        self.error = error

    def __getattr__(self, name: str) -> Callable[..., NoReturn]:
        def fail(*args: Any, **kwargs: Any) -> NoReturn:
            raise self.error

        return fail


def _settle(
    world: World, *, as_of: date = AS_OF, today: date = AS_OF, now: datetime = MORNING
) -> SettlementReport:
    return run_settle(world.deps, as_of=as_of, today=today, now=lambda: now)


# --- completed runs ----------------------------------------------------------------------


def test_a_clean_settle_records_every_fill_and_saves_the_report() -> None:
    world = World()
    report = _settle(world)
    assert report == SettlementReport(
        as_of=AS_OF,
        started_at=MORNING,
        outcome=SettleOutcome.CLEAN,
        failed_stage=None,
        error=None,
        new_fills=(_fill(BUY_AAA), _fill(SHORT_BBB), _fill(SELL_CCC)),
        pending=(),
        unreadable_reports={},
        unsettled=(),
        broker_positions=HOLDINGS,
        open_lots=(
            MarkedLot(lot=AAA_LOT, price=102.0, close_date=AS_OF, unrealised_pnl=10.0),
            MarkedLot(lot=BBB_LOT, price=49.0, close_date=AS_OF, unrealised_pnl=5.0),
        ),
        unmarked=(),
        closed_lots=(CCC_CLOSED,),
        breaks=(),
        realised_total=60.0,
        unrealised_total=15.0,
    )
    assert world.report_store.saved == [report]
    assert world.fills.saved == list(report.new_fills)
    assert world.broker.looked_up == sorted(ALL_FILLED)
    assert world.broker.submitted == []
    assert world.prices.closes_calls == [(("AAA", "BBB"), date(2026, 9, 17), AS_OF)]


def test_a_position_that_differs_from_the_lots_is_a_break() -> None:
    world = World(
        broker=ScriptedBroker(orders=ALL_FILLED, holdings=(Holding("AAA", 12), Holding("BBB", -5)))
    )
    report = _settle(world)
    assert report.outcome is SettleOutcome.BREAK
    assert report.breaks == (ReconciliationBreak(ticker="AAA", ledger_shares=10, broker_shares=12),)
    assert (report.realised_total, report.unrealised_total) == (60.0, 5.0)  # AAA left out


def test_an_unreadable_run_file_is_listed_and_makes_the_run_pending() -> None:
    world = World()
    world.cycle_reports.unreadable = {"2026-09-29_220000.json": "KeyError: 'started_at'"}
    report = _settle(world)
    assert report.outcome is SettleOutcome.PENDING
    assert report.unreadable_reports == {"2026-09-29_220000.json": "KeyError: 'started_at'"}
    assert len(report.new_fills) == 3


def test_a_planned_id_without_an_order_is_unsettled() -> None:
    ghost = "hf-2026-09-30-ZZZ-buy-1"
    interrupted = dataclasses.replace(
        _cycle(started_at=STARTED + timedelta(hours=1)),
        plan=None,
        execution=None,
        execution_interrupted=True,
        planned_client_order_ids=(ghost,),
    )
    world = World(cycle_reports=InMemoryReportStore([CYCLE, interrupted]))
    report = _settle(world)
    assert report.outcome is SettleOutcome.UNSETTLED
    assert report.unsettled == (UnsettledOrder(ghost, "", NO_PLANNED_ORDER),)
    assert report.breaks == ()
    assert ghost not in world.broker.looked_up


def test_recorded_ids_are_not_looked_up_again() -> None:
    world = World(fills=InMemoryFillStore([_fill(BUY_AAA)]))
    report = _settle(world)
    assert world.broker.looked_up == [_cid(SHORT_BBB), _cid(SELL_CCC)]
    assert report.new_fills == (_fill(SHORT_BBB), _fill(SELL_CCC))
    assert [marked.lot for marked in report.open_lots] == [AAA_LOT, BBB_LOT]
    assert report.outcome is SettleOutcome.CLEAN


def test_every_lookup_precedes_the_positions_read() -> None:
    events: list[str] = []

    class RecordingBroker(ScriptedBroker):
        def order_status(self, client_order_id: str) -> OrderStatus:
            events.append("order_status")
            return super().order_status(client_order_id)

        def positions(self) -> tuple[Holding, ...]:
            events.append("positions")
            return super().positions()

    _settle(World(broker=RecordingBroker(orders=ALL_FILLED, holdings=HOLDINGS)))
    assert events == ["order_status"] * 3 + ["positions"]


def test_a_second_run_over_the_same_stores_records_nothing_new() -> None:
    world = World()
    first = _settle(world)
    later = MORNING + timedelta(minutes=1)  # the report store refuses a repeated second
    second = _settle(world, now=later)
    assert second.started_at == later
    assert second.new_fills == ()
    assert len(world.broker.looked_up) == 3
    assert len(world.fills.saved) == 3
    assert (second.outcome, second.open_lots, second.closed_lots) == (
        first.outcome,
        first.open_lots,
        first.closed_lots,
    )
    assert world.report_store.saved == [first, second]


def test_without_open_lots_no_prices_are_read() -> None:
    world = World(
        broker=ScriptedBroker(orders=ALL_FILLED),
        cycle_reports=InMemoryReportStore([_cycle((SELL_CCC, SubmitStatus.ACCEPTED))]),
    )
    report = _settle(world)
    assert report.outcome is SettleOutcome.CLEAN
    assert (report.open_lots, report.unmarked) == ((), ())
    assert world.prices.closes_calls == []


def test_started_at_is_the_first_clock_read_and_each_fill_takes_a_fresh_one() -> None:
    reads = iter([MORNING + timedelta(seconds=second) for second in range(4)])
    report = run_settle(World().deps, as_of=AS_OF, today=AS_OF, now=lambda: next(reads))
    assert report.started_at == MORNING
    assert [fill.recorded_at for fill in report.new_fills] == [
        MORNING + timedelta(seconds=second) for second in (1, 2, 3)
    ]


def test_as_of_only_dates_the_marks() -> None:
    world = World()
    report = _settle(world, as_of=CYCLE_DAY)  # the previous weekday of today is live
    assert report.as_of == CYCLE_DAY
    assert [(m.price, m.close_date) for m in report.open_lots] == [
        (100.0, CYCLE_DAY),
        (51.0, CYCLE_DAY),
    ]
    assert world.prices.closes_calls == [(("AAA", "BBB"), date(2026, 9, 16), CYCLE_DAY)]


# --- failed runs -------------------------------------------------------------------------


@pytest.mark.parametrize(
    "as_of",
    [
        pytest.param(date(2026, 9, 29), id="before-the-previous-weekday"),
        pytest.param(date(2026, 10, 2), id="after-today"),
    ],
)
def test_an_as_of_that_is_not_live_fails_before_any_read(as_of: date) -> None:
    world = World(cycle_reports=Boom(AssertionError("reports were read")))
    with pytest.raises(ValueError, match=f"as_of {as_of} is not live"):
        _settle(world, as_of=as_of)
    [report] = world.report_store.saved
    assert (report.outcome, report.failed_stage) == (SettleOutcome.ERROR, "inputs")
    assert report.error is not None
    assert report.error.startswith(f"ValueError: as_of {as_of} is not live")
    assert world.broker.looked_up == []
    assert (report.realised_total, report.unrealised_total) == (0.0, 0.0)


@pytest.mark.parametrize(
    ("overrides", "stage", "error"),
    [
        pytest.param(
            {"cycle_reports": Boom(OSError("runs unreadable"))}, "reports", OSError, id="reports"
        ),
        pytest.param(
            {"fills": InMemoryFillStore(load_error=ValueError("x.json: malformed fill record"))},
            "fills",
            ValueError,
            id="fills",
        ),
        pytest.param(
            {"opening": StaticOpeningBalance(error=ValueError("opening.json entry 0: bad"))},
            "opening",
            ValueError,
            id="opening",
        ),
        pytest.param(
            {"opening": StaticOpeningBalance([OpeningEntry("CCC", 20, 30.0, AS_OF)])},
            "lots",
            ValueError,
            id="lots",
        ),
        pytest.param(
            {
                "broker": ScriptedBroker(
                    orders=ALL_FILLED, holdings=HOLDINGS, positions_error=BrokerError("down")
                )
            },
            "positions",
            BrokerError,
            id="positions",
        ),
        pytest.param(
            {"prices": Boom(DataProviderError("vendor down"))},
            "marks",
            DataProviderError,
            id="marks",
        ),
    ],
)
def test_a_failing_stage_raises_and_saves_an_error_report(
    overrides: dict[str, Any], stage: str, error: type[Exception]
) -> None:
    world = World(**overrides)
    with pytest.raises(error):
        _settle(world)
    [report] = world.report_store.saved
    assert (report.outcome, report.failed_stage) == (SettleOutcome.ERROR, stage)
    assert report.error is not None
    assert report.error.startswith(f"{error.__name__}: ")
    assert (report.realised_total, report.unrealised_total) == (0.0, 0.0)


def test_a_transport_error_mid_lookup_stops_the_run_and_keeps_earlier_fills() -> None:
    world = World()
    world.script(SHORT_BBB, BrokerError("GET /v2/orders:by_client_order_id: HTTP 401"))
    with pytest.raises(BrokerError, match="HTTP 401"):
        _settle(world)
    [report] = world.report_store.saved
    assert (report.outcome, report.failed_stage) == (SettleOutcome.ERROR, "lookup")
    assert report.error == "BrokerError: GET /v2/orders:by_client_order_id: HTTP 401"
    assert report.new_fills == (_fill(BUY_AAA),)
    assert world.fills.saved == [_fill(BUY_AAA)]
    assert world.broker.looked_up == [_cid(BUY_AAA), _cid(SHORT_BBB)]
    assert report.broker_positions is None


def test_an_interrupt_raises_and_saves_an_interrupted_report() -> None:
    world = World()
    world.script(SHORT_BBB, KeyboardInterrupt())
    with pytest.raises(KeyboardInterrupt):
        _settle(world)
    [report] = world.report_store.saved
    assert (report.outcome, report.failed_stage) == (SettleOutcome.INTERRUPTED, "lookup")
    assert report.new_fills == (_fill(BUY_AAA),)


def test_a_completed_run_whose_report_cannot_be_saved_raises_settlement_save_error() -> None:
    disk_full = OSError("disk full")
    world = World(report_store=InMemorySettlementStore(fail=disk_full))
    with pytest.raises(SettlementSaveError, match="the clean settlement report") as info:
        _settle(world)
    assert info.value.report.outcome is SettleOutcome.CLEAN
    assert info.value.__cause__ is disk_full
    assert len(world.fills.saved) == 3


def test_a_failed_run_whose_report_cannot_be_saved_raises_the_original_error() -> None:
    world = World(report_store=InMemorySettlementStore(fail=OSError("disk full")))
    world.script(SHORT_BBB, BrokerError("down"))
    with pytest.raises(BrokerError, match="down") as info:
        _settle(world)
    assert info.value.__notes__ == ["the settlement report could not be saved: OSError: disk full"]


# --- the lookup loop ---------------------------------------------------------------------


def _not_final(order: Order, state: OrderState, broker_state: str) -> OrderStatus:
    return dataclasses.replace(
        _filled(order),
        state=state,
        filled_shares=Decimal(0),
        avg_price=None,
        filled_at=None,
        broker_state=broker_state,
    )


def test_an_open_order_is_pending_and_its_ticker_is_not_a_break() -> None:
    world = World()  # HOLDINGS has BBB -5: the order filled after the lookup
    world.script(SHORT_BBB, _not_final(SHORT_BBB, OrderState.PENDING, "accepted"))
    report = _settle(world)
    assert report.outcome is SettleOutcome.PENDING
    assert report.pending == (PendingOrder(_cid(SHORT_BBB), "BBB", "open at broker: accepted"),)
    assert report.breaks == ()
    assert [fill.ticker for fill in report.new_fills] == ["AAA", "CCC"]


@pytest.mark.parametrize(
    ("outcome", "reason"),
    [
        pytest.param(
            _not_final(SHORT_BBB, OrderState.UNRECOGNISED, "held"),
            "unrecognised broker status 'held'",
            id="unrecognised",
        ),
        pytest.param(
            OrderDataError("GET /v2/orders:by_client_order_id: order 'x': side is missing"),
            "GET /v2/orders:by_client_order_id: order 'x': side is missing",
            id="order-data-error",
        ),
        pytest.param(
            dataclasses.replace(_filled(SHORT_BBB), side="buy"),
            "side 'buy' does not match sell_short",
            id="to-fill-refusal",
        ),
    ],
)
def test_an_order_that_cannot_be_recorded_is_unsettled_and_the_loop_goes_on(
    outcome: OrderStatus | Exception, reason: str
) -> None:
    world = World()
    world.script(SHORT_BBB, outcome)
    report = _settle(world)
    assert report.outcome is SettleOutcome.UNSETTLED
    assert report.unsettled == (UnsettledOrder(_cid(SHORT_BBB), "BBB", reason),)
    assert world.broker.looked_up == sorted(ALL_FILLED)
    assert [fill.client_order_id for fill in world.fills.saved] == [
        _cid(BUY_AAA),
        _cid(SELL_CCC),
    ]
    assert report.breaks == ()  # BBB -5 at the broker, no BBB lot: excluded, not a break


def test_error_legs_are_looked_up() -> None:
    world = World(
        cycle_reports=InMemoryReportStore([_cycle((BUY_AAA, SubmitStatus.ERROR))]),
        opening=StaticOpeningBalance(),
    )
    world.broker.holdings = (Holding("AAA", 10),)
    report = _settle(world)
    assert world.broker.looked_up == [_cid(BUY_AAA)]
    assert report.new_fills == (_fill(BUY_AAA),)
    assert report.outcome is SettleOutcome.CLEAN


def _unacknowledged_world(status: SubmitStatus = SubmitStatus.ERROR) -> World:
    """One BUY 10 AAA leg the broker cannot find; no positions, no opening."""
    return World(
        broker=ScriptedBroker(),
        cycle_reports=InMemoryReportStore([_cycle((BUY_AAA, status))]),
        opening=StaticOpeningBalance(),
    )


def test_an_unacknowledged_order_not_found_before_the_deadline_is_pending() -> None:
    world = _unacknowledged_world()
    report = _settle(world, now=DEADLINE - timedelta(seconds=1))
    assert report.outcome is SettleOutcome.PENDING
    assert report.pending == (
        PendingOrder(_cid(BUY_AAA), "AAA", "not found before 2026-10-01T20:00:00+00:00"),
    )
    assert world.fills.saved == []


@pytest.mark.parametrize(
    "now",
    [
        pytest.param(DEADLINE, id="at-the-deadline"),
        pytest.param(DEADLINE + timedelta(hours=3), id="after"),
    ],
)
def test_an_unacknowledged_order_not_found_after_the_deadline_is_not_submitted(
    now: datetime,
) -> None:
    world = _unacknowledged_world()
    report = _settle(world, now=now)
    never = Fill(
        client_order_id=_cid(BUY_AAA),
        broker_order_id=None,
        cycle_as_of=CYCLE_DAY,
        ticker="AAA",
        action=OrderAction.BUY,
        requested_shares=10,
        filled_shares=0,
        avg_price=None,
        filled_at=None,
        status=FillStatus.NOT_SUBMITTED,
        recorded_at=now,
    )
    assert report.new_fills == (never,)
    assert world.fills.saved == [never]
    assert report.outcome is SettleOutcome.CLEAN


@pytest.mark.parametrize(
    "now",
    [pytest.param(MORNING, id="before-deadline"), pytest.param(DEADLINE, id="after-deadline")],
)
def test_an_acknowledged_order_not_found_is_unsettled_and_never_recorded(now: datetime) -> None:
    world = _unacknowledged_world(SubmitStatus.ACCEPTED)
    report = _settle(world, now=now)
    assert report.outcome is SettleOutcome.UNSETTLED
    assert report.unsettled == (UnsettledOrder(_cid(BUY_AAA), "AAA", ACKNOWLEDGED_NOT_FOUND),)
    assert ACKNOWLEDGED_NOT_FOUND == "acknowledged by broker but not found"
    assert world.fills.saved == []


# --- precedence and totals ---------------------------------------------------------------

BROKEN_AAA = (Holding("AAA", 12), Holding("BBB", -5))


@pytest.mark.parametrize(
    ("bbb", "outcome"),
    [
        pytest.param(OrderState.UNRECOGNISED, SettleOutcome.UNSETTLED, id="unsettled-over-break"),
        pytest.param(OrderState.PENDING, SettleOutcome.BREAK, id="break-over-pending"),
    ],
)
def test_outcome_precedence_and_every_list_is_reported(
    bbb: OrderState, outcome: SettleOutcome
) -> None:
    world = World(broker=ScriptedBroker(orders=ALL_FILLED, holdings=BROKEN_AAA))
    world.script(SHORT_BBB, _not_final(SHORT_BBB, bbb, "held"))
    report = _settle(world)
    assert report.outcome is outcome
    assert report.breaks == (ReconciliationBreak("AAA", 10, 12),)
    assert len(report.pending) + len(report.unsettled) == 1


def test_an_error_outranks_unsettled_and_keeps_the_lists() -> None:
    world = World(
        broker=ScriptedBroker(
            orders=ALL_FILLED, holdings=HOLDINGS, positions_error=BrokerError("down")
        )
    )
    world.script(SHORT_BBB, _not_final(SHORT_BBB, OrderState.UNRECOGNISED, "held"))
    with pytest.raises(BrokerError):
        _settle(world)
    [report] = world.report_store.saved
    assert (report.outcome, report.failed_stage) == (SettleOutcome.ERROR, "positions")
    assert [order.ticker for order in report.unsettled] == ["BBB"]
    assert len(report.new_fills) == 2


def test_totals_leave_out_broken_pending_unsettled_and_unmarked_tickers() -> None:
    names = ("G", "K", "M", "P", "U")  # good, break, unmarked, pending, unsettled
    earlier = datetime(2026, 9, 15, 13, 30, tzinfo=UTC)
    recorded = [
        Fill(
            client_order_id=f"hf-2026-09-14-{name}-sell-5",
            broker_order_id=f"b-{name}",
            cycle_as_of=date(2026, 9, 14),
            ticker=name,
            action=OrderAction.SELL,
            requested_shares=5,
            filled_shares=5,
            avg_price=12.0,  # each realises (12 - 10) * 5 = 10.00
            filled_at=earlier,
            status=FillStatus.FILLED,
            recorded_at=earlier,
        )
        for name in names
    ]
    buy_p = Order(ticker="P", action=OrderAction.BUY, shares=1, ref_price=10.0, notional=10.0)
    buy_u = dataclasses.replace(buy_p, ticker="U")
    world = World(
        broker=ScriptedBroker(
            orders={
                _cid(buy_p): _not_final(buy_p, OrderState.PENDING, "new"),
                _cid(buy_u): _not_final(buy_u, OrderState.UNRECOGNISED, "held"),
            },
            holdings=tuple(Holding(name, 7 if name == "K" else 5) for name in names),
        ),
        prices=FakeProvider(
            pd.DataFrame(
                {name: [11.0] for name in ("G", "K", "P", "U")}, index=pd.to_datetime([AS_OF])
            ),
            {},
        ),
        cycle_reports=InMemoryReportStore(
            [_cycle((buy_p, SubmitStatus.ACCEPTED), (buy_u, SubmitStatus.ACCEPTED))]
        ),
        fills=InMemoryFillStore(recorded),
        opening=StaticOpeningBalance([OpeningEntry(name, 10, 10.0, OPENED) for name in names]),
    )
    report = _settle(world)
    assert report.outcome is SettleOutcome.UNSETTLED
    assert [closed.ticker for closed in report.closed_lots] == list(names)
    assert [marked.lot.ticker for marked in report.open_lots] == ["G", "K", "P", "U"]
    assert [lot.ticker for lot in report.unmarked] == ["M"]
    assert report.breaks == (ReconciliationBreak("K", 5, 7),)
    assert (report.realised_total, report.unrealised_total) == (10.0, 5.0)  # G only
