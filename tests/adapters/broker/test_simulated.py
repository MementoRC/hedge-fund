from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from fractions import Fraction

import pandas as pd
import pytest

from hedge_fund.adapters.broker.simulated import SimulatedBroker, SimulationError
from hedge_fund.domain.backtest import Dividend, SimCosts, SimEvent, SimEventKind
from hedge_fund.domain.execution import OrderRequest, SubmitStatus
from hedge_fund.domain.ledger import FillStatus, OrderState, Split
from hedge_fund.domain.orders import Holding, Order, OrderAction
from hedge_fund.engine.ledger.open_order import is_open
from hedge_fund.engine.ledger.validate import to_fill
from hedge_fund.ports.broker import Broker
from hedge_fund.services.account import record_snapshot
from tests.factories import make_frozen_data
from tests.fakes import InMemorySnapshotStore, not_found

DAYS = pd.bdate_range("2024-01-01", periods=5)
D1, D2, D3, D4, D5 = (d.date() for d in DAYS)
_NOW = datetime(2024, 1, 2, tzinfo=UTC)


def _data():
    closes = pd.DataFrame(
        {"AAA": [10.0, 11.0, 12.0, 13.0, 14.0], "BBB": [50.0, 49.0, 48.0, 47.0, 46.0]},
        index=DAYS,
    )
    return make_frozen_data(
        start=D1,
        end=D5,
        opens=closes - 0.5,
        closes=closes,
        splits=(),
        dividends=(),
    )


def _broker(cash: float = 100_000.0) -> SimulatedBroker:
    return SimulatedBroker(_data(), cash=cash)


def _request(
    cid: str = "c1",
    ticker: str = "AAA",
    shares: int = 10,
    action: OrderAction = OrderAction.BUY,
) -> OrderRequest:
    order = Order(ticker, action, shares, 10.0, 10.0 * shares)
    return OrderRequest(order=order, client_order_id=cid)


def test_simulated_broker_satisfies_the_port() -> None:
    broker: Broker = SimulatedBroker(_data(), cash=100_000.0)
    assert broker.account().equity == 100_000.0


def test_fresh_broker_has_cash_equity_and_no_positions() -> None:
    broker = _broker(12_345.0)
    assert broker.account().equity == 12_345.0
    assert broker.positions() == ()
    assert broker.events == ()


def test_equity_marks_holdings_and_positions_are_sorted_without_zeros() -> None:
    broker = _broker(1_000.0)
    broker.shares = {"BBB": -2, "AAA": 5, "ZERO": 0}
    broker.marks = {"BBB": 50.0, "AAA": 10.0, "ZERO": 1.0}
    assert broker.account().equity == 1_000.0 + 50.0 - 100.0
    assert broker.positions() == (Holding("AAA", 5), Holding("BBB", -2))


def test_submit_accepts_then_reports_duplicate_without_a_new_order() -> None:
    broker = _broker()
    first = broker.submit(_request())
    again = broker.submit(_request())
    other = broker.submit(_request("c2"))
    assert first.status is SubmitStatus.ACCEPTED
    assert first.broker_order_id == "sim-1"
    assert first.reason is None
    assert again.status is SubmitStatus.DUPLICATE
    assert again.broker_order_id == "sim-1"
    assert again.reason is None
    assert other.broker_order_id == "sim-2"
    assert len(broker.orders) == 2


@pytest.mark.parametrize(
    ("ticker", "shares", "reason"),
    [("AAA", 0, "shares"), ("AAA", -3, "shares"), ("ZZZ", 5, "ZZZ")],
)
def test_submit_rejects_bad_requests(ticker: str, shares: int, reason: str) -> None:
    broker = _broker()
    result = broker.submit(_request("bad", ticker, shares))
    assert result.status is SubmitStatus.REJECTED
    assert result.broker_order_id is None
    assert result.reason is not None
    assert reason in result.reason
    assert broker.orders == {}
    assert broker.order_status("bad") == not_found("bad")
    (event,) = broker.events
    assert event == SimEvent(
        day=event.day,
        kind=SimEventKind.REJECT,
        ticker=ticker,
        shares=0,
        price=0.0,
        cash=0.0,
        client_order_id="bad",
        reason=result.reason,
    )


def test_reject_event_carries_the_current_day() -> None:
    broker = _broker()
    broker.open(D1)
    broker.submit(_request("bad", shares=0))
    assert broker.events[0].day == D1


def test_pending_status_shape() -> None:
    broker = _broker()
    broker.submit(_request("c1", "BBB", 7, OrderAction.SELL_SHORT))
    status = broker.order_status("c1")
    assert status.client_order_id == "c1"
    assert status.state is OrderState.PENDING
    assert status.broker_order_id == "sim-1"
    assert status.ticker == "BBB"
    assert status.side == "sell"
    assert status.filled_shares == Decimal(0)
    assert status.avg_price is None
    assert status.filled_at is None
    assert status.broker_state == "pending"


@pytest.mark.parametrize(
    ("action", "side"),
    [
        (OrderAction.BUY, "buy"),
        (OrderAction.BUY_TO_COVER, "buy"),
        (OrderAction.SELL, "sell"),
        (OrderAction.SELL_SHORT, "sell"),
    ],
)
def test_status_side_follows_the_action(action: OrderAction, side: str) -> None:
    broker = _broker()
    broker.submit(_request(action=action))
    assert broker.order_status("c1").side == side


def test_status_of_a_filled_order_reports_the_fill_at_the_open_in_utc() -> None:
    broker = _broker()
    broker.submit(_request("c1", "AAA", 10))
    sim = broker.orders["c1"]
    sim.state, sim.fill_price, sim.filled_on = OrderState.FILLED, 10.05, D1
    status = broker.order_status("c1")
    assert status.state is OrderState.FILLED
    assert status.filled_shares == Decimal(10)
    assert status.avg_price == 10.05
    assert status.filled_at == datetime(2024, 1, 1, 14, 30, tzinfo=UTC)
    assert status.broker_state == "filled"
    assert not is_open(
        status.state, acknowledged=True, deadline=datetime(2030, 1, 1, tzinfo=UTC), now=_NOW
    )


def test_status_of_an_order_rejected_at_fill_time_has_no_fill() -> None:
    broker = _broker()
    broker.submit(_request("c1"))
    broker.orders["c1"].state = OrderState.REJECTED
    status = broker.order_status("c1")
    assert status.state is OrderState.REJECTED
    assert (status.filled_shares, status.avg_price, status.filled_at) == (Decimal(0), None, None)
    assert status.broker_state == "rejected"


def test_unknown_order_is_not_found() -> None:
    assert _broker().order_status("nope") == not_found("nope")


def test_pending_order_is_open() -> None:
    broker = _broker()
    broker.submit(_request())
    now = datetime(2024, 1, 2, tzinfo=UTC)
    state = broker.order_status("c1").state
    assert is_open(state, acknowledged=True, deadline=now + timedelta(hours=1), now=now)


def test_orders_remember_the_submission_day() -> None:
    broker = _broker()
    broker.submit(_request("early"))
    broker.open(D1)
    broker.submit(_request("late"))
    assert [o.submitted_on for o in broker.orders.values()] == [None, D1]
    assert list(broker.orders) == ["early", "late"]


def test_day_bookkeeping() -> None:
    broker = _broker()
    assert broker.day is None
    broker.open(D1)
    assert (broker.day, broker.phase) == (D1, "open")
    broker.close(D1)
    assert (broker.day, broker.phase) == (D1, "closed")
    broker.open(D3)
    assert (broker.day, broker.phase) == (D3, "open")


def test_open_twice_is_an_error() -> None:
    broker = _broker()
    broker.open(D1)
    with pytest.raises(SimulationError, match="after the last day"):
        broker.open(D1)


def test_close_before_open_is_an_error() -> None:
    with pytest.raises(SimulationError, match="not open"):
        _broker().close(D1)


def test_close_of_another_day_is_an_error() -> None:
    broker = _broker()
    broker.open(D1)
    with pytest.raises(SimulationError, match="not open"):
        broker.close(D2)


def test_close_twice_is_an_error() -> None:
    broker = _broker()
    broker.open(D1)
    broker.close(D1)
    with pytest.raises(SimulationError, match="not open"):
        broker.close(D1)


def test_open_of_a_date_outside_the_index_is_an_error() -> None:
    with pytest.raises(SimulationError, match="not a trading day"):
        _broker().open(date(2024, 1, 6))


def test_open_out_of_order_is_an_error() -> None:
    broker = _broker()
    broker.open(D3)
    broker.close(D3)
    with pytest.raises(SimulationError, match="after the last day"):
        broker.open(D2)


def test_open_without_closing_the_previous_day_is_an_error() -> None:
    broker = _broker()
    broker.open(D1)
    with pytest.raises(SimulationError, match="not closed"):
        broker.open(D2)


def _flat_data(first: str = "2024-01-02", opens: float = 100.0):
    days = pd.bdate_range(first, periods=3)
    closes = pd.DataFrame({"AAA": [101.0, 102.0, 103.0]}, index=days)
    opens_frame = pd.DataFrame({"AAA": [opens, opens + 1, opens + 2]}, index=days)
    return make_frozen_data(
        start=days[0].date(),
        end=days[-1].date(),
        opens=opens_frame,
        closes=closes,
        splits=(),
        dividends=(),
    ), [d.date() for d in days]


def _flat_broker(costs: SimCosts = SimCosts(), first: str = "2024-01-02"):  # noqa: B008
    data, days = _flat_data(first)
    return SimulatedBroker(data, cash=10_000.0, costs=costs), days


@pytest.mark.parametrize(
    ("action", "held", "price", "shares", "cash"),
    [
        (OrderAction.BUY, 0, 100.05, 10, 10_000.0 - 1000.5),
        (OrderAction.SELL, 10, 99.95, 0, 10_000.0 + 999.5),
        (OrderAction.SELL_SHORT, 0, 99.95, -10, 10_000.0 + 999.5),
        (OrderAction.BUY_TO_COVER, -10, 100.05, 0, 10_000.0 - 1000.5),
    ],
)
def test_each_action_fills_at_the_open_with_adverse_slippage(
    action: OrderAction, held: int, price: float, shares: int, cash: float
) -> None:
    broker, (d1, *_) = _flat_broker()
    broker.shares = {"AAA": held} if held else {}
    broker.marks = {"AAA": 100.0} if held else {}
    broker.submit(_request("c1", "AAA", 10, action))
    broker.open(d1)
    assert broker.orders["c1"].state is OrderState.FILLED
    assert broker.orders["c1"].fill_price == pytest.approx(price)
    assert broker.orders["c1"].filled_on == d1
    assert broker.shares["AAA"] == shares
    assert broker.cash == pytest.approx(cash)
    (event,) = broker.events
    assert event.kind is SimEventKind.FILL
    assert event.day == d1
    assert event.shares == (10 if action in (OrderAction.BUY, OrderAction.BUY_TO_COVER) else -10)
    assert event.price == pytest.approx(price)
    assert event.cash == pytest.approx(cash - 10_000.0)
    assert event.cost == pytest.approx(0.5)
    assert event.client_order_id == "c1"


def test_zero_slippage_fills_at_the_open() -> None:
    broker, (d1, *_) = _flat_broker(SimCosts(0.0, 0.0))
    broker.submit(_request("c1", "AAA", 10))
    broker.open(d1)
    assert broker.orders["c1"].fill_price == 100.0
    assert broker.cash == 9_000.0
    assert broker.events[0].cost == 0.0


def test_an_order_submitted_on_the_day_fills_on_the_next_open() -> None:
    broker, (d1, d2, _) = _flat_broker()
    broker.open(d1)
    broker.submit(_request("c1"))
    assert broker.order_status("c1").state is OrderState.PENDING
    broker.close(d1)
    broker.open(d2)
    assert broker.orders["c1"].state is OrderState.FILLED
    assert broker.orders["c1"].fill_price == pytest.approx(101.0 * 1.0005)


def test_orders_fill_in_submission_order() -> None:
    broker, (d1, *_) = _flat_broker()
    broker.submit(_request("c1", "AAA", 10))
    broker.submit(_request("c2", "AAA", 4, OrderAction.SELL))
    broker.open(d1)
    assert broker.shares["AAA"] == 6
    assert [e.client_order_id for e in broker.events] == ["c1", "c2"]


@pytest.mark.parametrize(
    ("first", "hour"), [("2024-01-02", 14), ("2024-07-02", 13)], ids=["winter", "summer"]
)
def test_filled_status_is_at_the_new_york_open_in_utc(first: str, hour: int) -> None:
    broker, (d1, *_) = _flat_broker(first=first)
    broker.submit(_request("c1"))
    broker.open(d1)
    status = broker.order_status("c1")
    assert status.state is OrderState.FILLED
    assert status.filled_shares == Decimal(10)
    assert status.avg_price == pytest.approx(100.05)
    assert status.filled_at == datetime(d1.year, d1.month, d1.day, hour, 30, tzinfo=UTC)
    assert not is_open(
        status.state, acknowledged=True, deadline=datetime(2030, 1, 1, tzinfo=UTC), now=_NOW
    )


def test_to_fill_accepts_a_simulated_fill() -> None:
    broker, (d1, *_) = _flat_broker()
    request = _request("c1")
    broker.submit(request)
    broker.open(d1)
    recorded_at = datetime(2024, 1, 3, tzinfo=UTC)
    fill = to_fill(broker.order_status("c1"), request, cycle_as_of=d1, recorded_at=recorded_at)
    assert fill.status is FillStatus.FILLED
    assert fill.filled_shares == 10
    assert fill.avg_price == pytest.approx(100.05)
    assert fill.filled_at == datetime(2024, 1, 2, 14, 30, tzinfo=UTC)


@pytest.mark.parametrize(
    ("action", "held", "reason"),
    [
        (OrderAction.SELL, 5, "would cross zero"),
        (OrderAction.SELL, 0, "would cross zero"),
        (OrderAction.BUY_TO_COVER, -5, "would cross zero"),
        (OrderAction.BUY_TO_COVER, 0, "would cross zero"),
        (OrderAction.BUY, -5, "wrong side for the holding"),
        (OrderAction.SELL_SHORT, 5, "wrong side for the holding"),
    ],
)
def test_fills_that_cross_zero_or_take_the_wrong_side_are_rejected(
    action: OrderAction, held: int, reason: str
) -> None:
    broker, (d1, *_) = _flat_broker()
    broker.shares = {"AAA": held} if held else {}
    broker.marks = {"AAA": 100.0} if held else {}
    broker.submit(_request("c1", "AAA", 10, action))
    broker.open(d1)
    assert broker.orders["c1"].state is OrderState.REJECTED
    assert broker.orders["c1"].fill_price is None
    assert broker.cash == 10_000.0
    assert broker.shares.get("AAA", 0) == held
    (event,) = broker.events
    assert (event.kind, event.reason, event.client_order_id) == (
        SimEventKind.REJECT,
        reason,
        "c1",
    )
    assert (event.shares, event.cash) == (0, 0.0)
    assert broker.order_status("c1").state is OrderState.REJECTED


def test_account_between_open_and_close_prices_a_new_position_at_the_open() -> None:
    broker, (d1, *_) = _flat_broker()
    broker.submit(_request("c1", "AAA", 10))
    broker.open(d1)
    assert broker.marks == {"AAA": 100.0}
    assert broker.account().equity == pytest.approx(10_000.0 - 0.5)
    broker.submit(_request("c2", "AAA", 10, OrderAction.SELL_SHORT))
    assert broker.account().equity == pytest.approx(10_000.0 - 0.5)


def _corp_data(splits=(), dividends=()):
    """Three days of AAA; stored (split-adjusted) opens 50/50.5/51 and closes 50.5/51/51.5."""
    days = pd.bdate_range("2024-01-02", periods=3)
    return make_frozen_data(
        start=days[0].date(),
        end=days[-1].date(),
        opens=pd.DataFrame({"AAA": [50.0, 50.5, 51.0]}, index=days),
        closes=pd.DataFrame({"AAA": [50.5, 51.0, 51.5]}, index=days),
        splits=splits,
        dividends=dividends,
    ), [d.date() for d in days]


def _held_on_day_two(
    held: int, splits=(), dividends=(), mark: float | None = 101.0
) -> tuple[SimulatedBroker, list[date]]:
    """A broker holding ``held`` AAA with 1000 cash, closed on day 1 and opened on day 2."""
    data, days = _corp_data(splits, dividends)
    broker = SimulatedBroker(data, cash=1_000.0, costs=SimCosts(0.0, 0.0))
    broker.open(days[0])
    broker.close(days[0])
    broker.shares = {"AAA": held}
    if mark is not None:
        broker.marks = {"AAA": mark}
    return broker, days


def _split(ratio: Fraction, day_index: int = 1) -> Split:
    return Split("AAA", _corp_data()[1][day_index], ratio)


def test_a_two_for_one_split_doubles_a_long_and_keeps_equity() -> None:
    broker, days = _held_on_day_two(10, splits=(_split(Fraction(2)),))
    assert broker.account().equity == pytest.approx(1_000.0 + 10 * 101.0)
    broker.open(days[1])
    assert broker.shares == {"AAA": 20}
    assert broker.marks == {"AAA": pytest.approx(50.5)}
    assert broker.cash == 1_000.0
    assert broker.account().equity == pytest.approx(1_000.0 + 10 * 101.0)
    (event,) = broker.events
    assert event == SimEvent(
        day=days[1],
        kind=SimEventKind.SPLIT,
        ticker="AAA",
        shares=10,
        price=pytest.approx(50.5),
        cash=0.0,
    )


def test_a_two_for_one_split_doubles_a_short() -> None:
    broker, days = _held_on_day_two(-10, splits=(_split(Fraction(2)),))
    broker.open(days[1])
    assert broker.shares == {"AAA": -20}
    assert broker.events[0].shares == -10
    assert broker.events[0].cash == 0.0


def test_a_reverse_split_pays_cash_in_lieu_of_the_fraction() -> None:
    broker, days = _held_on_day_two(10, splits=(_split(Fraction(1, 3)),))
    broker.open(days[1])
    assert broker.shares == {"AAA": 3}
    assert broker.cash == pytest.approx(1_000.0 + 50.5 / 3)
    assert broker.marks == {"AAA": pytest.approx(303.0)}
    (event,) = broker.events
    assert (event.kind, event.shares) == (SimEventKind.SPLIT, -7)
    assert event.price == pytest.approx(50.5)
    assert event.cash == pytest.approx(50.5 / 3)


def test_a_reverse_split_on_a_short_pays_the_cash_in_lieu() -> None:
    broker, days = _held_on_day_two(-10, splits=(_split(Fraction(1, 3)),))
    broker.open(days[1])
    assert broker.shares == {"AAA": -3}
    assert broker.cash == pytest.approx(1_000.0 - 50.5 / 3)
    assert broker.events[0].shares == 7
    assert broker.events[0].cash == pytest.approx(-50.5 / 3)


def test_a_split_on_a_held_name_without_a_mark_leaves_marks_alone() -> None:
    broker, days = _held_on_day_two(10, splits=(_split(Fraction(2)),), mark=None)
    broker.open(days[1])
    assert broker.shares == {"AAA": 20}
    assert broker.marks == {}


def test_a_pending_order_on_a_split_ex_date_is_rejected() -> None:
    broker, days = _held_on_day_two(10, splits=(_split(Fraction(2)),))
    broker.submit(_request("c1", "AAA", 5, OrderAction.SELL))
    broker.open(days[1])
    assert broker.orders["c1"].state is OrderState.REJECTED
    assert broker.shares == {"AAA": 20}
    reject = broker.events[0]
    assert (reject.kind, reject.reason, reject.client_order_id) == (
        SimEventKind.REJECT,
        "split on fill day",
        "c1",
    )
    assert [e.kind for e in broker.events] == [SimEventKind.REJECT, SimEventKind.SPLIT]


def test_a_dividend_credits_a_long_and_debits_a_short() -> None:
    div = (Dividend("AAA", _corp_data()[1][1], 0.25),)
    long_broker, days = _held_on_day_two(10, dividends=div)
    long_broker.open(days[1])
    assert long_broker.cash == pytest.approx(1_002.5)
    assert long_broker.events == (
        SimEvent(
            day=days[1],
            kind=SimEventKind.DIVIDEND,
            ticker="AAA",
            shares=10,
            price=0.25,
            cash=pytest.approx(2.5),
        ),
    )
    short_broker, _ = _held_on_day_two(-10, dividends=div)
    short_broker.open(days[1])
    assert short_broker.cash == pytest.approx(997.5)
    assert short_broker.events[0].shares == -10
    assert short_broker.events[0].cash == pytest.approx(-2.5)


def test_a_dividend_before_a_later_split_is_unwound_to_the_raw_amount() -> None:
    days = _corp_data()[1]
    broker, _ = _held_on_day_two(
        10,
        splits=(Split("AAA", days[2], Fraction(2)),),
        dividends=(Dividend("AAA", days[1], 0.25),),
    )
    broker.open(days[1])
    assert broker.cash == pytest.approx(1_005.0)
    assert broker.events[0].price == pytest.approx(0.5)
    assert broker.events[0].cash == pytest.approx(5.0)


def test_a_split_unwinds_the_dividend_on_the_same_day_only_after_it() -> None:
    days = _corp_data()[1]
    broker, _ = _held_on_day_two(
        10,
        splits=(_split(Fraction(2)),),
        dividends=(Dividend("AAA", days[1], 0.25),),
    )
    broker.open(days[1])
    assert broker.shares == {"AAA": 20}
    assert broker.cash == pytest.approx(1_000.0 + 20 * 0.25)
    assert [e.kind for e in broker.events] == [SimEventKind.SPLIT, SimEventKind.DIVIDEND]


@pytest.mark.parametrize("shares", [{}, {"AAA": 0}])
def test_no_corporate_action_event_fires_for_a_name_that_is_not_held(
    shares: dict[str, int],
) -> None:
    days = _corp_data()[1]
    broker, _ = _held_on_day_two(
        0,
        splits=(_split(Fraction(2)),),
        dividends=(Dividend("AAA", days[1], 0.25),),
    )
    broker.shares = shares
    broker.open(days[1])
    assert broker.events == ()
    assert broker.cash == 1_000.0


NAN = float("nan")


def _gap_data(splits=(), aaa_closes=None, aaa_opens=None, bbb_ends: bool = False):
    """Five days. AAA closes end on day 3 by default; BBB trades throughout unless ``bbb_ends``."""
    bbb = [50.0, 49.0, 48.0, NAN, NAN] if bbb_ends else [50.0, 49.0, 48.0, 47.0, 46.0]
    closes = pd.DataFrame(
        {"AAA": aaa_closes or [10.0, 11.0, 12.0, NAN, NAN], "BBB": bbb}, index=DAYS
    )
    opens = pd.DataFrame(
        {"AAA": aaa_opens or [9.5, 10.5, 11.5, NAN, NAN], "BBB": closes["BBB"] - 0.5},
        index=DAYS,
    )
    return make_frozen_data(
        start=D1, end=D5, opens=opens, closes=closes, splits=splits, dividends=()
    )


def _gap_broker(data, shares: dict[str, int], through: int = 3) -> SimulatedBroker:
    """1000 cash, days 1..``through`` driven and closed, ``shares`` held, no slippage."""
    broker = SimulatedBroker(data, cash=1_000.0, costs=SimCosts(0.0, 0.0))
    for day in (D1, D2, D3, D4, D5)[:through]:
        broker.open(day)
        broker.close(day)
    broker.shares = dict(shares)
    broker.marks = dict.fromkeys(shares, 1.0)
    return broker


@pytest.mark.parametrize(("held", "cash"), [(10, 1_120.0), (-10, 880.0)])
def test_a_name_past_its_last_close_is_closed_out_at_that_close(held: int, cash: float) -> None:
    broker = _gap_broker(_gap_data(), {"AAA": held})
    broker.open(D4)
    assert broker.cash == cash
    assert broker.positions() == ()
    assert broker.shares == {}
    assert broker.marks == {}
    assert broker.delisted == ("AAA",)
    assert broker.events == (
        SimEvent(
            day=D4,
            kind=SimEventKind.DELIST,
            ticker="AAA",
            shares=-held,
            price=12.0,
            cash=cash - 1_000.0,
        ),
    )


def test_a_name_on_its_last_trading_day_is_not_delisted() -> None:
    broker = _gap_broker(_gap_data(), {"AAA": 10}, through=2)
    broker.open(D3)
    assert broker.shares == {"AAA": 10}
    assert broker.delisted == ()
    assert broker.events == ()


def test_delisting_runs_once_and_in_delisting_order() -> None:
    data = _gap_data(bbb_ends=True)
    broker = _gap_broker(data, {"BBB": -2, "AAA": 10, "SPY": 0})
    broker.open(D4)
    broker.close(D4)
    broker.open(D5)
    assert broker.delisted == ("AAA", "BBB")
    assert broker.cash == 1_000.0 + 10 * 12.0 - 2 * 48.0
    assert [e.ticker for e in broker.events] == ["AAA", "BBB"]


def test_the_close_out_price_is_the_raw_close_unwound_for_a_later_split() -> None:
    splits = (Split("AAA", D4, Fraction(2)),)
    broker = _gap_broker(_gap_data(splits=splits), {"AAA": 10})
    broker.open(D4)
    assert broker.cash == 1_000.0 + 10 * 24.0
    assert [e.kind for e in broker.events] == [SimEventKind.DELIST]
    assert broker.events[0].price == 24.0


@pytest.mark.parametrize("held", [0, 10])
def test_a_pending_order_on_a_delisted_name_is_rejected_as_not_trading(held: int) -> None:
    broker = _gap_broker(_gap_data(), {"AAA": held} if held else {}, through=2)
    broker.open(D3)
    broker.submit(_request("c1", "AAA", 5, OrderAction.BUY))
    broker.close(D3)
    broker.open(D4)
    assert broker.orders["c1"].state is OrderState.REJECTED
    assert broker.order_status("c1").filled_at is None
    reject = broker.events[-1]
    assert (reject.kind, reject.reason, reject.client_order_id) == (
        SimEventKind.REJECT,
        "not trading",
        "c1",
    )
    assert broker.delisted == (("AAA",) if held else ())


def test_a_pending_order_on_a_split_day_of_a_delisted_name_is_not_trading() -> None:
    splits = (Split("AAA", D4, Fraction(2)),)
    broker = _gap_broker(_gap_data(splits=splits), {}, through=2)
    broker.open(D3)
    broker.submit(_request("c1", "AAA", 5, OrderAction.BUY))
    broker.close(D3)
    broker.open(D4)
    assert [(e.kind, e.reason) for e in broker.events] == [(SimEventKind.REJECT, "not trading")]


def test_a_nan_open_while_still_trading_rejects_the_order_as_no_open() -> None:
    data = _gap_data(
        aaa_opens=[9.5, NAN, 11.5, 12.5, 13.5], aaa_closes=[10.0, 11.0, 12.0, 13.0, 14.0]
    )
    broker = _gap_broker(data, {}, through=1)
    broker.submit(_request("c1", "AAA", 5, OrderAction.BUY))
    broker.submit(_request("c2", "BBB", 5, OrderAction.BUY))
    broker.open(D2)
    assert broker.orders["c1"].state is OrderState.REJECTED
    assert broker.orders["c2"].state is OrderState.FILLED
    assert broker.shares == {"BBB": 5}
    assert broker.cash == 1_000.0 - 5 * 48.5
    assert [(e.kind, e.ticker, e.reason) for e in broker.events] == [
        (SimEventKind.REJECT, "AAA", "no open"),
        (SimEventKind.FILL, "BBB", None),
    ]
    assert broker.events[0].shares == 0
    assert broker.events[0].cash == 0.0


def test_a_name_that_never_traded_has_no_open() -> None:
    data = _gap_data(aaa_closes=[NAN] * 5, aaa_opens=[NAN] * 5)
    broker = _gap_broker(data, {}, through=1)
    broker.submit(_request("c1", "AAA", 5, OrderAction.BUY))
    broker.open(D2)
    assert broker.orders["c1"].state is OrderState.REJECTED
    assert broker.events[0].reason == "no open"
    assert broker.delisted == ()


def _nan_open_split_broker(ratio: Fraction, held: int, prior_close_nan: bool = False):
    days = pd.bdate_range("2024-01-02", periods=3)
    closes = [NAN, 51.0, 51.5] if prior_close_nan else [50.5, 51.0, 51.5]
    data = make_frozen_data(
        start=days[0].date(),
        end=days[-1].date(),
        opens=pd.DataFrame({"AAA": [50.0, NAN, 51.0]}, index=days),
        closes=pd.DataFrame({"AAA": closes}, index=days),
        splits=(Split("AAA", days[1].date(), ratio),),
        dividends=(),
    )
    broker = SimulatedBroker(data, cash=1_000.0, costs=SimCosts(0.0, 0.0))
    broker.open(days[0].date())
    broker.close(days[0].date())
    broker.shares = {"AAA": held}
    return broker, [d.date() for d in days]


@pytest.mark.parametrize(("held", "sign"), [(10, 1), (-10, -1)])
def test_a_nan_open_on_a_split_day_pays_the_cash_in_lieu_at_the_prior_close(
    held: int, sign: int
) -> None:
    broker, days = _nan_open_split_broker(Fraction(1, 3), held)
    broker.open(days[1])
    # Prior stored close 50.5 is already on the post-split basis, so the third of a share
    # is paid at 50.5.
    assert broker.shares == {"AAA": 3 * sign}
    assert broker.cash == pytest.approx(1_000.0 + sign * 50.5 / 3)
    assert broker.cash == broker.cash  # not NaN
    (event,) = broker.events
    assert event.price == pytest.approx(50.5)
    assert event.cash == pytest.approx(sign * 50.5 / 3)


def test_a_nan_open_on_a_whole_share_split_still_moves_the_shares() -> None:
    broker, days = _nan_open_split_broker(Fraction(2), 10)
    broker.open(days[1])
    assert broker.shares == {"AAA": 20}
    assert broker.events[0].price == pytest.approx(50.5)


def test_a_nan_open_on_a_split_day_with_no_prior_close_is_a_simulation_error() -> None:
    broker, days = _nan_open_split_broker(Fraction(1, 3), 10, prior_close_nan=True)
    with pytest.raises(SimulationError, match="no prior close"):
        broker.open(days[1])


def test_close_marks_held_names_at_the_raw_close() -> None:
    broker, (d1, *_) = _flat_broker(SimCosts(0.0, 0.0))
    broker.submit(_request("c1", "AAA", 10))
    broker.open(d1)
    broker.close(d1)
    assert broker.marks == {"AAA": 101.0}
    assert broker.cash == 9_000.0
    assert broker.account().equity == 9_000.0 + 10 * 101.0
    assert broker.events[-1].kind is SimEventKind.FILL


def test_close_unwinds_the_close_for_a_later_split() -> None:
    data, days = _corp_data(splits=(_split(Fraction(2)),))
    broker = SimulatedBroker(data, cash=1_000.0, costs=SimCosts(0.0, 0.0))
    broker.shares = {"AAA": 10}
    broker.marks = {"AAA": 1.0}
    broker.open(days[0])
    broker.close(days[0])
    assert broker.marks == {"AAA": pytest.approx(101.0)}


def _borrow_broker(rate: float = 0.005) -> tuple[SimulatedBroker, list[date]]:
    """Thu 2024-01-04, Fri 05, Mon 08, Tue 09; AAA closes 52/51/50/49, opens 0.5 below."""
    days = pd.bdate_range("2024-01-04", periods=4)
    closes = pd.DataFrame({"AAA": [52.0, 51.0, 50.0, 49.0]}, index=days)
    data = make_frozen_data(
        start=days[0].date(),
        end=days[-1].date(),
        opens=closes - 0.5,
        closes=closes,
        splits=(),
        dividends=(),
    )
    return SimulatedBroker(data, cash=10_000.0, costs=SimCosts(0.0, rate)), [d.date() for d in days]


def _short_through_monday(broker: SimulatedBroker, days: list[date]) -> None:
    thu, fri, mon, _ = days
    broker.open(thu)
    broker.submit(_request("c1", "AAA", 100, OrderAction.SELL_SHORT))
    broker.close(thu)
    broker.open(fri)
    broker.close(fri)
    broker.open(mon)
    broker.close(mon)


def test_a_short_accrues_borrow_for_the_calendar_days_since_the_last_close() -> None:
    broker, days = _borrow_broker()
    _short_through_monday(broker, days)
    borrows = [e for e in broker.events if e.kind is SimEventKind.BORROW]
    assert len(borrows) == 2
    thursday_to_friday, friday_to_monday = borrows
    assert thursday_to_friday.cost == pytest.approx(100 * 51.0 * 0.005 * 1 / 365)
    expected = 100 * 50.0 * 0.005 * 3 / 365
    assert friday_to_monday == SimEvent(
        day=days[2],
        kind=SimEventKind.BORROW,
        ticker="AAA",
        shares=-100,
        price=50.0,
        cash=pytest.approx(-expected),
        cost=pytest.approx(expected),
    )
    start = 10_000.0 + 100 * 50.5
    assert broker.cash == pytest.approx(start - thursday_to_friday.cost - expected)
    assert broker.account().equity == pytest.approx(broker.cash - 100 * 50.0)


def test_the_first_close_accrues_one_day_of_borrow() -> None:
    broker, days = _borrow_broker()
    broker.shares = {"AAA": -100}
    broker.marks = {"AAA": 1.0}
    broker.open(days[2])
    broker.close(days[2])
    (event,) = broker.events
    assert event.cost == pytest.approx(100 * 50.0 * 0.005 / 365)


def test_a_long_pays_no_borrow() -> None:
    broker, days = _borrow_broker()
    broker.open(days[0])
    broker.submit(_request("c1", "AAA", 100))
    broker.close(days[0])
    broker.open(days[1])
    broker.close(days[1])
    assert all(e.kind is not SimEventKind.BORROW for e in broker.events)
    assert broker.cash == 10_000.0 - 100 * 50.5


def test_a_nan_close_keeps_the_previous_mark() -> None:
    data = _gap_data(
        aaa_opens=[9.5, 10.5, 11.5, 12.5, 13.5], aaa_closes=[10.0, NAN, 12.0, 13.0, 14.0]
    )
    broker = SimulatedBroker(data, cash=1_000.0, costs=SimCosts(0.0, 0.0))
    broker.submit(_request("c1", "AAA", 10))
    broker.open(D1)
    broker.close(D1)
    assert broker.marks == {"AAA": 10.0}
    broker.open(D2)
    broker.close(D2)
    assert broker.marks == {"AAA": 10.0}
    assert broker.account().equity == 1_000.0 - 10 * 9.5 + 10 * 10.0


def test_a_nan_close_with_no_mark_is_a_simulation_error() -> None:
    data = _gap_data(
        aaa_opens=[9.5, 10.5, 11.5, 12.5, 13.5], aaa_closes=[10.0, NAN, 12.0, 13.0, 14.0]
    )
    broker = SimulatedBroker(data, cash=1_000.0, costs=SimCosts(0.0, 0.0))
    broker.shares = {"AAA": 10}
    broker.open(D1)
    broker.close(D1)
    del broker.marks["AAA"]
    broker.open(D2)
    with pytest.raises(SimulationError, match="no mark"):
        broker.close(D2)


def test_the_close_snapshot_fits_the_account_service() -> None:
    broker, (d1, *_) = _flat_broker(SimCosts(0.0, 0.0))
    broker.submit(_request("c1", "AAA", 10))
    broker.open(d1)
    broker.close(d1)
    store = InMemorySnapshotStore()
    observed = datetime(d1.year, d1.month, d1.day, 21, 30, tzinfo=UTC)
    snapshot = record_snapshot(broker, store, as_of=d1, now=lambda: observed)
    assert store.saved == [snapshot]
    assert snapshot.as_of == d1
    assert snapshot.observed_at == observed
    assert snapshot.equity == 9_000.0 + 10 * 101.0
    assert snapshot.holdings == (Holding("AAA", 10),)


def _script(broker: SimulatedBroker, days: list[date]) -> None:
    thu, fri, mon, tue = days
    broker.open(thu)
    broker.submit(_request("c1", "AAA", 100, OrderAction.SELL_SHORT))
    broker.close(thu)
    broker.open(fri)
    broker.submit(_request("c2", "AAA", 40, OrderAction.BUY_TO_COVER))
    broker.close(fri)
    broker.open(mon)
    broker.close(mon)
    broker.open(tue)
    broker.close(tue)


def test_two_brokers_driven_by_one_script_agree() -> None:
    (first, days), (second, _) = _borrow_broker(), _borrow_broker()
    _script(first, days)
    _script(second, days)
    assert first.events == second.events
    assert first.account().equity == second.account().equity
    assert first.positions() == second.positions() == (Holding("AAA", -60),)


@pytest.mark.parametrize("cash", [0.0, -1.0, float("nan"), float("inf")])
def test_constructor_rejects_bad_cash(cash: float) -> None:
    with pytest.raises(SimulationError, match="cash must be finite and > 0"):
        SimulatedBroker(_data(), cash=cash)
