"""Ledger and settlement types: broker order states, fills, lots and the settlement report.

Fills are the only stored ledger records; lots, marks and breaks are derived on every settle.
None of these types validate themselves.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from enum import StrEnum

from hedge_fund.domain.cycle import CycleReport
from hedge_fund.domain.execution import OrderRequest
from hedge_fund.domain.orders import Holding, OrderAction


class FillStatus(StrEnum):
    """Final state of one submitted order, as recorded.

    PARTIAL_TERMINAL: some shares filled, then canceled or expired. CANCELED, REJECTED,
    EXPIRED: zero shares filled. NOT_SUBMITTED: never acknowledged by the broker and not
    found there after the fill deadline.
    """

    FILLED = "filled"
    PARTIAL_TERMINAL = "partial-terminal"
    CANCELED = "canceled"
    REJECTED = "rejected"
    EXPIRED = "expired"
    NOT_SUBMITTED = "not-submitted"


@dataclass(frozen=True)
class Fill:
    """One final, write-once order record.

    ``ticker`` is the request ticker (dash form) and ``action`` the planned action; the
    signed share delta is ``+filled_shares`` for BUY / BUY_TO_COVER and ``-filled_shares``
    for SELL / SELL_SHORT. ``avg_price`` and ``filled_at`` are None when ``filled_shares``
    is 0; ``broker_order_id`` is None for NOT_SUBMITTED. Instants are tz-aware UTC.
    """

    client_order_id: str
    broker_order_id: str | None
    cycle_as_of: date
    ticker: str
    action: OrderAction
    requested_shares: int
    filled_shares: int
    avg_price: float | None
    filled_at: datetime | None
    status: FillStatus
    recorded_at: datetime


class OrderState(StrEnum):
    """What the broker port reports for one client_order_id.

    PENDING: not final yet. UNRECOGNISED: a status the adapter cannot map (the raw value is
    in ``OrderStatus.broker_state``).
    """

    FILLED = "filled"
    PARTIAL_TERMINAL = "partial-terminal"
    CANCELED = "canceled"
    REJECTED = "rejected"
    EXPIRED = "expired"
    PENDING = "pending"
    NOT_FOUND = "not-found"
    UNRECOGNISED = "unrecognised"


@dataclass(frozen=True)
class OrderStatus:
    """One broker lookup. ``ticker`` is the raw broker symbol and ``side`` ``"buy"`` /
    ``"sell"`` (both None for NOT_FOUND); ``filled_shares`` is as reported (the whole-share
    check is ``to_fill``'s); ``broker_state`` is the raw broker status.
    """

    client_order_id: str
    state: OrderState
    broker_order_id: str | None
    ticker: str | None
    side: str | None
    filled_shares: Decimal
    avg_price: float | None
    filled_at: datetime | None
    broker_state: str | None


@dataclass(frozen=True)
class SubmittedOrder:
    """An order a cycle may have put live; ``started_at`` is of the earliest report carrying
    it and ``acknowledged`` is True if any report saw the broker accept or duplicate it.
    """

    request: OrderRequest
    cycle_as_of: date
    started_at: datetime
    acknowledged: bool


@dataclass(frozen=True)
class OpeningEntry:
    """A holding before the system's first fill: signed non-zero ``shares`` at ``price``."""

    ticker: str
    shares: int
    price: float
    on: date


@dataclass(frozen=True)
class Lot:
    """An open lot: ``shares`` signed (+ long, - short); ``source`` is the opening
    client_order_id, or ``"opening"`` (then ``entered_at`` is midnight UTC of its date).
    """

    ticker: str
    shares: int
    entry_price: float
    entered_at: datetime
    source: str


@dataclass(frozen=True)
class ClosedLot:
    """The signed portion ``shares`` of a lot closed by ``exit_client_order_id``;
    ``realised_pnl = (exit_price - entry_price) * shares``.
    """

    ticker: str
    shares: int
    entry_price: float
    entered_at: datetime
    source: str
    exit_price: float
    exited_at: datetime
    exit_client_order_id: str
    realised_pnl: float


@dataclass(frozen=True)
class MarkedLot:
    """An open lot at the close of ``close_date``; ``unrealised_pnl = (price - entry) * shares``."""

    lot: Lot
    price: float
    close_date: date
    unrealised_pnl: float


@dataclass(frozen=True)
class ReconciliationBreak:
    """A ticker whose summed lot shares differ from the broker's position."""

    ticker: str
    ledger_shares: int
    broker_shares: int


@dataclass(frozen=True)
class PendingOrder:
    """An order not final yet: ``"open at broker: <state>"`` or ``"not found before <ISO>"``."""

    client_order_id: str
    ticker: str
    reason: str


@dataclass(frozen=True)
class UnsettledOrder:
    """An order settle could not record; ``ticker`` is ``""`` when no planned order matches."""

    client_order_id: str
    ticker: str
    reason: str


@dataclass(frozen=True)
class ReportLoad:
    """Every decodable cycle report (by ``started_at``) and the run files that did not
    decode: file name -> ``"<ExceptionType>: <message>"``.
    """

    reports: tuple[CycleReport, ...]
    unreadable: Mapping[str, str]


class SettleOutcome(StrEnum):
    """How one settle ended; ERROR and INTERRUPTED are partial reports of a run that raised.

    Precedence of a completed run: UNSETTLED > BREAK > PENDING > CLEAN.
    """

    CLEAN = "clean"
    PENDING = "pending"
    BREAK = "break"
    UNSETTLED = "unsettled"
    ERROR = "error"
    INTERRUPTED = "interrupted"


@dataclass(frozen=True)
class SettlementReport:
    """Audit record of one settle; fields not reached are empty or None, totals ``0.0``.

    ``started_at`` is tz-aware UTC. ``failed_stage`` and ``error`` (``"<Type>: <message>"``)
    are set only for ERROR and INTERRUPTED. ``new_fills`` are the fills this run recorded.
    ``broker_positions`` holds raw broker symbols. Both totals exclude tickers with a break,
    a pending order, an unsettled order or an unmarked lot.
    """

    as_of: date
    started_at: datetime
    outcome: SettleOutcome
    failed_stage: str | None
    error: str | None
    new_fills: tuple[Fill, ...]
    pending: tuple[PendingOrder, ...]
    unreadable_reports: Mapping[str, str]
    unsettled: tuple[UnsettledOrder, ...]
    broker_positions: tuple[Holding, ...] | None
    open_lots: tuple[MarkedLot, ...]
    unmarked: tuple[Lot, ...]
    closed_lots: tuple[ClosedLot, ...]
    breaks: tuple[ReconciliationBreak, ...]
    realised_total: float
    unrealised_total: float
