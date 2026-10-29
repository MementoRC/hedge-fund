"""In-memory providers and stores, vendor-shaped frames, a scripted researcher, broker and gate."""

import threading
from collections.abc import Mapping, Sequence
from datetime import UTC, date
from decimal import Decimal
from typing import Any

import pandas as pd

from hedge_fund.domain.account import AccountSnapshot
from hedge_fund.domain.attribution import AttributionReport
from hedge_fund.domain.cycle import CycleReport
from hedge_fund.domain.execution import Account, OrderRequest, SubmitResult, SubmitStatus
from hedge_fund.domain.ledger import (
    Fill,
    OpeningEntry,
    OrderState,
    OrderStatus,
    ReportLoad,
    SettlementReport,
)
from hedge_fund.domain.orders import Holding
from hedge_fund.domain.research import Rating, ResearchResult, RunRecord


class FakeProvider:
    """DataProvider serving a canned closes frame and sector map, recording every call.

    Tests may replace `frame` between calls to simulate vendor restatements.
    """

    def __init__(self, frame: pd.DataFrame, sectors: Mapping[str, str | None]) -> None:
        self.frame = frame
        self.sector_map = dict(sectors)
        self.closes_calls: list[tuple[tuple[str, ...], date, date]] = []
        self.sectors_calls: list[tuple[str, ...]] = []

    def closes(self, tickers: Sequence[str], start: date, end: date) -> pd.DataFrame:
        requested = tuple(tickers)
        self.closes_calls.append((requested, start, end))
        window = self.frame.loc[pd.Timestamp(start) : pd.Timestamp(end)]
        return window.reindex(columns=list(requested))

    def sectors(self, tickers: Sequence[str]) -> dict[str, str | None]:
        requested = tuple(tickers)
        self.sectors_calls.append(requested)
        return {ticker: self.sector_map.get(ticker) for ticker in requested}


class FakeFundamentalsProvider:
    """FundamentalsProvider serving a canned facts frame, recording every call.

    Tests may replace `frame` between calls to simulate new filings.
    """

    def __init__(self, frame: pd.DataFrame) -> None:
        self.frame = frame
        self.calls: list[tuple[str, ...]] = []

    def facts(self, tickers: Sequence[str]) -> pd.DataFrame:
        requested = tuple(tickers)
        self.calls.append(requested)
        return self.frame.loc[self.frame["ticker"].isin(requested)].reset_index(drop=True)


class RecordingDownload:
    """Stand-in for yf.download: returns a fixed frame (or raises) and records each call."""

    def __init__(self, result: pd.DataFrame | None = None, error: Exception | None = None) -> None:
        self.result = result
        self.error = error
        self.calls: list[tuple[list[str], dict[str, Any]]] = []

    def __call__(self, tickers: list[str], **kwargs: Any) -> pd.DataFrame | None:
        self.calls.append((list(tickers), kwargs))
        if self.error is not None:
            raise self.error
        return self.result


def yf_multi(closes: pd.DataFrame) -> pd.DataFrame:
    """yf.download shape for several tickers: (Price, Ticker) MultiIndex columns."""
    return pd.concat({"Close": closes, "Open": closes * 0.99}, axis=1, names=["Price", "Ticker"])


def yf_single(closes: pd.Series) -> pd.DataFrame:
    """Flat frame (one column per price field) as yfinance returns with multi_level_index=False.

    Not yfinance's default, which is a (Price, Ticker) MultiIndex even for a single ticker.
    """
    return pd.DataFrame({"Open": closes * 0.99, "Close": closes, "Volume": 1000.0})


def assert_closes_equal(actual: pd.DataFrame, expected: pd.DataFrame) -> None:
    """Frame equality ignoring index frequency and datetime resolution (parquet may change both)."""
    pd.testing.assert_frame_equal(
        actual.set_axis(pd.DatetimeIndex(actual.index).as_unit("ns")),
        expected.set_axis(pd.DatetimeIndex(expected.index).as_unit("ns")),
        check_freq=False,
    )


class ScriptedResearcher:
    """Researcher returning a scripted Rating (or raising) per (ticker, run_index).

    Unscripted runs return `default`. Safe to call from several threads; every call is
    recorded in `calls`.
    """

    def __init__(
        self,
        script: Mapping[tuple[str, int], Rating | BaseException] | None = None,
        default: Rating = Rating.HOLD,
    ) -> None:
        self.script = dict(script or {})
        self.default = default
        self.calls: list[tuple[str, date, int]] = []
        self._lock = threading.Lock()

    def run(self, ticker: str, as_of: date, run_index: int) -> RunRecord:
        with self._lock:
            self.calls.append((ticker, as_of, run_index))
        outcome = self.script.get((ticker, run_index), self.default)
        if isinstance(outcome, BaseException):
            raise outcome
        return RunRecord(
            ticker=ticker,
            as_of=as_of,
            run_index=run_index,
            rating=outcome,
            decision=f"{ticker}:{outcome.value}",
            reports={},
            seconds=0.5,
            error=None,
        )


SCRIPTED_REJECTION = "scripted rejection"
BROKER_STATUSES = frozenset({SubmitStatus.ACCEPTED, SubmitStatus.DUPLICATE, SubmitStatus.REJECTED})


def not_found(client_order_id: str) -> OrderStatus:
    """The NOT_FOUND status a broker reports for an id it has never seen."""
    return OrderStatus(
        client_order_id=client_order_id,
        state=OrderState.NOT_FOUND,
        broker_order_id=None,
        ticker=None,
        side=None,
        filled_shares=Decimal(0),
        avg_price=None,
        filled_at=None,
        broker_state=None,
    )


class ScriptedBroker:
    """Broker returning a scripted status (or raising) per client_order_id; records submits.

    Unscripted ids are ACCEPTED. ACCEPTED and DUPLICATE results carry broker id
    ``b-<client_order_id>``; REJECTED results carry reason SCRIPTED_REJECTION. Every
    submitted id is appended to ``submitted``, including ids whose script raises.
    ``account()`` and ``positions()`` raise ``account_error`` / ``positions_error`` when
    set; like ``equity`` and ``holdings`` they are plain attributes a test may change.
    ``order_status`` returns the status scripted in ``orders`` (or raises it), else
    ``not_found(id)``; every looked-up id is appended to ``looked_up``.
    """

    def __init__(
        self,
        script: Mapping[str, SubmitStatus | BaseException] | None = None,
        *,
        equity: float = 1_000_000.0,
        holdings: tuple[Holding, ...] = (),
        account_error: Exception | None = None,
        positions_error: Exception | None = None,
        orders: Mapping[str, OrderStatus | BaseException] | None = None,
    ) -> None:
        self.script = dict(script or {})
        for outcome in self.script.values():
            if isinstance(outcome, SubmitStatus) and outcome not in BROKER_STATUSES:
                raise ValueError(
                    f"ScriptedBroker cannot return {outcome.name}: a real broker only "
                    "returns ACCEPTED, DUPLICATE or REJECTED (raise BrokerError for ERROR)"
                )
        self.equity = equity
        self.holdings = holdings
        self.account_error = account_error
        self.positions_error = positions_error
        self.submitted: list[str] = []
        self.orders = dict(orders or {})
        self.looked_up: list[str] = []

    def account(self) -> Account:
        if self.account_error is not None:
            raise self.account_error
        return Account(equity=self.equity)

    def positions(self) -> tuple[Holding, ...]:
        if self.positions_error is not None:
            raise self.positions_error
        return self.holdings

    def submit(self, request: OrderRequest) -> SubmitResult:
        client_order_id = request.client_order_id
        self.submitted.append(client_order_id)
        outcome = self.script.get(client_order_id, SubmitStatus.ACCEPTED)
        if isinstance(outcome, BaseException):
            raise outcome
        if outcome is SubmitStatus.REJECTED:
            return SubmitResult(
                request=request, status=outcome, broker_order_id=None, reason=SCRIPTED_REJECTION
            )
        return SubmitResult(
            request=request,
            status=outcome,
            broker_order_id=f"b-{client_order_id}",
            reason=None,
        )

    def order_status(self, client_order_id: str) -> OrderStatus:
        self.looked_up.append(client_order_id)
        outcome = self.orders.get(client_order_id)
        if outcome is None:
            return not_found(client_order_id)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class ScriptedApprovalGate:
    """ApprovalGate returning scripted answers in order; records every summary it is shown.

    Running out of answers raises AssertionError: the code under test asked more often than
    the test expected.
    """

    def __init__(self, answers: Sequence[bool]) -> None:
        self.answers = list(answers)
        self.summaries: list[str] = []

    def confirm(self, summary: str) -> bool:
        self.summaries.append(summary)
        if not self.answers:
            raise AssertionError("ScriptedApprovalGate is out of answers")
        return self.answers.pop(0)


class InMemorySnapshotStore:
    """SnapshotStore keeping snapshots in ``saved``, in save order; ``save`` appends.

    ``snapshots()`` raises ``fail_on_read`` when it is set.
    """

    def __init__(
        self,
        snapshots: Sequence[AccountSnapshot] = (),
        *,
        fail_on_read: Exception | None = None,
    ) -> None:
        self.saved = list(snapshots)
        self.fail_on_read = fail_on_read

    def save(self, snapshot: AccountSnapshot) -> None:
        self.saved.append(snapshot)

    def snapshots(self) -> tuple[AccountSnapshot, ...]:
        if self.fail_on_read is not None:
            raise self.fail_on_read
        return tuple(self.saved)


class InMemoryResearchStore:
    """ResearchStore keeping results in ``saved``, in save order; ``save`` appends.

    ``load`` returns the last saved result for the date (None if there is none) and
    records the date in ``loaded``.
    """

    def __init__(self, results: Sequence[ResearchResult] = ()) -> None:
        self.saved = list(results)
        self.loaded: list[date] = []

    def save(self, result: ResearchResult) -> None:
        self.saved.append(result)

    def load(self, as_of: date) -> ResearchResult | None:
        self.loaded.append(as_of)
        matches = [result for result in self.saved if result.as_of == as_of]
        return matches[-1] if matches else None


class InMemoryReportStore:
    """CycleReportStore keeping reports in ``saved``, in save order.

    ``save`` raises ``fail`` instead, keeping nothing, when it is set. ``load_all`` returns
    the saved reports sorted by ``started_at`` and ``unreadable`` (a plain attribute).
    """

    def __init__(
        self, reports: Sequence[CycleReport] = (), *, fail: Exception | None = None
    ) -> None:
        self.saved = list(reports)
        self.fail = fail
        self.unreadable: dict[str, str] = {}

    def save(self, report: CycleReport) -> None:
        if self.fail is not None:
            raise self.fail
        self.saved.append(report)

    def load_all(self) -> ReportLoad:
        reports = tuple(sorted(self.saved, key=lambda report: report.started_at))
        return ReportLoad(reports=reports, unreadable=dict(self.unreadable))


class InMemoryFillStore:
    """FillStore keeping fills in ``saved``, in save order.

    ``save`` raises FileExistsError for an id already saved, keeping the first; ``load_all``
    raises ``load_error`` when it is set.
    """

    def __init__(self, fills: Sequence[Fill] = (), *, load_error: Exception | None = None) -> None:
        self.saved = list(fills)
        self.load_error = load_error

    def save(self, fill: Fill) -> None:
        if any(saved.client_order_id == fill.client_order_id for saved in self.saved):
            raise FileExistsError(f"fill {fill.client_order_id!r} is already recorded")
        self.saved.append(fill)

    def load_all(self) -> tuple[Fill, ...]:
        if self.load_error is not None:
            raise self.load_error
        return tuple(self.saved)


class StaticOpeningBalance:
    """OpeningBalance returning ``entries``, or raising ``error`` when it is set."""

    def __init__(
        self, entries: Sequence[OpeningEntry] = (), *, error: Exception | None = None
    ) -> None:
        self.entries = tuple(entries)
        self.error = error

    def load(self) -> tuple[OpeningEntry, ...]:
        if self.error is not None:
            raise self.error
        return self.entries


class InMemorySettlementStore:
    """SettlementReportStore keeping reports in ``saved``; ``save`` raises ``fail`` when set.

    Like JsonSettlementReportStore, whose file is named ``<as_of>_<HHMMSS>.json`` (UTC),
    ``save`` raises FileExistsError for a report whose ``as_of`` and ``started_at`` second
    were already saved.
    """

    def __init__(self, *, fail: Exception | None = None) -> None:
        self.saved: list[SettlementReport] = []
        self.fail = fail

    def save(self, report: SettlementReport) -> None:
        if self.fail is not None:
            raise self.fail
        name = self._name(report)
        if any(self._name(saved) == name for saved in self.saved):
            raise FileExistsError(f"{name}: a report for a settle started in this second exists")
        self.saved.append(report)

    @staticmethod
    def _name(report: SettlementReport) -> str:
        stamp = report.started_at.astimezone(UTC).strftime("%H%M%S")
        return f"{report.as_of.isoformat()}_{stamp}"


class InMemoryAttributionStore:
    """AttributionReportStore keeping reports in ``saved``; ``save`` raises ``fail`` when set.

    Like JsonAttributionReportStore, whose file is named ``<start>_<end>_<HHMMSS>.json``
    (UTC), ``save`` raises FileExistsError for a report whose window and ``started_at``
    second were already saved.
    """

    def __init__(self, *, fail: Exception | None = None) -> None:
        self.saved: list[AttributionReport] = []
        self.fail = fail

    def save(self, report: AttributionReport) -> None:
        if self.fail is not None:
            raise self.fail
        name = self._name(report)
        if any(self._name(saved) == name for saved in self.saved):
            raise FileExistsError(f"{name}: a report for this window and second exists")
        self.saved.append(report)

    @staticmethod
    def _name(report: AttributionReport) -> str:
        stamp = report.started_at.astimezone(UTC).strftime("%H%M%S")
        return f"{report.start}_{report.end}_{stamp}"
