"""Layer 7b attribution over fakes: stages, the window, breaks, exclusions and the report."""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from typing import Any, NoReturn

import pytest

import hedge_fund.services.attribute as attribute_service
from hedge_fund.domain.attribution import NO_PNL, AttributeOutcome, AttributionReport
from hedge_fund.domain.ledger import Fill, FillStatus, OpeningEntry, ReconciliationBreak
from hedge_fund.domain.orders import Holding, OrderAction
from hedge_fund.engine.portfolio.risk_model import RiskParams
from hedge_fund.ports.broker import BrokerError
from hedge_fund.ports.data import DataProviderError
from hedge_fund.services.attribute import AttributeDeps, AttributionSaveError, run_attribute
from hedge_fund.universe import Universe
from tests.factories import make_closes
from tests.fakes import (
    FakeProvider,
    InMemoryAttributionStore,
    InMemoryFillStore,
    ScriptedBroker,
    StaticOpeningBalance,
)

NOW = datetime(2026, 9, 30, 22, 0, tzinfo=UTC)  # 18:00 in New York
FRI, MON, TUE, WED = date(2026, 9, 25), date(2026, 9, 28), date(2026, 9, 29), date(2026, 9, 30)
TICKERS = ("AAA", "BBB", "CCC", "DDD", "EEE", "FFF")
SECTORS: dict[str, str | None] = {
    "AAA": "Tech",
    "BBB": "Tech",
    "CCC": "Tech",
    "DDD": "Energy",
    "EEE": "Energy",
    "FFF": "Energy",
}
UNIVERSE = Universe(
    name="test",
    list_date=date(2026, 9, 1),
    tickers=TICKERS,
    sectors={ticker: str(sector) for ticker, sector in SECTORS.items()},
)
RISK = RiskParams(lookback=30, min_obs=20)
FRAME = make_closes([*TICKERS, "ZZZ", "SPY"], n_days=80)
REQUESTED = (*TICKERS, "ZZZ", "SPY")


def _fill(cid: str, ticker: str, action: OrderAction, shares: int, at: datetime) -> Fill:
    return Fill(
        client_order_id=cid,
        broker_order_id=f"b-{cid}",
        cycle_as_of=at.date(),
        ticker=ticker,
        action=action,
        requested_shares=shares,
        filled_shares=shares,
        avg_price=50.0,
        filled_at=at,
        status=FillStatus.FILLED,
        recorded_at=at,
    )


OPENING = (OpeningEntry("AAA", 10, 100.0, FRI),)
FILLS = (
    _fill("hf-z", "ZZZ", OrderAction.BUY, 5, datetime(2026, 9, 28, 13, 30, tzinfo=UTC)),
    _fill("hf-a", "AAA", OrderAction.SELL, 4, datetime(2026, 9, 29, 13, 30, tzinfo=UTC)),
)
HOLDINGS = (Holding("AAA", 6), Holding("ZZZ", 5))


@dataclass
class World:
    """The fakes one attribution runs over; every field may be replaced by a test."""

    broker: Any = field(default_factory=lambda: ScriptedBroker(holdings=HOLDINGS))
    prices: Any = field(default_factory=lambda: FakeProvider(FRAME, SECTORS))
    universe: Universe = UNIVERSE
    risk: RiskParams = RISK
    fills: Any = field(default_factory=lambda: InMemoryFillStore(FILLS))
    opening: Any = field(default_factory=lambda: StaticOpeningBalance(OPENING))
    report_store: InMemoryAttributionStore = field(default_factory=InMemoryAttributionStore)

    @property
    def deps(self) -> AttributeDeps:
        return AttributeDeps(
            broker=self.broker,
            prices=self.prices,
            universe=self.universe,
            risk=self.risk,
            fills=self.fills,
            opening=self.opening,
            report_store=self.report_store,
        )


def _attribute(
    world: World, *, start: date | None = None, end: date | None = None, now: datetime = NOW
) -> AttributionReport:
    return run_attribute(world.deps, start=start, end=end, now=lambda: now)


def _component_sum(report: AttributionReport) -> float:
    return math.fsum(
        part
        for row in report.days
        for part in (row.market, row.sector_pnl, row.residual, row.trading)
    )


# --- completed runs ----------------------------------------------------------------------


def test_a_clean_run_over_the_default_window() -> None:
    world = World()
    report = _attribute(world)
    assert report.outcome is AttributeOutcome.CLEAN
    assert (report.start, report.end, report.started_at) == (FRI, WED, NOW)
    assert (report.failed_stage, report.error) == (None, None)
    assert [(row.day, row.ticker) for row in report.days] == [
        (FRI, "AAA"),
        (MON, "AAA"),
        (MON, "ZZZ"),
        (TUE, "AAA"),
        (TUE, "ZZZ"),
        (WED, "AAA"),
        (WED, "ZZZ"),
    ]
    assert [row.prior_shares for row in report.days] == [0, 10, 0, 10, 5, 6, 5]
    assert [row.regressed for row in report.days] == [False, True, False, True, False, True, False]
    assert [row.sector for row in report.days] == [
        "Tech",
        "Tech",
        "Unknown",
        "Tech",
        "Unknown",
        "Tech",
        "Unknown",
    ]
    assert report.no_beta == {"ZZZ": (TUE, WED)}
    assert (report.unregressed_days, report.excluded, report.breaks) == ((), {}, ())
    assert list(report.by_ticker) == ["AAA", "ZZZ"]
    assert list(report.by_sector) == ["Tech", "Unknown"]
    assert report.total.total == pytest.approx(_component_sum(report))
    assert world.report_store.saved == [report]
    assert world.prices.closes_calls == [(REQUESTED, date(2026, 7, 12), WED)]
    assert world.prices.sectors_calls == [(*TICKERS, "ZZZ")]
    assert world.broker.submitted == []


def test_an_explicit_window() -> None:
    world = World()
    report = _attribute(world, start=TUE, end=TUE)
    assert (report.start, report.end) == (TUE, TUE)
    assert [(row.day, row.ticker, row.prior_shares) for row in report.days] == [
        (TUE, "AAA", 10),
        (TUE, "ZZZ", 5),
    ]
    assert world.prices.closes_calls == [(REQUESTED, date(2026, 7, 16), TUE)]


def test_a_position_that_differs_from_the_ledger_is_a_break_left_out_of_the_totals() -> None:
    world = World(broker=ScriptedBroker(holdings=(Holding("AAA", 7), Holding("ZZZ", 5))))
    report = _attribute(world)
    assert report.outcome is AttributeOutcome.BREAK
    assert report.breaks == (ReconciliationBreak(ticker="AAA", ledger_shares=6, broker_shares=7),)
    assert "AAA" in {row.ticker for row in report.days}  # listed, not totalled
    assert list(report.by_ticker) == ["ZZZ"]
    assert list(report.by_sector) == ["Unknown"]
    assert report.total == report.by_ticker["ZZZ"]
    assert world.report_store.saved == [report]


def test_breaks_compare_canonical_tickers_and_a_name_without_closes_is_excluded() -> None:
    opening = (*OPENING, OpeningEntry("BRK-B", 3, 400.0, FRI))
    world = World(
        broker=ScriptedBroker(holdings=(*HOLDINGS, Holding("BRK.B", 3))),
        opening=StaticOpeningBalance(opening),
    )
    report = _attribute(world)
    assert report.outcome is AttributeOutcome.CLEAN
    assert report.breaks == ()
    assert report.excluded == {"BRK-B": "no close on 2026-09-25"}  # the first reason is kept
    assert "BRK-B" not in report.by_ticker
    assert world.prices.closes_calls[0][0] == (*TICKERS, "BRK-B", "ZZZ", "SPY")


def test_a_held_name_outside_the_universe_is_never_regressed() -> None:
    world = World(prices=FakeProvider(FRAME, {**SECTORS, "ZZZ": "Tech"}))
    report = _attribute(world)
    zzz = [row for row in report.days if row.ticker == "ZZZ"]
    assert [(row.sector, row.regressed) for row in zzz] == [("Tech", False)] * 3
    assert report.no_beta == {"ZZZ": (TUE, WED)}


def test_days_without_a_usable_regression_are_unregressed() -> None:
    world = World(prices=FakeProvider(FRAME, {}))
    report = _attribute(world)
    assert report.unregressed_days == (MON, TUE, WED)  # 09-25: nothing held at the 09-24 close
    assert report.no_beta == {}
    assert not any(row.regressed for row in report.days)
    assert {row.sector for row in report.days} == {"Unknown"}
    assert all(row.market == row.sector_pnl == 0.0 for row in report.days)


def test_an_empty_ledger_gives_a_clean_one_day_report() -> None:
    world = World(
        broker=ScriptedBroker(),
        fills=InMemoryFillStore(),
        opening=StaticOpeningBalance(),
    )
    report = _attribute(world)
    assert report.outcome is AttributeOutcome.CLEAN
    assert (report.start, report.end) == (WED, WED)
    assert (report.days, report.by_ticker, report.by_sector) == ((), {}, {})
    assert report.total == NO_PNL
    assert world.prices.closes_calls == [((*TICKERS, "SPY"), date(2026, 7, 17), WED)]
    assert world.prices.sectors_calls == [TICKERS]
    assert world.report_store.saved == [report]


# --- failures ----------------------------------------------------------------------------


class Boom:
    """A port whose every method raises ``error``."""

    def __init__(self, error: BaseException) -> None:
        self.error = error

    def __getattr__(self, name: str) -> Callable[..., NoReturn]:
        def fail(*args: Any, **kwargs: Any) -> NoReturn:
            raise self.error

        return fail


@pytest.mark.parametrize(
    ("overrides", "window", "error_type", "message", "stage"),
    [
        pytest.param(
            {"fills": InMemoryFillStore(load_error=ValueError("malformed fill"))},
            {},
            ValueError,
            "malformed fill",
            "load",
            id="fills",
        ),
        pytest.param(
            {"opening": StaticOpeningBalance(error=ValueError("bad opening"))},
            {},
            ValueError,
            "bad opening",
            "load",
            id="opening",
        ),
        pytest.param(
            {"risk": RiskParams(lookback=5, min_obs=1)},
            {},
            ValueError,
            "min_obs must satisfy 2 <= min_obs <= lookback, got min_obs=1, lookback=5",
            "load",
            id="risk-params",
        ),
        pytest.param(
            {"opening": StaticOpeningBalance([OpeningEntry("AAA", 10, 100.0, MON)])},
            {},
            ValueError,
            "opening entries must be dated before 2026-09-28, the New York date of the first "
            "fill: AAA",
            "reconcile",
            id="lots",
        ),
        pytest.param(
            {"broker": ScriptedBroker(positions_error=BrokerError("down"))},
            {},
            BrokerError,
            "down",
            "reconcile",
            id="positions",
        ),
        pytest.param(
            {"prices": Boom(DataProviderError("vendor down"))},
            {},
            DataProviderError,
            "vendor down",
            "fetch",
            id="closes",
        ),
        pytest.param(
            {"prices": FakeProvider(FRAME.drop(columns="SPY"), SECTORS)},
            {},
            ValueError,
            "no SPY closes up to 2026-09-30",
            "fetch",
            id="no-spy",
        ),
        pytest.param(
            {},
            {"start": WED, "end": FRI},
            ValueError,
            "start 2026-09-30 is after end 2026-09-25",
            "attribute",
            id="start-after-end",
        ),
        pytest.param(
            {},
            {"start": date(2026, 10, 3), "end": date(2026, 10, 4)},
            ValueError,
            "no trading day between 2026-10-03 and 2026-10-04",
            "attribute",
            id="weekend-window",
        ),
        pytest.param(
            {"prices": FakeProvider(FRAME.loc["2026-09-25":], SECTORS)},
            {},
            ValueError,
            "no trading day before 2026-09-25 in the closes",
            "attribute",
            id="no-previous-close",
        ),
    ],
)
def test_a_failing_stage_saves_a_failed_report_and_re_raises(
    overrides: dict[str, Any],
    window: dict[str, date],
    error_type: type[Exception],
    message: str,
    stage: str,
) -> None:
    world = World(**overrides)
    with pytest.raises(error_type, match=re.escape(message)):
        _attribute(world, **window)
    [report] = world.report_store.saved
    assert report.outcome is AttributeOutcome.FAILED
    assert (report.failed_stage, report.error) == (stage, f"{error_type.__name__}: {message}")
    assert report.total == NO_PNL


def test_a_failing_aggregate_stage_keeps_the_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*args: Any, **kwargs: Any) -> NoReturn:
        raise RuntimeError("boom")

    monkeypatch.setattr(attribute_service, "aggregate", boom)
    world = World()
    with pytest.raises(RuntimeError, match="boom"):
        _attribute(world)
    [report] = world.report_store.saved
    assert (report.outcome, report.failed_stage) == (AttributeOutcome.FAILED, "aggregate")
    assert report.error == "RuntimeError: boom"
    assert len(report.days) == 7


def test_a_failed_report_keeps_what_was_reached() -> None:
    world = World(
        broker=ScriptedBroker(holdings=(Holding("AAA", 7), Holding("ZZZ", 5))),
        prices=Boom(DataProviderError("down")),
    )
    with pytest.raises(DataProviderError):
        _attribute(world)
    [report] = world.report_store.saved
    assert report.breaks == (ReconciliationBreak("AAA", 6, 7),)
    assert (report.start, report.end, report.days) == (FRI, None, ())


def test_an_interrupt_saves_a_failed_report_and_re_raises() -> None:
    world = World(fills=Boom(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        _attribute(world)
    [report] = world.report_store.saved
    assert (report.outcome, report.failed_stage) == (AttributeOutcome.FAILED, "load")
    assert report.error == "KeyboardInterrupt: "


def test_a_completed_run_whose_report_cannot_be_saved_raises_attribution_save_error() -> None:
    world = World(report_store=InMemoryAttributionStore(fail=OSError("disk full")))
    with pytest.raises(AttributionSaveError, match="the clean attribution report") as caught:
        _attribute(world)
    assert caught.value.report.outcome is AttributeOutcome.CLEAN
    assert len(caught.value.report.days) == 7
    assert isinstance(caught.value.__cause__, OSError)


def test_a_failed_run_whose_report_cannot_be_saved_re_raises_with_a_note() -> None:
    world = World(
        fills=InMemoryFillStore(load_error=ValueError("malformed fill")),
        report_store=InMemoryAttributionStore(fail=OSError("disk full")),
    )
    with pytest.raises(ValueError, match="malformed fill") as caught:
        _attribute(world)
    assert caught.value.__notes__ == [
        "the attribution report could not be saved: OSError: disk full"
    ]
