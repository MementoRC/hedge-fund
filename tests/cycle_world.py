"""Offline world for pipeline-cycle tests: a six-name universe over seeded fakes."""

from __future__ import annotations

import functools
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, date, datetime

from hedge_fund.domain.cost import PriceTable
from hedge_fund.domain.research import Rating
from hedge_fund.domain.shortlist import Side
from hedge_fund.engine.factors.score import score
from hedge_fund.engine.risk.limits import RiskLimits
from hedge_fund.engine.shortlist.select import shortlist
from hedge_fund.services.cycle import CycleDeps, CycleParams
from hedge_fund.services.fundamentals import build_fundamentals_snapshot
from hedge_fund.services.snapshots import build_snapshot
from hedge_fund.universe import Universe
from tests.factories import make_closes, make_passing_facts
from tests.fakes import (
    FakeFundamentalsProvider,
    FakeProvider,
    InMemoryReportStore,
    InMemoryResearchStore,
    InMemorySnapshotStore,
    ScriptedApprovalGate,
    ScriptedBroker,
    ScriptedResearcher,
)

AS_OF = date(2026, 9, 30)  # Wednesday: the last close of make_closes
TODAY = date(2026, 10, 1)  # Thursday: AS_OF is its previous weekday, so a mixed-up date shows
EVENING = datetime(2026, 9, 30, 22, 0, tzinfo=UTC)  # Wed 18:00 EDT: after the window
SESSION = datetime(2026, 9, 30, 14, 0, tzinfo=UTC)  # Wed 10:00 EDT: inside the window
WEEKEND = datetime(2026, 10, 3, 14, 0, tzinfo=UTC)  # Sat 10:00 EDT: no window
SECTORS = {
    f"{prefix}{index}": sector
    for prefix, sector in (("T", "Tech"), ("H", "Health"), ("E", "Energy"))
    for index in range(2)
}  # seed 0 shortlists all six: three longs and three shorts
HELD_OUTSIDE = "ZZZ"  # priced by the provider but not in the universe
FRAME = make_closes([*SECTORS, "SPY", HELD_OUTSIDE], seed=0)  # SPY is the benchmark
FACTS = make_passing_facts(list(SECTORS))
LOOSE_LIMITS = RiskLimits(vol=1.0, name_risk=1.0, factor_share=1.0)  # the seeded book passes


def universe(sectors: Mapping[str, str] = SECTORS) -> Universe:
    return Universe(
        name="test",
        list_date=date(2026, 9, 1),
        tickers=tuple(sectors),
        sectors=dict(sectors),
        ciks={ticker: f"{index:010d}" for index, ticker in enumerate(sectors)},
    )


@functools.cache
def shorts() -> frozenset[str]:
    """The seeded shortlist's short-side tickers (computed once with the real engine)."""
    snapshot = build_snapshot(FakeProvider(FRAME, SECTORS), list(SECTORS), AS_OF).snapshot
    fundamentals = build_fundamentals_snapshot(
        FakeFundamentalsProvider(FACTS), list(SECTORS), AS_OF, ciks=universe().ciks
    ).snapshot
    candidates = shortlist(score(snapshot, fundamentals)).candidates
    return frozenset(c.ticker for c in candidates if c.side is Side.SHORT)


def confirming(
    overrides: Mapping[tuple[str, int], Rating | BaseException] | None = None,
) -> ScriptedResearcher:
    """Longs rated Buy and shorts Sell on every run (all CONFIRM); ``overrides`` win."""
    script: dict[tuple[str, int], Rating | BaseException] = {
        (ticker, run): Rating.SELL for ticker in shorts() for run in range(5)
    }
    script.update(overrides or {})
    return ScriptedResearcher(script, default=Rating.BUY)


class Clock:
    """``now()`` for the cycle: the given instants in turn, then the last one forever."""

    def __init__(self, *times: datetime) -> None:
        self.times = list(times)
        self.calls = 0

    def __call__(self) -> datetime:
        self.calls += 1
        return self.times.pop(0) if len(self.times) > 1 else self.times[0]


@dataclass
class World:
    """Every fake the cycle talks to; ``deps`` wires them into CycleDeps."""

    prices: FakeProvider = field(default_factory=lambda: FakeProvider(FRAME, SECTORS))
    fundamentals: FakeFundamentalsProvider = field(
        default_factory=lambda: FakeFundamentalsProvider(FACTS)
    )
    researcher: ScriptedResearcher = field(default_factory=confirming)
    research_store: InMemoryResearchStore = field(default_factory=InMemoryResearchStore)
    broker: ScriptedBroker = field(default_factory=ScriptedBroker)
    snapshot_store: InMemorySnapshotStore = field(default_factory=InMemorySnapshotStore)
    gate: ScriptedApprovalGate = field(default_factory=lambda: ScriptedApprovalGate([True]))
    report_store: InMemoryReportStore = field(default_factory=InMemoryReportStore)
    universe: Universe = field(default_factory=universe)
    price_table: PriceTable = field(default_factory=PriceTable)
    params: CycleParams = CycleParams(limits=LOOSE_LIMITS)

    @property
    def deps(self) -> CycleDeps:
        return CycleDeps(
            universe=self.universe,
            prices=self.prices,
            fundamentals=self.fundamentals,
            researcher=self.researcher,
            research_store=self.research_store,
            broker=self.broker,
            snapshot_store=self.snapshot_store,
            gate=self.gate,
            report_store=self.report_store,
            params=self.params,
            price_table=self.price_table,
        )
