"""Golden end-to-end backtest: the real stack over a small frozen store, nothing patched.

``write_store`` -> ``load_store`` -> ``build_backtest_deps`` -> ``run_backtest`` (the real
``plan_shortlist`` and ``run_cycle``, default ``CycleParams``) and, on a fresh root, the CLI
``backtest run``. The store is tests/backtest_world.py. The golden numbers below were generated
once from this code and pasted in, so they are regression goldens, not recomputed here; they
are pinned to 1e-9 relative (shares are whole numbers, so only float noise could move them).
"""

from __future__ import annotations

import dataclasses
import hashlib
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from hedge_fund import cli
from hedge_fund.adapters.account import JsonSnapshotStore
from hedge_fund.adapters.backtest.report import save_backtest_report
from hedge_fund.adapters.broker.simulated import SimulatedBroker
from hedge_fund.adapters.cycle import JsonCycleReportStore
from hedge_fund.adapters.data.frozen_providers import FrozenFundamentalsProvider
from hedge_fund.adapters.data.frozen_store import MANIFEST, load_store, write_store
from hedge_fund.domain.backtest import BacktestReport, FrozenData, SimCosts
from hedge_fund.domain.cycle import CycleOutcome, CycleReport
from hedge_fund.domain.execution import SubmitStatus
from hedge_fund.domain.shortlist import Side
from hedge_fund.services.account import load_nav_history
from hedge_fund.services.backtest import backtest_run_id, run_backtest
from hedge_fund.services.fundamentals import build_fundamentals_snapshot
from tests import backtest_world as world

START, END = date(2024, 1, 10), date(2024, 3, 8)
FIRST, SECOND = date(2024, 2, 1), date(2024, 3, 1)  # the monthly rebalance days in START..END
CASH = 100_000.0
CREATED = datetime(2024, 6, 1, tzinfo=UTC)
runner = CliRunner()

# Regression goldens (generated once; see the module docstring).
GOLDEN_NAV_SECOND = 100311.70163555838
GOLDEN_FINAL_EQUITY = 101549.58782493576
GOLDEN_FINAL_CASH = 97824.68106394626
GOLDEN_TOTAL_RETURN = 0.01549587824935772
GOLDEN_BENCHMARK_RETURN = 0.1412692374887463
GOLDEN_TRADED_NOTIONAL = 250780.70164808145
GOLDEN_SLIPPAGE = 125.3926837892853
GOLDEN_BORROW = 34.90168829395957
GOLDEN_RUN_ID = "2024-01-10_2024-03-08_c6eda840a31d"
GOLDEN_FIRST_ORDERS = {
    "AE1": ("sell_short", 20),
    "AE2": ("sell_short", 56),
    "AE3": ("buy", 78),
    "AE4": ("sell_short", 39),
    "AE5": ("buy", 47),
    "AE6": ("buy", 75),
    "BT1": ("sell_short", 199),
    "BT2": ("sell_short", 54),
    "BT3": ("buy", 96),
    "BT4": ("buy", 58),
    "BT5": ("sell_short", 60),
    "CH1": ("sell_short", 63),
    "CH2": ("sell_short", 41),
    "CH3": ("buy", 60),
    "CH4": ("sell_short", 47),
    "CH5": ("buy", 62),
    "CH6": ("buy", 50),
}
GOLDEN_SECOND_ORDER_COUNT = 22
# long/short flips: the closing leg goes out, the opening leg waits for the next cycle
GOLDEN_SECOND_DEFERRED = {"AE4", "BT3", "BT4", "CH3", "CH4"}
GOLDEN_FINAL_POSITIONS = {
    "AE1": -75,
    "AE2": -52,
    "AE3": 67,
    "AE5": 45,
    "BT5": -43,
    "BT6": 45,
    "CH1": -44,
    "CH2": -62,
    "CH5": 73,
    "CH6": 37,
}
GOLDEN_EVENTS = [
    (date(2024, 2, 14), "split", "AE1", -20),
    (date(2024, 2, 21), "dividend", "AE2", -56),
    (date(2024, 2, 28), "delist", "BT1", 199),
]


@dataclass(frozen=True)
class Replay:
    root: Path
    broker: SimulatedBroker
    report: BacktestReport
    cycles: tuple[CycleReport, ...]
    run_id: str


def _freeze(path: Path, data: FrozenData) -> Path:
    write_store(path, data, created_at=CREATED)
    return path


def _digest(store: Path) -> str:
    """The store digest as ``backtest run`` computes it: sha256 of the manifest's bytes."""
    return hashlib.sha256((store / MANIFEST).read_bytes()).hexdigest()


def _replay(store: Path, root: Path, *, end: date = END) -> Replay:
    deps = cli.build_backtest_deps(
        root, load_store(store), store_digest=_digest(store), cash=CASH, costs=SimCosts()
    )
    assert isinstance(deps.broker, SimulatedBroker)
    report = run_backtest(deps, start=START, end=end)
    cycles = JsonCycleReportStore(root).load_all().reports
    return Replay(root, deps.broker, report, cycles, backtest_run_id(deps, START, end))


def _tickers(cycle: CycleReport) -> dict[str, Side]:
    assert cycle.shortlist is not None
    return {c.ticker: c.side for c in cycle.shortlist.candidates}


@pytest.fixture(scope="module")
def store(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _freeze(tmp_path_factory.mktemp("golden") / "store", world.make_golden_data())


@pytest.fixture(scope="module")
def api(store: Path, tmp_path_factory: pytest.TempPathFactory) -> Replay:
    return _replay(store, tmp_path_factory.mktemp("api") / "root")


@pytest.fixture(scope="module")
def cli_run(store: Path, tmp_path_factory: pytest.TempPathFactory) -> tuple[str, Path]:
    root = tmp_path_factory.mktemp("cli") / "root"
    args = ["--store", str(store), "--root", str(root), "--start", START.isoformat()]
    result = runner.invoke(cli.app, ["backtest", "run", *args, "--end", END.isoformat()])
    assert result.exit_code == 0, result.output
    return result.stdout, root


@pytest.fixture(scope="module")
def halted(api: Replay, tmp_path_factory: pytest.TempPathFactory) -> Replay:
    """The store with the book's longs down 10% from the second rebalance day on."""
    account = api.cycles[1].account
    assert account is not None
    longs = [h.ticker for h in account.holdings if h.shares > 0]
    data = world.make_golden_data()
    opens, closes = data.opens.copy(), data.closes.copy()
    crash = closes.index >= pd.Timestamp(SECOND)
    opens.loc[crash, longs] *= 0.9
    closes.loc[crash, longs] *= 0.9
    store = _freeze(
        tmp_path_factory.mktemp("crash") / "store",
        dataclasses.replace(data, opens=opens, closes=closes),
    )
    return _replay(store, tmp_path_factory.mktemp("crash_run") / "root", end=SECOND)


def test_one_cycle_per_month_and_one_snapshot_per_day(api: Replay) -> None:
    assert api.report.rebalances == 2
    assert [c.as_of for c in api.cycles] == [FIRST, SECOND]
    assert len(list((api.root / "runs").glob("*.json"))) == 2
    days = [d.date() for d in pd.bdate_range(START, END)]
    store = JsonSnapshotStore(api.root)
    assert api.report.trading_days == len(days) == 43
    # a cycle day also holds the snapshot the cycle itself saves; the NAV curve keeps one per day
    assert len(store.snapshots()) == len(days) + api.report.rebalances
    assert [p.date for p in load_nav_history(store, as_of=END)] == days


def test_no_cycle_failed_or_found_orders_pending(api: Replay) -> None:
    for cycle in api.cycles:
        assert cycle.failed_stage is None
        assert cycle.error is None
        assert "PendingOrdersError" not in str(cycle.error)
        assert not cycle.execution_interrupted
        assert cycle.outcome is CycleOutcome.SUBMITTED
    assert dict(api.report.metrics.honesty.outcomes) == {CycleOutcome.SUBMITTED.value: 2}


def test_every_submitted_order_reached_the_broker(api: Replay) -> None:
    accepted: set[str] = set()
    deferred: list[set[str]] = []
    for cycle in api.cycles:
        assert cycle.execution is not None
        results = cycle.execution.results
        accepted |= {
            r.request.client_order_id for r in results if r.status is SubmitStatus.ACCEPTED
        }
        deferred.append(
            {r.request.order.ticker for r in results if r.status is SubmitStatus.DEFERRED}
        )
        assert {r.status for r in results} <= {SubmitStatus.ACCEPTED, SubmitStatus.DEFERRED}
    assert accepted
    assert set(api.broker.orders) == accepted
    assert deferred == [set(), GOLDEN_SECOND_DEFERRED]


def test_split_dividend_and_delisting_each_produced_their_event(api: Replay) -> None:
    kinds = {"split", "dividend", "delist"}
    events = [(e.day, e.kind.value, e.ticker, e.shares) for e in api.broker.events]
    assert [e for e in events if e[1] in kinds] == GOLDEN_EVENTS
    honesty = api.report.metrics.honesty
    assert honesty.stopped_trading == 1
    assert dict(honesty.rejects) == {}


def test_a_filing_made_after_a_rebalance_does_not_reach_that_cycle(
    store: Path, api: Replay
) -> None:
    data = load_store(store)
    provider = FrozenFundamentalsProvider(data)

    def retained_earnings(day: date) -> float:
        build = build_fundamentals_snapshot(
            provider, [world.POISON_TICKER], day, ciks=data.ciks, splits=data.splits
        )
        return float(build.snapshot.current.at[world.POISON_TICKER, "retained_earnings"])

    # the FY2023 10-K is filed 2024-02-20: invisible on 2024-02-01, visible on 2024-03-01
    assert retained_earnings(FIRST) == 300.0
    assert retained_earnings(SECOND) == -1_000_000.0
    # so the name is shortlisted (and held) by the first cycle, and gated out of the second
    assert _tickers(api.cycles[0])[world.POISON_TICKER] is Side.SHORT
    assert world.POISON_TICKER not in _tickers(api.cycles[1])
    held_at_second = api.cycles[1].account
    assert held_at_second is not None
    assert world.POISON_TICKER in {h.ticker for h in held_at_second.holdings}


def test_the_membership_change_is_effective_on_the_rebalance_day(api: Replay) -> None:
    first, second = _tickers(api.cycles[0]), _tickers(api.cycles[1])
    assert world.REMOVED in first
    assert world.REMOVED not in second
    assert world.ADDED not in first
    assert world.ADDED in second


def test_the_first_rebalance_the_cli_names_is_the_first_cycle_that_ran(
    cli_run: tuple[str, Path],
) -> None:
    out, root = cli_run
    cycles = JsonCycleReportStore(root).load_all().reports
    assert f"first rebalance on {cycles[0].as_of}" in out
    assert cycles[0].as_of == FIRST
    assert f"{START} to {END}, 43 trading days, 2 rebalances" in out


def test_the_cli_run_and_the_api_run_are_identical(
    api: Replay, cli_run: tuple[str, Path], store: Path
) -> None:
    out, cli_root = cli_run
    assert api.run_id == GOLDEN_RUN_ID == api.report.run_id
    assert f"backtest run {api.run_id}" in out
    assert api.report.store_digest == _digest(store)
    # created_at stamps in the account and research stores use the wall clock by design, so
    # compare decoded records and the run and summary files, which carry none
    assert JsonCycleReportStore(cli_root).load_all().reports == api.cycles
    assert JsonSnapshotStore(cli_root).snapshots() == JsonSnapshotStore(api.root).snapshots()
    save_backtest_report(api.root, api.report)
    for name in sorted(p.name for p in (api.root / "runs").glob("*.json")):
        assert (cli_root / "runs" / name).read_bytes() == (api.root / "runs" / name).read_bytes()
    summary = f"backtest/{api.run_id}.json"
    assert (cli_root / summary).read_bytes() == (api.root / summary).read_bytes()


def test_golden_nav_orders_positions_and_metrics(api: Replay) -> None:
    snapshots = JsonSnapshotStore(api.root).snapshots()
    nav = {s.as_of: s.equity for s in snapshots}
    # hand check: nothing trades before the first cycle's orders fill, so equity is the cash
    assert all(nav[s.as_of] == CASH for s in snapshots if s.as_of <= FIRST)
    assert nav[SECOND] == pytest.approx(GOLDEN_NAV_SECOND, rel=1e-9)
    assert nav[END] == pytest.approx(GOLDEN_FINAL_EQUITY, rel=1e-9)
    first = api.cycles[0].plan
    assert first is not None
    assert {o.ticker: (o.action.value, o.shares) for o in first.orders} == GOLDEN_FIRST_ORDERS
    second = api.cycles[1].plan
    assert second is not None
    assert len(second.orders) == GOLDEN_SECOND_ORDER_COUNT
    assert {h.ticker: h.shares for h in api.broker.positions()} == GOLDEN_FINAL_POSITIONS
    assert api.broker.cash == pytest.approx(GOLDEN_FINAL_CASH, rel=1e-9)
    metrics = api.report.metrics
    assert metrics.strategy.total_return == pytest.approx(GOLDEN_TOTAL_RETURN, rel=1e-9)
    assert metrics.benchmark.total_return == pytest.approx(GOLDEN_BENCHMARK_RETURN, rel=1e-9)
    assert metrics.costs.traded_notional == pytest.approx(GOLDEN_TRADED_NOTIONAL, rel=1e-9)
    assert metrics.costs.slippage == pytest.approx(GOLDEN_SLIPPAGE, rel=1e-9)
    assert metrics.costs.borrow == pytest.approx(GOLDEN_BORROW, rel=1e-9)


def test_a_planted_crash_halts_that_cycle_before_planning(halted: Replay) -> None:
    assert [c.outcome for c in halted.cycles] == [
        CycleOutcome.SUBMITTED,
        CycleOutcome.HALTED_PREFLIGHT,
    ]
    assert halted.cycles[1].plan is None
    honesty = halted.report.metrics.honesty
    assert dict(honesty.outcomes) == {
        CycleOutcome.SUBMITTED.value: 1,
        CycleOutcome.HALTED_PREFLIGHT.value: 1,
    }
    assert honesty.daily_breaker_days == 1
