"""Run-once CLI: ``hedge-fund cycle`` runs one paper-trading cycle (Layer 6d),
``hedge-fund settle`` records what it filled (Layer 7a) and ``hedge-fund attribute`` splits
the book's P&L (Layer 7b) and ``hedge-fund verify`` scores the stored research ratings
against what followed (Layer 7d). ``hedge-fund backtest snapshot`` freezes point-in-time
data for the offline backtest and ``hedge-fund backtest run`` replays it (Layer 8).
``hedge-fund demo build`` writes the synthetic demo world (a frozen store and a backtest run over it).

This is the only composition root: ``build_deps``, ``build_settle_deps``,
``build_attribute_deps``, ``build_verify_deps``, ``build_snapshot_sources``,
``build_backtest_deps`` (simulated broker only) and ``hedge-fund demo build`` build the adapters
and ``clock`` is the only clock read.
Tests replace them. Paper trading only.
"""

from __future__ import annotations

import hashlib
import os
import shutil
from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Annotated, NoReturn

import typer

from hedge_fund.adapters.account import JsonSnapshotStore
from hedge_fund.adapters.approval import ConsoleApprovalGate
from hedge_fund.adapters.approval.auto import AutoApprovalGate
from hedge_fund.adapters.attribution import JsonAttributionReportStore
from hedge_fund.adapters.attribution.json_report import ATTRIBUTIONS_DIR
from hedge_fund.adapters.backtest.report import (
    BACKTEST_DIR,
    load_backtest_reports,
    save_backtest_report,
)
from hedge_fund.adapters.broker import AlpacaPaperBroker
from hedge_fund.adapters.broker.simulated import SimulatedBroker, SimulationError
from hedge_fund.adapters.cost import PRICES_FILE, load_prices
from hedge_fund.adapters.cycle import JsonCycleReportStore
from hedge_fund.adapters.cycle.json_report import RUNS_DIR
from hedge_fund.adapters.data.edgar_provider import EdgarProvider
from hedge_fund.adapters.data.factory import default_fundamentals_provider, default_provider
from hedge_fund.adapters.data.frozen_providers import FrozenDataProvider, FrozenFundamentalsProvider
from hedge_fund.adapters.data.frozen_store import (
    MANIFEST,
    check_fresh_root,
    load_store,
    write_store,
)
from hedge_fund.adapters.data.sec_sources import SecLookups
from hedge_fund.adapters.data.yfinance_history import YFinanceHistory
from hedge_fund.adapters.data.yfinance_provider import YFinanceProvider
from hedge_fund.adapters.demo_marker import (
    DEMO_MARKER,
    demo_store_dir,
    read_demo_marker,
    write_demo_marker,
)
from hedge_fund.adapters.ledger import (
    JsonFillStore,
    JsonOpeningBalance,
    JsonSettlementReportStore,
)
from hedge_fund.adapters.ledger.settlement_report import SETTLEMENTS_DIR
from hedge_fund.adapters.research.neutral import NeutralResearcher
from hedge_fund.adapters.research.store import JsonResearchStore
from hedge_fund.adapters.research.tradingagents import ResearchConfig, TradingAgentsResearcher
from hedge_fund.adapters.verification import JsonVerificationReportStore
from hedge_fund.adapters.verification.json_report import VERIFICATIONS_DIR
from hedge_fund.demo import world
from hedge_fund.domain.attribution import (
    DIVIDEND_NOTE,
    AttributeOutcome,
    AttributionReport,
    AttributionTotals,
)
from hedge_fund.domain.backtest import (
    MEMBERSHIP_FLOOR,
    BacktestReport,
    FrozenData,
    Honesty,
    Performance,
    SimCosts,
)
from hedge_fund.domain.cost import BudgetPlan, Spend
from hedge_fund.domain.cycle import CycleOptions, CycleReport
from hedge_fund.domain.demo import DemoMarker
from hedge_fund.domain.execution import SubmitStatus
from hedge_fund.domain.ledger import Fill, MarkedLot, SettlementReport, SettleOutcome
from hedge_fund.domain.research import ResearchResult
from hedge_fund.domain.risk import RiskVerdict
from hedge_fund.domain.verification import (
    Bucket,
    HorizonSummary,
    VerificationReport,
    VerifyOutcome,
)
from hedge_fund.engine.backtest.calendar import in_window, rebalance_days, trading_days
from hedge_fund.engine.backtest.membership import (
    ever_members,
    parse_flagged_spans,
    parse_membership,
)
from hedge_fund.engine.execution import MARKET_TZ
from hedge_fund.ports.broker import BrokerError
from hedge_fund.ports.fundamentals import FundamentalsProviderError
from hedge_fund.services.attribute import AttributeDeps, AttributionSaveError, run_attribute
from hedge_fund.services.backtest import BacktestDeps, BacktestError, run_backtest
from hedge_fund.services.backtest_snapshot import SnapshotSources, build_frozen_data
from hedge_fund.services.cycle import (
    CycleDeps,
    CycleParams,
    PendingOrdersError,
    ReportSaveError,
    SubmissionWindowError,
    run_cycle,
)
from hedge_fund.services.settle import SettleDeps, SettlementSaveError, run_settle
from hedge_fund.services.verify import VerificationSaveError, VerifyDeps, run_verify
from hedge_fund.universe import load_universe, resource_text

UNIVERSE = "sp500"
MEMBERSHIP = "sp500_membership"
FLAGGED_SPANS = "sp500_flagged_spans"
DEFAULT_SNAPSHOT_START = date(2010, 1, 1)
DEFAULT_BACKTEST_CASH = 100_000.0
DEFAULT_DEMO_ROOT = Path("data", "demo")
_FAILED_LEGS = frozenset({SubmitStatus.ERROR, SubmitStatus.REJECTED})
_SETTLE_EXIT: Mapping[SettleOutcome, int] = {
    SettleOutcome.CLEAN: 0,
    SettleOutcome.PENDING: 0,
    SettleOutcome.BREAK: 2,
    SettleOutcome.UNSETTLED: 1,
}
_ATTRIBUTE_EXIT: Mapping[AttributeOutcome, int] = {
    AttributeOutcome.CLEAN: 0,
    AttributeOutcome.BREAK: 2,
}
_VERIFY_EXIT: Mapping[VerifyOutcome, int] = {
    VerifyOutcome.CLEAN: 0,
    VerifyOutcome.GAPS: 2,
    VerifyOutcome.FAILED: 1,
}
_OVER_BUDGET: Mapping[bool | None, str] = {None: "n/a", True: "yes", False: "no"}

app = typer.Typer(add_completion=False)
backtest_app = typer.Typer(add_completion=False, help="Offline backtest of the quant core.")
app.add_typer(backtest_app, name="backtest")
demo_app = typer.Typer(add_completion=False, help="The synthetic demo world.")
app.add_typer(demo_app, name="demo")


@app.callback()
def main() -> None:
    """Hedge-fund paper-trading commands."""


def clock() -> datetime:
    """The wall clock in UTC."""
    return datetime.now(UTC)


def build_deps(root: Path, prices: Path) -> CycleDeps:
    """The real adapters, with every file under ``root``, and the LLM price table at ``prices``.

    The broker and the fundamentals provider are built first, so missing credentials
    (BrokerError, FundamentalsProviderError) fail before anything is read; a malformed price
    table raises ValueError.
    """
    broker = AlpacaPaperBroker()
    fundamentals = default_fundamentals_provider(root / "cache" / "fundamentals", UNIVERSE)
    price_table = load_prices(prices)
    return CycleDeps(
        universe=load_universe(UNIVERSE),
        prices=default_provider(root / "cache" / "prices", UNIVERSE),
        fundamentals=fundamentals,
        researcher=TradingAgentsResearcher(ResearchConfig(root=root / "scratch")),
        research_store=JsonResearchStore(root),
        broker=broker,
        snapshot_store=JsonSnapshotStore(root),
        gate=ConsoleApprovalGate(),
        report_store=JsonCycleReportStore(root),
        price_table=price_table,
    )


def build_settle_deps(root: Path) -> SettleDeps:
    """The real adapters for ``settle``, with every file under ``root``.

    The broker is built first, so missing Alpaca credentials (BrokerError) fail before
    anything is read. Prices come from the cycle's cache (``root/cache/prices``).
    """
    broker = AlpacaPaperBroker()
    return SettleDeps(
        broker=broker,
        prices=default_provider(root / "cache" / "prices", UNIVERSE),
        cycle_reports=JsonCycleReportStore(root),
        fills=JsonFillStore(root),
        opening=JsonOpeningBalance(root),
        report_store=JsonSettlementReportStore(root),
    )


def build_attribute_deps(root: Path) -> AttributeDeps:
    """The real adapters for ``attribute``, with every file under ``root``.

    The broker is built first, so missing Alpaca credentials (BrokerError) fail before
    anything is read. Prices come from the cycle's cache (``root/cache/prices``); the
    universe and the risk params are the cycle's.
    """
    broker = AlpacaPaperBroker()
    return AttributeDeps(
        broker=broker,
        prices=default_provider(root / "cache" / "prices", UNIVERSE),
        universe=load_universe(UNIVERSE),
        risk=CycleParams().risk,
        fills=JsonFillStore(root),
        opening=JsonOpeningBalance(root),
        report_store=JsonAttributionReportStore(root),
    )


def build_verify_deps(root: Path) -> VerifyDeps:
    """The real adapters for ``verify``, with every file under ``root``.

    No broker: verification reads stored research, recorded fills and prices only. Prices
    come from the cycle's cache (``root/cache/prices``).
    """
    return VerifyDeps(
        research=JsonResearchStore(root),
        fills=JsonFillStore(root),
        data=default_provider(root / "cache" / "prices", UNIVERSE),
        reports=JsonVerificationReportStore(root),
    )


def build_backtest_deps(
    root: Path, data: FrozenData, *, store_digest: str, cash: float, costs: SimCosts
) -> BacktestDeps:
    """The backtest's composition root: a SimulatedBroker over ``data`` and nothing else.

    No Alpaca broker, vendor or LLM is built here, so no real or paper-live broker can reach
    a backtest. Every file the replay writes is under ``root``.
    """
    return BacktestDeps(
        data=data,
        store_digest=store_digest,
        cash=cash,
        costs=costs,
        broker=SimulatedBroker(data, cash, costs),
        snapshot_store=JsonSnapshotStore(root),
        report_store=JsonCycleReportStore(root),
        research_store=JsonResearchStore(root),
        gate=AutoApprovalGate(),
        prices=lambda day: FrozenDataProvider(data, day),
        fundamentals=FrozenFundamentalsProvider(data),
        researcher=NeutralResearcher,
        params=CycleParams(),
    )


def build_snapshot_sources() -> SnapshotSources:
    """The real vendors for ``backtest snapshot``: yfinance prices and splits, SEC EDGAR.

    The SEC client is built first, so a missing SEC_USER_AGENT (FundamentalsProviderError)
    fails before anything is fetched.
    """
    sec = SecLookups(EdgarProvider({}))
    splits = YFinanceProvider()
    return SnapshotSources(
        history=YFinanceHistory(),
        splits=lambda tickers: splits.splits(tickers, date.min, date.max),
        company_ciks=sec.company_ciks,
        sic=sec.sic,
        fundamentals=EdgarProvider,
    )


@app.command()
def cycle(
    as_of: Annotated[
        datetime | None,
        typer.Option(
            "--as-of",
            formats=["%Y-%m-%d"],
            help="Default: today in New York. The previous weekday only before 09:30 New York.",
        ),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Stop at the order plan: no approval, orders or snapshot."),
    ] = False,
    fresh_research: Annotated[
        bool,
        typer.Option("--fresh-research", help="Research again even if stored research matches."),
    ] = False,
    any_time: Annotated[
        bool, typer.Option("--any-time", help="Skip the weekday 09:25-16:15 ET window check.")
    ] = False,
    k: Annotated[int, typer.Option("-k", "--k", min=1, help="Research runs per name.")] = 3,
    budget: Annotated[
        float | None,
        typer.Option(
            "--budget", help="LLM budget in USD for this cycle's research: cuts K, then N."
        ),
    ] = None,
    prices: Annotated[
        Path | None,
        typer.Option("--prices", help="LLM price table (TOML). Default: <root>/prices.toml."),
    ] = None,
    root: Annotated[Path, typer.Option("--root", help="Data root.")] = Path("data"),
) -> None:
    """Run one paper-trading cycle, from data to execution, and save its report.

    Refuses (exit 1) while orders from an earlier cycle are still open at the broker; run
    `settle` or wait for the open first. A dry run does not check.
    """
    min_valid = CycleParams().conviction.min_valid
    if k < min_valid:
        raise typer.BadParameter(f"k={k} is below min_valid={min_valid}", param_hint="-k")
    runs = root / RUNS_DIR
    try:
        deps = build_deps(root, root / PRICES_FILE if prices is None else prices)
    except (BrokerError, FundamentalsProviderError, ValueError) as exc:
        _fail(f"cannot start the cycle: {exc}", exc)
    today = clock().astimezone(MARKET_TZ).date()
    options = CycleOptions(
        dry_run=dry_run, fresh_research=fresh_research, k=k, any_time=any_time, budget=budget
    )
    day = today if as_of is None else as_of.date()
    try:
        report = run_cycle(deps, as_of=day, today=today, options=options, now=clock)
    except ReportSaveError as exc:
        typer.echo(_summary(exc.report, runs))
        typer.echo(f"error: the cycle report was not saved: {exc.__cause__!r}", err=True)
        raise typer.Exit(max(_exit_code(exc.report), 1)) from exc
    except SubmissionWindowError as exc:
        _fail(f"{exc}; --any-time overrides this check", exc)
    except PendingOrdersError as exc:
        _fail(str(exc), exc)
    except KeyboardInterrupt as exc:
        _fail(f"interrupted; the partial report is under {runs}", exc)
    except Exception as exc:
        _fail(f"{type(exc).__name__}: {exc}; the partial report is under {runs}", exc)
    typer.echo(_summary(report, runs))
    raise typer.Exit(_exit_code(report))


@app.command()
def settle(
    as_of: Annotated[
        datetime | None,
        typer.Option(
            "--as-of", formats=["%Y-%m-%d"], help="Date of the marks. Default: today in New York."
        ),
    ] = None,
    root: Annotated[Path, typer.Option("--root", help="Data root.")] = Path("data"),
) -> None:
    """Record final fills, rebuild and reconcile the lots, mark them, and save a report.

    Exit 0 when clean or pending, 2 on a reconciliation break, 1 when an order is unsettled
    or the run fails.
    """
    settlements = root / SETTLEMENTS_DIR
    try:
        deps = build_settle_deps(root)
    except (BrokerError, ValueError) as exc:
        _fail(f"cannot start settlement: {exc}", exc)
    today = clock().astimezone(MARKET_TZ).date()
    day = today if as_of is None else as_of.date()
    try:
        report = run_settle(deps, as_of=day, today=today, now=clock)
    except SettlementSaveError as exc:
        typer.echo(_settle_summary(exc.report, settlements))
        typer.echo(f"error: the settlement report was not saved: {exc.__cause__!r}", err=True)
        raise typer.Exit(max(_SETTLE_EXIT[exc.report.outcome], 1)) from exc
    except KeyboardInterrupt as exc:
        _fail(f"interrupted; the partial report is under {settlements}", exc)
    except Exception as exc:
        _fail(f"{type(exc).__name__}: {exc}; the partial report is under {settlements}", exc)
    typer.echo(_settle_summary(report, settlements))
    raise typer.Exit(_SETTLE_EXIT[report.outcome])


@app.command()
def attribute(
    start: Annotated[
        datetime | None,
        typer.Option(
            "--start",
            formats=["%Y-%m-%d"],
            help="First day. Default: the first opening or fill date.",
        ),
    ] = None,
    end: Annotated[
        datetime | None,
        typer.Option("--end", formats=["%Y-%m-%d"], help="Last day. Default: the last SPY close."),
    ] = None,
    root: Annotated[Path, typer.Option("--root", help="Data root.")] = Path("data"),
) -> None:
    """Split the book's daily P&L into market, sector, residual and trading; run it after settle.

    A fill settle has not recorded yet shows as a break. Exit 0 when clean, 2 on a
    reconciliation break, 1 when the run fails.
    """
    attributions = root / ATTRIBUTIONS_DIR
    try:
        deps = build_attribute_deps(root)
    except (BrokerError, ValueError) as exc:
        _fail(f"cannot start attribution: {exc}", exc)
    first = None if start is None else start.date()
    last = None if end is None else end.date()
    try:
        report = run_attribute(deps, start=first, end=last, now=clock)
    except AttributionSaveError as exc:
        typer.echo(_attribute_summary(exc.report, attributions))
        typer.echo(f"error: the attribution report was not saved: {exc.__cause__!r}", err=True)
        raise typer.Exit(max(_ATTRIBUTE_EXIT[exc.report.outcome], 1)) from exc
    except KeyboardInterrupt as exc:
        _fail(f"interrupted; the partial report is under {attributions}", exc)
    except Exception as exc:
        _fail(f"{type(exc).__name__}: {exc}; the partial report is under {attributions}", exc)
    typer.echo(_attribute_summary(report, attributions))
    raise typer.Exit(_ATTRIBUTE_EXIT[report.outcome])


@app.command()
def verify(
    start: Annotated[
        datetime | None,
        typer.Option(
            "--start", formats=["%Y-%m-%d"], help="First research date. Default: the earliest."
        ),
    ] = None,
    end: Annotated[
        datetime | None,
        typer.Option(
            "--end", formats=["%Y-%m-%d"], help="Last research date. Default: today in New York."
        ),
    ] = None,
    root: Annotated[Path, typer.Option("--root", help="Data root.")] = Path("data"),
) -> None:
    """Score stored research ratings against the excess return over SPY after 5, 21 and 63 days.

    Independent of settle; run it any time. Exit 0 when clean, 2 when a name had no price, 1
    when the run fails.
    """
    verifications = root / VERIFICATIONS_DIR
    try:
        deps = build_verify_deps(root)
    except ValueError as exc:
        _fail(f"cannot start verification: {exc}", exc)
    first = None if start is None else start.date()
    last = None if end is None else end.date()
    try:
        report = run_verify(deps, start=first, end=last, now=clock)
    except VerificationSaveError as exc:
        typer.echo(_verify_summary(exc.report, verifications))
        typer.echo(f"error: the verification report was not saved: {exc.__cause__!r}", err=True)
        raise typer.Exit(max(_VERIFY_EXIT[exc.report.outcome], 1)) from exc
    except KeyboardInterrupt as exc:
        _fail(f"interrupted; the partial report is under {verifications}", exc)
    except Exception as exc:
        _fail(f"{type(exc).__name__}: {exc}; the partial report is under {verifications}", exc)
    typer.echo(_verify_summary(report, verifications))
    raise typer.Exit(_VERIFY_EXIT[report.outcome])


@backtest_app.command("snapshot")
def backtest_snapshot(
    root: Annotated[Path, typer.Option("--root", help="Store root; must be absent or empty.")],
    start: Annotated[
        datetime | None,
        typer.Option(
            "--start",
            formats=["%Y-%m-%d"],
            help="First day of prices. Default: 2010-01-01. Earlier than 2008-01-01 is refused.",
        ),
    ] = None,
    end: Annotated[
        datetime | None,
        typer.Option("--end", formats=["%Y-%m-%d"], help="Last day. Default: yesterday in NY."),
    ] = None,
    membership: Annotated[
        Path | None,
        typer.Option("--membership", help="Membership CSV. Default: the packaged S&P 500 file."),
    ] = None,
) -> None:
    """Freeze point-in-time prices, splits, dividends, facts and sectors (online, once).

    Refuses a non-empty --root before fetching anything. Exit 0 when the store is written,
    1 on any failure; a failed run leaves no manifest, so its root never loads as a store.
    """
    first = DEFAULT_SNAPSHOT_START if start is None else start.date()
    try:
        if first < MEMBERSHIP_FLOOR:
            raise ValueError(
                f"--start {first} is before {MEMBERSHIP_FLOOR}: membership is not reliable earlier"
            )
        check_fresh_root(root)
        spans = parse_membership(_membership_text(membership))
        flagged = parse_flagged_spans(resource_text(FLAGGED_SPANS))
        universe = load_universe(UNIVERSE)
        sources = build_snapshot_sources()
    except (ValueError, OSError, FundamentalsProviderError) as exc:
        _fail(f"cannot start the snapshot: {exc}", exc)
    last = _yesterday_in_new_york() if end is None else end.date()
    try:
        data = build_frozen_data(
            sources,
            spans,
            sectors=universe.sectors,
            ciks=universe.ciks,
            start=first,
            end=last,
            flagged=flagged,
        )
        digest = write_store(root, data, created_at=clock())
    except KeyboardInterrupt as exc:
        _fail(f"interrupted; {root} has no manifest: delete it and use a fresh root", exc)
    except Exception as exc:
        _fail(f"{type(exc).__name__}: {exc}", exc)
    typer.echo(_snapshot_summary(data, root, digest))


@backtest_app.command("run")
def backtest_run(
    store: Annotated[
        Path, typer.Option("--store", help="A frozen store from `backtest snapshot`.")
    ],
    root: Annotated[Path, typer.Option("--root", help="Run root; must be absent or empty.")],
    start: Annotated[
        datetime,
        typer.Option("--start", formats=["%Y-%m-%d"], help="First day (default 2012-01-01)."),
    ] = datetime(2012, 1, 1),  # noqa: B008, DTZ001
    end: Annotated[
        datetime | None,
        typer.Option("--end", formats=["%Y-%m-%d"], help="Last day. Default: the store's end."),
    ] = None,
    cash: Annotated[float, typer.Option("--cash", help="Starting cash.")] = DEFAULT_BACKTEST_CASH,
    slippage_bps: Annotated[float, typer.Option("--slippage-bps")] = SimCosts.slippage_bps,
    borrow_rate: Annotated[float, typer.Option("--borrow-rate")] = SimCosts.borrow_rate,
) -> None:
    """Replay the unchanged cycle over a frozen store with a simulated paper broker.

    Offline: no network, no LLM, no broker credentials. Cycles run on the first trading day of
    each month. Exit 0 with the summary and a report under --root, 1 on any failure.
    """
    try:
        data = load_store(store)
        digest = hashlib.sha256((store / MANIFEST).read_bytes()).hexdigest()
    except Exception as exc:  # a corrupt store can raise KeyError or a parquet reader error
        _fail(f"cannot start the backtest: {type(exc).__name__}: {exc}", exc)
    try:
        check_fresh_root(root)
        _refuse_root_in_store(root, store)
        costs = SimCosts(slippage_bps=slippage_bps, borrow_rate=borrow_rate)
    except (ValueError, OSError) as exc:
        _fail(f"cannot start the backtest: {exc}", exc)
    first, last = start.date(), data.end if end is None else end.date()
    try:
        deps = build_backtest_deps(root, data, store_digest=digest, cash=cash, costs=costs)
        report = run_backtest(deps, start=first, end=last)
        path = save_backtest_report(root, report)
    except (BacktestError, SimulationError) as exc:
        _fail(str(exc), exc)
    except Exception as exc:
        _fail(f"{type(exc).__name__}: {exc}", exc)
    typer.echo(_backtest_summary(report, data, path))


@demo_app.command("build")
def demo_build(
    root: Annotated[
        Path,
        typer.Option(
            "--root", help="Demo root; its frozen store goes to the sibling <root>.store."
        ),
    ] = DEFAULT_DEMO_ROOT,
    seed: Annotated[
        int,
        typer.Option(
            "--seed", min=1, help="Seed of the synthetic world (>= 1; the facts use seed - 1)."
        ),
    ] = world.DEFAULT_SEED,
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            help="Replace an existing demo: needs <root>/DEMO.json and <root>.store/manifest.json.",
        ),
    ] = False,
) -> None:
    """Build the synthetic demo world: a frozen store, a real backtest over it, a marker.

    Writes the store to <root>.store, replays the unchanged cycle into <root> exactly as
    `backtest run` does, then writes <root>/DEMO.json last. Refuses existing paths unless
    --force, which deletes both directories only when they are a demo this command built. A
    failed build leaves no marker: delete <root> and <root>.store by hand and retry.
    """
    root = root.resolve()
    store = demo_store_dir(root)
    try:
        _clear_demo(root, store, force=force)
    except (ValueError, OSError) as exc:
        _fail(str(exc), exc)
    try:
        digest = write_store(store, world.make_golden_data(seed), created_at=world.FROZEN_AT)
    except Exception as exc:
        _fail(f"cannot write the demo store: {type(exc).__name__}: {exc}", exc)
    backtest_run(
        store=store,
        root=root,
        start=datetime.combine(world.RUN_START, time.min),
        end=None,
        cash=DEFAULT_BACKTEST_CASH,
        slippage_bps=SimCosts.slippage_bps,
        borrow_rate=SimCosts.borrow_rate,
    )
    try:
        run_id = _demo_run_id(root, digest)
    except ValueError as exc:
        _fail(str(exc), exc)
    write_demo_marker(root, DemoMarker(seed=seed, store_digest=digest, run_id=run_id))
    typer.echo(f"demo root {root}: seed {seed}, store {store}, run {run_id}")


def _clear_demo(root: Path, store: Path, *, force: bool) -> None:
    """Leave ``root`` and ``store`` absent, or raise ValueError without touching either.

    Existing paths need ``force`` and must look like a demo: ``root/DEMO.json`` parses, and
    the store has a manifest (or is absent). A live root or ``data/`` can never be wiped.
    """
    if not (os.path.lexists(root) or os.path.lexists(store)):
        return
    if not force:
        raise ValueError(f"{root} or {store} already exists; use --force to replace a demo")
    marker = read_demo_marker(root)
    if marker is None:
        raise ValueError(f"{root} has no {DEMO_MARKER}: not a demo root, refusing to delete it")
    if os.path.lexists(store) and not (store / MANIFEST).is_file():
        raise ValueError(f"{store} has no {MANIFEST}: not a demo store, refusing to delete it")
    summary = root / BACKTEST_DIR / f"{marker.run_id}.json"
    if not summary.is_file():
        raise ValueError(
            f"{summary} is missing: not the demo this command built, refusing to delete"
        )
    shutil.rmtree(root)
    if store.exists():
        shutil.rmtree(store)


def _demo_run_id(root: Path, store_digest: str) -> str:
    runs = load_backtest_reports(root)
    if len(runs.items) != 1 or runs.unreadable:
        raise ValueError(
            f"expected exactly one readable backtest summary under {root}, found "
            f"{len(runs.items)} and {len(runs.unreadable)} unreadable"
        )
    report = runs.items[0]
    if report.store_digest != store_digest:
        raise ValueError(
            f"the backtest summary's store_digest {report.store_digest!r} is not the store "
            f"just written ({store_digest!r})"
        )
    return report.run_id


def _refuse_root_in_store(root: Path, store: Path) -> None:
    resolved = root.resolve()
    if resolved == store.resolve() or store.resolve() in resolved.parents:
        raise ValueError(f"--root {root} is the store or inside it")


def _backtest_summary(report: BacktestReport, data: FrozenData, path: Path) -> str:
    metrics = report.metrics
    lines = [
        f"backtest run {report.run_id}",
        f"{report.start} to {report.end}, {report.trading_days} trading days, "
        f"{report.rebalances} rebalances",
    ]
    note = _first_rebalance_note(report, data)
    if note:
        lines.append(note)
    lines += _section(
        "strategy vs SPY",
        [
            _performance_line("strategy", metrics.strategy),
            _performance_line("SPY", metrics.benchmark),
            f"beta {_number(metrics.beta)}",
        ],
    )
    costs = metrics.costs
    lines += _section(
        "costs",
        [
            f"turnover {costs.turnover:.2f}x a year, traded ${costs.traded_notional:,.0f}",
            f"slippage ${costs.slippage:,.2f}, borrow ${costs.borrow:,.2f}",
        ],
    )
    lines += _section("honesty", _honesty_lines(metrics.honesty))
    lines += _section("residual biases", list(report.biases))
    lines.append(f"report {path}")
    return "\n".join(lines)


def _first_rebalance_note(report: BacktestReport, data: FrozenData) -> str | None:
    calendar = trading_days(data.closes.index)
    window = set(in_window(calendar, report.start, report.end))
    firsts = sorted(rebalance_days(calendar) & window)
    if not firsts or firsts[0] == report.start:
        return None
    return f"first rebalance on {firsts[0]}"


def _performance_line(name: str, perf: Performance) -> str:
    return (
        f"{name}: CAGR {_percent(perf.cagr)}, volatility {_percent(perf.volatility)}, "
        f"Sharpe {_number(perf.sharpe)}, max drawdown {_percent(perf.max_drawdown)}"
    )


def _number(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def _counts(counts: Mapping[str, int]) -> str:
    return ", ".join(f"{key} {count}" for key, count in sorted(counts.items())) or "none"


def _honesty_lines(honesty: Honesty) -> list[str]:
    return [
        f"price unresolved: {_counts(honesty.price_unresolved)}",
        f"fundamentals unresolved {honesty.fundamentals_unresolved}, "
        f"flagged excluded {honesty.flagged_excluded}, stopped trading {honesty.stopped_trading}",
        f"rejects: {_counts(honesty.rejects)}",
        f"breaker days would trip: daily {honesty.daily_breaker_days}, "
        f"drawdown {honesty.drawdown_breaker_days}",
        f"cycle outcomes: {_counts(honesty.outcomes)}",
    ]


def _fail(message: str, cause: BaseException) -> NoReturn:
    typer.echo(f"error: {message}", err=True)
    raise typer.Exit(1) from cause


def _membership_text(path: Path | None) -> str:
    return resource_text(MEMBERSHIP) if path is None else path.read_text(encoding="utf-8")


def _yesterday_in_new_york() -> date:
    return clock().astimezone(MARKET_TZ).date() - timedelta(days=1)


def _snapshot_summary(data: FrozenData, root: Path, digest: str) -> str:
    members = len(ever_members(data.membership, data.start, data.end))
    return "\n".join(
        [
            f"frozen store {root}: {data.start} to {data.end}, manifest sha256 {digest}",
            f"members {members}, priced {len(data.closes.columns)} (SPY included), "
            f"unpriced {len(data.unpriced)}",
            f"splits {len(data.splits)}, dividends {len(data.dividends)}, "
            f"fact rows {len(data.facts)}",
            f"no sector {len(data.no_sector)}, sector from SIC {len(data.sic_sectors)}, "
            f"no CIK {len(data.no_cik)}",
        ]
    )


def _exit_code(report: CycleReport) -> int:
    """2 when a submitted leg was ERROR or REJECTED, else 0."""
    results = () if report.execution is None else report.execution.results
    return 2 if any(result.status in _FAILED_LEGS for result in results) else 0


def _summary(report: CycleReport, runs: Path) -> str:
    """Outcome, account, shortlist, research, budget, spend, verdict, plan and legs; reports dir."""
    lines = [f"outcome: {report.outcome.value}", f"as_of: {report.as_of}"]
    if report.account is not None:
        holdings = len(report.account.holdings)
        lines.append(f"account: equity {report.account.equity:,.2f}, {holdings} holdings")
    if report.shortlist is not None:
        lines.append(f"shortlist: {len(report.shortlist.candidates)} candidates")
    lines += _research_lines(report)
    if report.verdict is not None:
        lines.append(f"verdict: {_verdict_line(report.verdict)}")
    if report.plan is not None:
        lines.append(f"orders: {len(report.plan.orders)}")
    if report.execution is not None:
        for result in report.execution.results:
            reason = "" if result.reason is None else f" ({result.reason})"
            lines.append(f"  {result.request.client_order_id}: {result.status.value}{reason}")
    lines.append(f"reports: {runs}")
    return "\n".join(lines)


def _research_line(research: ResearchResult, *, reused: bool) -> str:
    records = [record for name in research.names for record in name.runs]
    errors = sum(1 for record in records if record.error is not None)
    suffix = " (reused)" if reused else ""
    return (
        f"research: {len(research.names)} names, {len(records)} runs, error runs: {errors}{suffix}"
    )


def _research_lines(report: CycleReport) -> list[str]:
    """The research line, then the budget plan and the spend when the cycle has them."""
    lines: list[str] = []
    if report.research is not None:
        lines.append(_research_line(report.research, reused=report.research_reused))
    if report.budget is not None:
        lines += _budget_lines(report.budget)
    if report.spend is not None:
        lines.append(_spend_line(report.spend))
    return lines


def _budget_lines(plan: BudgetPlan) -> list[str]:
    estimate = plan.estimate
    skipped = ", research skipped" if plan.skipped else ""
    head = (
        f"budget: ${plan.budget:,.2f}, estimate ${estimate.cost:,.2f}/run "
        f"({estimate.source.value}, {estimate.n_samples} samples), "
        f"N×K {plan.requested_n}×{plan.requested_k} → {plan.n}×{plan.k}, "
        f"projected ${plan.projected:,.2f}{skipped}"
    )
    return [head, *(f"  {reason}" for reason in plan.reasons)]


def _spend_line(spend: Spend) -> str:
    return (
        f"spend: ${spend.total:,.2f} over {spend.runs} runs, "
        f"unknown cost: {spend.unknown_runs}, over budget: {_OVER_BUDGET[spend.over_budget]}"
    )


def _verdict_line(verdict: RiskVerdict) -> str:
    if verdict.halt:
        return "halted"
    if verdict.approved:
        return "approved"
    return f"vetoed ({len(verdict.violations)} violations)"


def _settle_summary(report: SettlementReport, settlements: Path) -> str:
    """Outcome, new fills, pending, unsettled, unreadable files, breaks, marks and totals."""
    lines = [f"outcome: {report.outcome.value}", f"as_of: {report.as_of}"]
    lines += _section("new fills", [_fill_line(fill) for fill in report.new_fills])
    lines += _section(
        "pending",
        [f"{order.client_order_id} ({order.ticker}): {order.reason}" for order in report.pending],
    )
    lines += _section(
        "unsettled",
        [
            f"{order.client_order_id} ({order.ticker or '?'}): {order.reason}"
            for order in report.unsettled
        ],
    )
    lines += _section(
        "unreadable run files",
        [f"{name}: {reason}" for name, reason in report.unreadable_reports.items()],
    )
    lines += _section(
        "breaks",
        [f"{b.ticker}: ledger {b.ledger_shares}, broker {b.broker_shares}" for b in report.breaks],
    )
    lines += _section("open lots", [_lot_line(marked) for marked in report.open_lots])
    unmarked = sorted({lot.ticker for lot in report.unmarked})
    lines.append(f"unmarked: {', '.join(unmarked) if unmarked else 'none'}")
    lines.append(f"realised: {report.realised_total:,.2f}")
    lines.append(f"unrealised: {report.unrealised_total:,.2f}")
    lines.append(f"settlements: {settlements}")
    return "\n".join(lines)


def _section(title: str, rows: Sequence[str]) -> list[str]:
    return [f"{title}: {len(rows)}", *(f"  {row}" for row in rows)]


def _fill_line(fill: Fill) -> str:
    price = "" if fill.avg_price is None else f" @ {fill.avg_price:,.2f}"
    return (
        f"{fill.client_order_id}: {fill.status.value} "
        f"{fill.filled_shares}/{fill.requested_shares}{price}"
    )


def _lot_line(marked: MarkedLot) -> str:
    lot = marked.lot
    return (
        f"{lot.ticker} {lot.shares:+d} @ {lot.entry_price:,.2f}, close {marked.price:,.2f} "
        f"on {marked.close_date}: {marked.unrealised_pnl:+,.2f}"
    )


def _attribute_summary(report: AttributionReport, attributions: Path) -> str:
    """Outcome, window, totals by component and by sector, breaks, exclusions, the
    dividend note and the reports directory."""
    lines = [
        f"outcome: {report.outcome.value}",
        f"window: {report.start} to {report.end}",
        f"total: {_totals_line(report.total)}",
    ]
    lines += _section(
        "sectors", [f"{sector}: {_totals_line(t)}" for sector, t in report.by_sector.items()]
    )
    lines += _section(
        "breaks",
        [f"{b.ticker}: ledger {b.ledger_shares}, broker {b.broker_shares}" for b in report.breaks],
    )
    lines += _section("excluded", [f"{t}: {why}" for t, why in report.excluded.items()])
    lines += _section(
        "no beta",
        [
            f"{t}: {', '.join(day.isoformat() for day in days)}"
            for t, days in report.no_beta.items()
        ],
    )
    lines += _section("unregressed days", [day.isoformat() for day in report.unregressed_days])
    lines.append(f"note: {DIVIDEND_NOTE}")
    lines.append(f"attributions: {attributions}")
    return "\n".join(lines)


def _totals_line(totals: AttributionTotals) -> str:
    return (
        f"market {totals.market:+,.2f}, sector {totals.sector:+,.2f}, "
        f"residual {totals.residual:+,.2f}, trading {totals.trading:+,.2f}, "
        f"total {totals.total:+,.2f}"
    )


def _verify_summary(report: VerificationReport, verifications: Path) -> str:
    """Outcome, window, per horizon the counts, hit rate, mean excess and IC, the verdict
    buckets and the executed bucket; the reports directory. By-score and by-sector buckets
    are only in the JSON."""
    lines = [f"outcome: {report.outcome.value}", f"window: {report.start} to {report.end}"]
    for summary in report.summaries:
        lines += _horizon_lines(summary)
    lines.append(f"verifications: {verifications}")
    return "\n".join(lines)


def _horizon_lines(summary: HorizonSummary) -> list[str]:
    overall = summary.overall
    verdicts = "; ".join(
        f"{key} {_bucket_text(bucket)}" for key, bucket in summary.by_verdict.items()
    )
    return [
        f"{summary.horizon}d: scored {summary.scored}, pending {summary.pending}, "
        f"excluded {summary.excluded}; hit rate {_percent(overall.hit_rate)}, "
        f"mean excess {_excess(overall.mean_excess)}, ic {_ic(summary.ic)}",
        f"  verdicts: {verdicts or 'none'}",
        f"  executed: {_bucket_text(summary.executed)}",
    ]


def _bucket_text(bucket: Bucket) -> str:
    return f"n={bucket.n} hit {_percent(bucket.hit_rate)} excess {_excess(bucket.mean_excess)}"


def _percent(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def _excess(value: float | None) -> str:
    return "n/a" if value is None else f"{value:+.2%}"


def _ic(value: float | None) -> str:
    return "n/a" if value is None else f"{value:+.2f}"
