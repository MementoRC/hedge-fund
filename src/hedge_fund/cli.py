"""Run-once CLI: ``hedge-fund cycle`` runs one paper-trading cycle (Layer 6d),
``hedge-fund settle`` records what it filled (Layer 7a) and ``hedge-fund attribute`` splits
the book's P&L (Layer 7b).

This is the only composition root: ``build_deps``, ``build_settle_deps`` and
``build_attribute_deps`` build the real adapters and ``clock`` is the only clock read.
Tests replace them. Paper trading only.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, NoReturn

import typer

from hedge_fund.adapters.account import JsonSnapshotStore
from hedge_fund.adapters.approval import ConsoleApprovalGate
from hedge_fund.adapters.attribution import JsonAttributionReportStore
from hedge_fund.adapters.attribution.json_report import ATTRIBUTIONS_DIR
from hedge_fund.adapters.broker import AlpacaPaperBroker
from hedge_fund.adapters.cycle import JsonCycleReportStore
from hedge_fund.adapters.cycle.json_report import RUNS_DIR
from hedge_fund.adapters.data.factory import default_fundamentals_provider, default_provider
from hedge_fund.adapters.ledger import (
    JsonFillStore,
    JsonOpeningBalance,
    JsonSettlementReportStore,
)
from hedge_fund.adapters.ledger.settlement_report import SETTLEMENTS_DIR
from hedge_fund.adapters.research.store import JsonResearchStore
from hedge_fund.adapters.research.tradingagents import ResearchConfig, TradingAgentsResearcher
from hedge_fund.domain.attribution import (
    DIVIDEND_NOTE,
    AttributeOutcome,
    AttributionReport,
    AttributionTotals,
)
from hedge_fund.domain.cycle import CycleOptions, CycleReport
from hedge_fund.domain.execution import SubmitStatus
from hedge_fund.domain.ledger import Fill, MarkedLot, SettlementReport, SettleOutcome
from hedge_fund.domain.research import ResearchResult
from hedge_fund.domain.risk import RiskVerdict
from hedge_fund.engine.execution import MARKET_TZ
from hedge_fund.ports.broker import BrokerError
from hedge_fund.ports.fundamentals import FundamentalsProviderError
from hedge_fund.services.attribute import AttributeDeps, AttributionSaveError, run_attribute
from hedge_fund.services.cycle import (
    CycleDeps,
    CycleParams,
    ReportSaveError,
    SubmissionWindowError,
    run_cycle,
)
from hedge_fund.services.settle import SettleDeps, SettlementSaveError, run_settle
from hedge_fund.universe import load_universe

UNIVERSE = "sp500"
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

app = typer.Typer(add_completion=False)


@app.callback()
def main() -> None:
    """Hedge-fund paper-trading commands."""


def clock() -> datetime:
    """The wall clock in UTC."""
    return datetime.now(UTC)


def build_deps(root: Path) -> CycleDeps:
    """The real adapters, with every file under ``root``.

    The broker and the fundamentals provider are built first, so missing credentials
    (BrokerError, FundamentalsProviderError) fail before anything is read.
    """
    broker = AlpacaPaperBroker()
    fundamentals = default_fundamentals_provider(root / "cache" / "fundamentals", UNIVERSE)
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


@app.command()
def cycle(
    as_of: Annotated[
        datetime | None,
        typer.Option("--as-of", formats=["%Y-%m-%d"], help="Default: today in New York."),
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
    root: Annotated[Path, typer.Option("--root", help="Data root.")] = Path("data"),
) -> None:
    """Run one paper-trading cycle, from data to execution, and save its report."""
    runs = root / RUNS_DIR
    try:
        deps = build_deps(root)
    except (BrokerError, FundamentalsProviderError, ValueError) as exc:
        _fail(f"cannot start the cycle: {exc}", exc)
    today = clock().astimezone(MARKET_TZ).date()
    options = CycleOptions(dry_run=dry_run, fresh_research=fresh_research, k=k, any_time=any_time)
    day = today if as_of is None else as_of.date()
    try:
        report = run_cycle(deps, as_of=day, today=today, options=options, now=clock)
    except ReportSaveError as exc:
        typer.echo(_summary(exc.report, runs))
        typer.echo(f"error: the cycle report was not saved: {exc.__cause__!r}", err=True)
        raise typer.Exit(max(_exit_code(exc.report), 1)) from exc
    except SubmissionWindowError as exc:
        _fail(f"{exc}; --any-time overrides this check", exc)
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


def _fail(message: str, cause: BaseException) -> NoReturn:
    typer.echo(f"error: {message}", err=True)
    raise typer.Exit(1) from cause


def _exit_code(report: CycleReport) -> int:
    """2 when a submitted leg was ERROR or REJECTED, else 0."""
    results = () if report.execution is None else report.execution.results
    return 2 if any(result.status in _FAILED_LEGS for result in results) else 0


def _summary(report: CycleReport, runs: Path) -> str:
    """Outcome, account, shortlist, research, verdict, plan and legs; the reports directory."""
    lines = [f"outcome: {report.outcome.value}", f"as_of: {report.as_of}"]
    if report.account is not None:
        holdings = len(report.account.holdings)
        lines.append(f"account: equity {report.account.equity:,.2f}, {holdings} holdings")
    if report.shortlist is not None:
        lines.append(f"shortlist: {len(report.shortlist.candidates)} candidates")
    if report.research is not None:
        lines.append(_research_line(report.research, reused=report.research_reused))
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
