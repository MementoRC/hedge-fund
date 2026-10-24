"""Run-once CLI: ``hedge-fund cycle`` runs one paper-trading cycle (Layer 6d).

This is the only composition root: ``build_deps`` builds the real adapters and ``clock``
is the only clock read. Tests replace both. Paper trading only.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, NoReturn

import typer

from hedge_fund.adapters.account import JsonSnapshotStore
from hedge_fund.adapters.approval import ConsoleApprovalGate
from hedge_fund.adapters.broker import AlpacaPaperBroker
from hedge_fund.adapters.cycle import JsonCycleReportStore
from hedge_fund.adapters.cycle.json_report import RUNS_DIR
from hedge_fund.adapters.data.factory import default_fundamentals_provider, default_provider
from hedge_fund.adapters.research.store import JsonResearchStore
from hedge_fund.adapters.research.tradingagents import ResearchConfig, TradingAgentsResearcher
from hedge_fund.domain.cycle import CycleOptions, CycleReport
from hedge_fund.domain.execution import SubmitStatus
from hedge_fund.domain.research import ResearchResult
from hedge_fund.domain.risk import RiskVerdict
from hedge_fund.engine.execution import MARKET_TZ
from hedge_fund.ports.broker import BrokerError
from hedge_fund.ports.fundamentals import FundamentalsProviderError
from hedge_fund.services.cycle import CycleDeps, ReportSaveError, SubmissionWindowError, run_cycle
from hedge_fund.universe import load_universe

UNIVERSE = "sp500"
_FAILED_LEGS = frozenset({SubmitStatus.ERROR, SubmitStatus.REJECTED})

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
