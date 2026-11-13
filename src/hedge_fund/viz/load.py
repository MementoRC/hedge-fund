"""Read a store root into domain objects, and say what kind of root it is.

The only module outside ``cli.py`` that imports ``adapters/`` (read side only, never a
broker). Unreadable files are collected as warning strings and never raised, so a page can
degrade instead of crash.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from enum import StrEnum
from functools import partial
from pathlib import Path

from hedge_fund.adapters.account import JsonSnapshotStore
from hedge_fund.adapters.account.json_store import ACCOUNT_DIR
from hedge_fund.adapters.attribution import JsonAttributionReportStore
from hedge_fund.adapters.attribution.json_report import ATTRIBUTIONS_DIR
from hedge_fund.adapters.backtest.report import BACKTEST_DIR, load_backtest_reports
from hedge_fund.adapters.cycle import JsonCycleReportStore
from hedge_fund.adapters.cycle.json_report import RUNS_DIR
from hedge_fund.adapters.data.frozen_store import MANIFEST, load_store
from hedge_fund.adapters.demo_marker import DEMO_MARKER, demo_store_dir, read_demo_marker
from hedge_fund.adapters.ledger import JsonSettlementReportStore
from hedge_fund.adapters.ledger.settlement_report import SETTLEMENTS_DIR
from hedge_fund.adapters.research.store import RESEARCH_DIR, JsonResearchStore
from hedge_fund.adapters.verification import JsonVerificationReportStore
from hedge_fund.adapters.verification.json_report import VERIFICATIONS_DIR
from hedge_fund.domain.account import AccountSnapshot
from hedge_fund.domain.attribution import AttributionReport
from hedge_fund.domain.backtest import BacktestReport
from hedge_fund.domain.cycle import CycleReport
from hedge_fund.domain.demo import DemoMarker
from hedge_fund.domain.ledger import ReportLoad, SettlementReport
from hedge_fund.domain.loaded import Loaded
from hedge_fund.domain.research import ResearchResult
from hedge_fund.domain.risk import NavHistory
from hedge_fund.domain.verification import VerificationReport
from hedge_fund.engine.account.history import nav_history
from hedge_fund.engine.backtest.benchmark import benchmark_curve


class RootKind(StrEnum):
    DEMO = "demo"
    BACKTEST = "backtest"


class RootError(ValueError):
    """A root that is neither a demo root nor a backtest run root."""


@dataclass(frozen=True)
class Workspace:
    """Everything a page reads from one root, plus what could not be read.

    ``benchmark`` is SPY on the strategy NAV dates, read from the frozen store; None when no
    store is known, it does not load, or its manifest digest is not the ``store_digest`` of a
    backtest report in the root (then there is a warning). ``unreadable`` holds one warning per bad file (or per
    directory, for stores that fail as a whole): ``"<where>: <reason>"``. ``demo`` is None for
    both "no marker" and "invalid marker"; decide demo vs real only via ``classify_root``.
    """

    root: Path
    demo: DemoMarker | None
    snapshots: tuple[AccountSnapshot, ...]
    cycles: tuple[CycleReport, ...]
    research: tuple[ResearchResult, ...]
    settlements: tuple[SettlementReport, ...]
    attributions: tuple[AttributionReport, ...]
    verifications: tuple[VerificationReport, ...]
    backtests: tuple[BacktestReport, ...]
    benchmark: NavHistory | None
    unreadable: tuple[str, ...]


def load_workspace(root: Path, *, store: Path | None = None) -> Workspace:
    """Load ``root``; ``store`` (a frozen store) feeds the benchmark and defaults, for a demo
    root, to the sibling ``<root>.store`` its marker implies. A backtest root has a benchmark
    only when ``store`` is given and verifies against the report's ``store_digest``.
    """
    root = Path(root)
    problems: list[str] = []
    marker = _attempt(DEMO_MARKER, partial(read_demo_marker, root), None, problems)
    no_snapshots: tuple[AccountSnapshot, ...] = ()
    snapshots = _attempt(ACCOUNT_DIR, JsonSnapshotStore(root).snapshots, no_snapshots, problems)
    backtests = _loaded(BACKTEST_DIR, partial(load_backtest_reports, root), problems)
    if store is None and marker is not None:
        store = demo_store_dir(root)
    return Workspace(
        root=root,
        demo=marker,
        snapshots=snapshots,
        cycles=_cycles(root, problems),
        research=_research(root, problems),
        settlements=_loaded(SETTLEMENTS_DIR, JsonSettlementReportStore(root).load_all, problems),
        attributions=_loaded(ATTRIBUTIONS_DIR, JsonAttributionReportStore(root).load_all, problems),
        verifications=_loaded(
            VERIFICATIONS_DIR, JsonVerificationReportStore(root).load_all, problems
        ),
        backtests=backtests,
        benchmark=_benchmark(snapshots, backtests, store, problems),
        unreadable=tuple(problems),
    )


def classify_root(root: Path) -> RootKind:
    """DEMO or BACKTEST, else RootError (fail closed).

    DEMO: ``DEMO.json`` parses and its ``run_id`` and ``store_digest`` are the single backtest
    summary's.
    BACKTEST: exactly one summary that decodes, whose file name is ``<run_id>.json``, and no
    ``DEMO.json``. Anything else, a live root included, raises.
    """
    root = Path(root)
    try:
        marker = read_demo_marker(root)
    except ValueError as exc:
        raise RootError(str(exc)) from exc
    runs = load_backtest_reports(root)
    if len(runs.items) != 1 or runs.unreadable:
        raise RootError(
            f"{root}: expected exactly one readable backtest summary, found "
            f"{len(runs.items)} and {len(runs.unreadable)} unreadable"
        )
    run_id = runs.items[0].run_id
    if runs.names[0] != f"{run_id}.json":
        raise RootError(f"{root}: summary file {runs.names[0]!r} is not named for run {run_id!r}")
    if marker is None:
        return RootKind.BACKTEST
    if marker.run_id != runs.items[0].run_id:
        raise RootError(
            f"{root}: {DEMO_MARKER} names run {marker.run_id!r}, the summary is "
            f"{runs.items[0].run_id!r}"
        )
    if marker.store_digest != runs.items[0].store_digest:
        raise RootError(
            f"{root}: {DEMO_MARKER} store_digest {marker.store_digest!r} is not the summary's "
            f"{runs.items[0].store_digest!r}"
        )
    return RootKind.DEMO


def _attempt[T](label: str, load: Callable[[], T], fallback: T, problems: list[str]) -> T:
    """``load()``, or ``fallback`` with a warning: a page must never crash on a bad store."""
    try:
        return load()
    except Exception as exc:  # any reader failure (corrupt parquet, bad JSON) is a warning
        problems.append(f"{label}: {type(exc).__name__}: {exc}")
        return fallback


def _loaded[T](label: str, load: Callable[[], Loaded[T]], problems: list[str]) -> tuple[T, ...]:
    empty: Loaded[T] = Loaded((), {})
    result = _attempt(label, load, empty, problems)
    problems.extend(f"{label}/{name}: {reason}" for name, reason in result.unreadable.items())
    return result.items


def _cycles(root: Path, problems: list[str]) -> tuple[CycleReport, ...]:
    loaded = _attempt(RUNS_DIR, JsonCycleReportStore(root).load_all, ReportLoad((), {}), problems)
    problems.extend(f"{RUNS_DIR}/{name}: {reason}" for name, reason in loaded.unreadable.items())
    return loaded.reports


def _research(root: Path, problems: list[str]) -> tuple[ResearchResult, ...]:
    store = JsonResearchStore(root)
    no_days: tuple[date, ...] = ()
    days = _attempt(RESEARCH_DIR, store.dates, no_days, problems)
    found = (
        _attempt(f"{RESEARCH_DIR}/{day}", partial(store.load, day), None, problems) for day in days
    )
    return tuple(result for result in found if result is not None)


def _benchmark(
    snapshots: tuple[AccountSnapshot, ...],
    backtests: tuple[BacktestReport, ...],
    store: Path | None,
    problems: list[str],
) -> NavHistory | None:
    """SPY from ``store``, but only a store whose manifest digest is a run's ``store_digest``."""
    if store is None or not snapshots:
        return None

    def curve() -> NavHistory:
        digest = hashlib.sha256((store / MANIFEST).read_bytes()).hexdigest()
        if digest not in {report.store_digest for report in backtests}:
            raise ValueError(
                f"manifest digest {digest[:12]} is not the store of any run in the root"
            )
        nav = nav_history(snapshots, as_of=max(snapshot.as_of for snapshot in snapshots))
        return benchmark_curve(load_store(store), [point.date for point in nav], nav[-1].date)

    return _attempt(f"store {store.name}", curve, None, problems)
