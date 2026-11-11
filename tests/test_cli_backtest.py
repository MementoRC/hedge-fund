"""The backtest snapshot CLI: wiring, the summary, fail-closed roots and the source builder."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner, Result

from hedge_fund import cli
from hedge_fund.adapters.account import JsonSnapshotStore
from hedge_fund.adapters.approval.auto import AutoApprovalGate
from hedge_fund.adapters.broker import AlpacaPaperBroker
from hedge_fund.adapters.broker.simulated import SimulatedBroker
from hedge_fund.adapters.cycle import JsonCycleReportStore
from hedge_fund.adapters.data.edgar_provider import EdgarProvider
from hedge_fund.adapters.data.frozen_providers import FrozenDataProvider, FrozenFundamentalsProvider
from hedge_fund.adapters.data.frozen_store import MANIFEST, load_store, write_store
from hedge_fund.adapters.data.sec_sources import SecLookups
from hedge_fund.adapters.data.yfinance_history import YFinanceHistory
from hedge_fund.adapters.research.neutral import NeutralResearcher
from hedge_fund.adapters.research.store import JsonResearchStore
from hedge_fund.domain.backtest import (
    RESIDUAL_BIASES,
    BacktestMetrics,
    BacktestReport,
    FlaggedSpan,
    Honesty,
    MemberSpan,
    Performance,
    SimCosts,
    TradingCosts,
)
from hedge_fund.domain.shortlist import Shortlist
from hedge_fund.ports.data import DataProviderError
from hedge_fund.ports.fundamentals import FundamentalsProviderError
from hedge_fund.services.backtest import BacktestDeps
from hedge_fund.services.backtest_snapshot import SnapshotSources
from hedge_fund.services.cycle import CycleParams
from hedge_fund.universe import parse_universe
from tests.factories import make_facts, make_frozen_data
from tests.fakes import FakeFundamentalsProvider

NOW = datetime(2024, 1, 16, 22, 0, tzinfo=UTC)  # 17:00 in New York: yesterday is 2024-01-15
DAYS = pd.bdate_range("2024-01-02", periods=10)  # 2024-01-02 .. 2024-01-15
MEMBERSHIP = "ticker,start,end\nAAA,2020-01-02,\nBBB,2020-01-02,2024-01-08\n"
FLAGGED = (
    "ticker,start,end,reason\n"
    "AAA,2023-06-01,2024-01-05,ticker reused\n"
    "ZZZ,2015-01-01,2016-01-01,outside the window\n"
)
UNIVERSE = (
    "# list_date: 2024-01-01\n"
    "ticker,gics_sector,cik\n"
    "AAA,Energy,0000000001\n"
    "BBB,Financials,0000000002\n"
)
runner = CliRunner()


def _history(ticker: str, start: date, end: date) -> pd.DataFrame | None:
    if ticker == "BBB":
        return None
    return pd.DataFrame({"Open": 99.0, "Close": 100.0, "Dividends": 0.0}, index=DAYS)


def _sources() -> SnapshotSources:
    return SnapshotSources(
        history=_history,
        splits=lambda tickers: (),
        company_ciks=lambda: {},
        sic=lambda cik: None,
        fundamentals=lambda ciks: FakeFundamentalsProvider(make_facts(list(ciks))),
    )


def _invoke(*args: str) -> Result:
    return runner.invoke(cli.app, ["backtest", "snapshot", *args])


@pytest.fixture
def membership(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Fake vendors, a two-name universe and the fixed clock; returns the membership file."""
    monkeypatch.setattr(cli, "build_snapshot_sources", _sources)
    monkeypatch.setattr(cli, "load_universe", lambda name: parse_universe(name, UNIVERSE))
    monkeypatch.setattr(cli, "clock", lambda: NOW)
    path = tmp_path / "membership.csv"
    path.write_text(MEMBERSHIP, encoding="utf-8")
    return path


def test_snapshot_writes_a_loadable_store_and_prints_the_counts(
    membership: Path, tmp_path: Path
) -> None:
    root = tmp_path / "store"
    result = _invoke("--root", str(root), "--start", "2024-01-02", "--membership", str(membership))
    assert result.exit_code == 0, result.output
    data = load_store(root)
    assert (data.start, data.end) == (date(2024, 1, 2), date(2024, 1, 15))
    assert data.unpriced == ("BBB",)
    lines = result.stdout.splitlines()
    assert lines[0].startswith(f"frozen store {root}: 2024-01-02 to 2024-01-15, manifest sha256 ")
    assert lines[1] == "members 2, priced 2 (SPY included), unpriced 1"
    assert lines[2] == f"splits 0, dividends 0, fact rows {len(data.facts)}"
    assert lines[3] == "no sector 0, sector from SIC 0, no CIK 0"


def test_end_bounds_the_store(membership: Path, tmp_path: Path) -> None:
    root = tmp_path / "store"
    args = ["--root", str(root), "--start", "2024-01-02", "--end", "2024-01-10"]
    result = _invoke(*args, "--membership", str(membership))
    assert result.exit_code == 0, result.output
    assert load_store(root).end == date(2024, 1, 10)


def test_the_packaged_membership_is_the_default(
    membership: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    names: list[str] = []

    def resource_text(name: str) -> str:
        names.append(name)
        return FLAGGED if name == "sp500_flagged_spans" else MEMBERSHIP

    monkeypatch.setattr(cli, "resource_text", resource_text)
    result = _invoke("--root", str(tmp_path / "store"), "--start", "2024-01-02")
    assert result.exit_code == 0, result.output
    assert names == ["sp500_membership", "sp500_flagged_spans"]


def test_the_flagged_spans_reach_the_written_store(
    membership: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "resource_text", lambda name: FLAGGED)
    root = tmp_path / "store"
    result = _invoke("--root", str(root), "--start", "2024-01-02", "--membership", str(membership))
    assert result.exit_code == 0, result.output
    expected = FlaggedSpan(MemberSpan("AAA", date(2023, 6, 1), date(2024, 1, 5)), "ticker reused")
    assert load_store(root).flagged == (expected,)


def test_a_start_before_the_membership_floor_is_refused(membership: Path, tmp_path: Path) -> None:
    root = tmp_path / "store"
    result = _invoke("--root", str(root), "--start", "2007-12-31", "--membership", str(membership))
    assert result.exit_code == 1
    assert "cannot start the snapshot" in result.stderr
    assert "2008-01-01" in result.stderr
    assert not root.exists()


def test_a_start_on_the_membership_floor_is_accepted(membership: Path, tmp_path: Path) -> None:
    root = tmp_path / "store"
    result = _invoke("--root", str(root), "--start", "2008-01-01", "--membership", str(membership))
    assert result.exit_code == 0, result.output


def test_a_non_empty_root_is_refused_before_any_fetch(
    membership: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "store"
    root.mkdir()
    (root / "old.txt").write_text("x", encoding="utf-8")
    calls: list[int] = []
    monkeypatch.setattr(cli, "build_snapshot_sources", lambda: calls.append(1))
    result = _invoke("--root", str(root), "--membership", str(membership))
    assert result.exit_code == 1
    assert "cannot start the snapshot" in result.stderr
    assert "not empty" in result.stderr
    assert calls == []


def test_a_missing_membership_file_exits_1(membership: Path, tmp_path: Path) -> None:
    result = _invoke("--root", str(tmp_path / "store"), "--membership", str(tmp_path / "none"))
    assert result.exit_code == 1
    assert "cannot start the snapshot" in result.stderr


def test_a_missing_sec_user_agent_exits_1_before_any_fetch(
    membership: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "build_snapshot_sources", cli_build_snapshot_sources)
    monkeypatch.delenv("SEC_USER_AGENT", raising=False)
    result = _invoke("--root", str(tmp_path / "store"), "--membership", str(membership))
    assert result.exit_code == 1
    assert "SEC_USER_AGENT is not set" in result.stderr


def test_a_vendor_failure_exits_1_and_leaves_no_manifest(
    membership: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def failing(ticker: str, start: date, end: date) -> pd.DataFrame | None:
        raise DataProviderError("yahoo is down")

    monkeypatch.setattr(
        cli, "build_snapshot_sources", lambda: _replace_history(_sources(), failing)
    )
    root = tmp_path / "store"
    result = _invoke("--root", str(root), "--membership", str(membership))
    assert result.exit_code == 1
    assert "DataProviderError: yahoo is down" in result.stderr
    assert not (root / MANIFEST).exists()


def test_an_interrupt_exits_1(
    membership: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def interrupted(ticker: str, start: date, end: date) -> pd.DataFrame | None:
        raise KeyboardInterrupt

    monkeypatch.setattr(
        cli, "build_snapshot_sources", lambda: _replace_history(_sources(), interrupted)
    )
    result = _invoke("--root", str(tmp_path / "store"), "--membership", str(membership))
    assert result.exit_code == 1
    assert "interrupted" in result.stderr


def test_build_snapshot_sources_wires_the_real_vendors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SEC_USER_AGENT", "Test test@example.com")
    sources = cli_build_snapshot_sources()
    assert isinstance(sources.history, YFinanceHistory)
    assert sources.fundamentals is EdgarProvider
    assert isinstance(sources.company_ciks.__self__, SecLookups)  # type: ignore[attr-defined]
    assert isinstance(sources.sic.__self__, SecLookups)  # type: ignore[attr-defined]


def test_build_snapshot_sources_needs_the_sec_user_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SEC_USER_AGENT", raising=False)
    with pytest.raises(FundamentalsProviderError, match="SEC_USER_AGENT"):
        cli_build_snapshot_sources()


cli_build_snapshot_sources = cli.build_snapshot_sources


def _write(root: Path) -> Path:
    write_store(root, make_frozen_data(), created_at=NOW)
    return root


def _run(*args: str) -> Result:
    return runner.invoke(cli.app, ["backtest", "run", *args])


def _performance(cagr: float) -> Performance:
    return Performance(
        start=date(2024, 1, 1),
        end=date(2024, 2, 9),
        total_return=0.01,
        cagr=cagr,
        volatility=0.2,
        sharpe=None,
        max_drawdown=-0.05,
        peak=date(2024, 1, 2),
        trough=date(2024, 1, 3),
    )


def _report(run_id: str = "2024-01-01_2024-02-09_abc123def456") -> BacktestReport:
    honesty = Honesty(
        price_unresolved={"no closes": 2},
        fundamentals_unresolved=3,
        flagged_excluded=1,
        stopped_trading=4,
        rejects={"no cash": 5},
        daily_breaker_days=6,
        drawdown_breaker_days=7,
        outcomes={"submitted": 2},
    )
    metrics = BacktestMetrics(
        strategy=_performance(0.11),
        benchmark=_performance(0.22),
        beta=None,
        costs=TradingCosts(traded_notional=1000.0, turnover=1.5, slippage=2.0, borrow=3.0),
        honesty=honesty,
    )
    return BacktestReport(
        run_id=run_id,
        store_digest="d",
        start=date(2024, 1, 1),
        end=date(2024, 2, 9),
        cash=100000.0,
        costs=SimCosts(),
        trading_days=30,
        rebalances=2,
        metrics=metrics,
    )


@pytest.fixture
def stubbed(monkeypatch: pytest.MonkeyPatch) -> list[BacktestDeps]:
    seen: list[BacktestDeps] = []

    def fake(deps: BacktestDeps, *, start: date, end: date) -> BacktestReport:
        seen.append(deps)
        return dataclasses.replace(_report(), start=start, end=end)

    monkeypatch.setattr(cli, "run_backtest", fake)
    return seen


def test_run_prints_the_summary_and_writes_the_report(
    stubbed: list[BacktestDeps], tmp_path: Path
) -> None:
    store, root = _write(tmp_path / "store"), tmp_path / "run"
    result = _run("--store", str(store), "--root", str(root), "--start", "2024-01-01")
    assert result.exit_code == 0, result.output
    report_file = root / "backtest" / "2024-01-01_2024-02-09_abc123def456.json"
    assert report_file.exists()
    out = result.stdout
    assert "run 2024-01-01_2024-02-09_abc123def456" in out
    assert "2024-01-01 to 2024-02-09, 30 trading days, 2 rebalances" in out
    assert "first rebalance" not in out
    assert "CAGR" in out
    assert "no closes 2" in out
    assert RESIDUAL_BIASES[0] in out
    assert str(report_file) in out
    assert stubbed[0].cash == 100000.0
    assert stubbed[0].data.end == date(2024, 2, 9)


def test_run_notes_a_mid_month_start(stubbed: list[BacktestDeps], tmp_path: Path) -> None:
    store = _write(tmp_path / "store")
    args = ["--store", str(store), "--root", str(tmp_path / "run"), "--start", "2024-01-10"]
    result = _run(*args)
    assert result.exit_code == 0, result.output
    assert "first rebalance on 2024-02-01" in result.stdout


def test_run_a_digest_mismatch_exits_1(stubbed: list[BacktestDeps], tmp_path: Path) -> None:
    store = _write(tmp_path / "store")
    victim = next(p for p in sorted(store.rglob("*")) if p.is_file() and p.name != MANIFEST)
    victim.write_bytes(victim.read_bytes() + b"x")
    result = _run("--store", str(store), "--root", str(tmp_path / "run"))
    assert result.exit_code == 1
    assert "digest" in result.stderr
    assert stubbed == []


def test_run_a_missing_manifest_exits_1(stubbed: list[BacktestDeps], tmp_path: Path) -> None:
    (tmp_path / "store").mkdir()
    result = _run("--store", str(tmp_path / "store"), "--root", str(tmp_path / "run"))
    assert result.exit_code == 1
    assert "error:" in result.stderr


def test_run_a_non_empty_root_exits_1(stubbed: list[BacktestDeps], tmp_path: Path) -> None:
    store, root = _write(tmp_path / "store"), tmp_path / "run"
    root.mkdir()
    (root / "old.txt").write_text("x", encoding="utf-8")
    result = _run("--store", str(store), "--root", str(root))
    assert result.exit_code == 1
    assert "not empty" in result.stderr


def test_run_a_root_inside_the_store_exits_1(stubbed: list[BacktestDeps], tmp_path: Path) -> None:
    store = _write(tmp_path / "store")
    result = _run("--store", str(store), "--root", str(store / "inner"))
    assert result.exit_code == 1
    assert "inside" in result.stderr


def test_run_the_store_itself_as_root_exits_1(stubbed: list[BacktestDeps], tmp_path: Path) -> None:
    store = _write(tmp_path / "store")
    result = _run("--store", str(store), "--root", str(store))
    assert result.exit_code == 1
    assert stubbed == []


def test_run_a_negative_slippage_exits_1(stubbed: list[BacktestDeps], tmp_path: Path) -> None:
    store = _write(tmp_path / "store")
    args = ["--store", str(store), "--root", str(tmp_path / "run"), "--slippage-bps", "-1"]
    result = _run(*args)
    assert result.exit_code == 1
    assert "slippage_bps" in result.stderr


def _clean_failure(result: Result) -> None:
    assert result.exit_code == 1, result.output
    assert result.stderr.startswith("error:")
    assert "Traceback" not in result.output
    assert not isinstance(result.exception, (KeyError, AttributeError, TypeError))


@pytest.mark.parametrize(
    ("flag", "value", "fragment"),
    [
        ("--cash", "0", "cash"),
        ("--cash", "-5", "cash"),
        ("--borrow-rate", "-0.01", "borrow_rate"),
    ],
)
def test_run_a_bad_option_exits_1_cleanly(
    stubbed: list[BacktestDeps], tmp_path: Path, flag: str, value: str, fragment: str
) -> None:
    store = _write(tmp_path / "store")
    args = ["--store", str(store), "--root", str(tmp_path / "run"), "--start", "2024-01-01"]
    result = _run(*args, flag, value)
    _clean_failure(result)
    assert fragment in result.stderr
    assert stubbed == []


def test_run_an_end_before_the_start_exits_1_cleanly(tmp_path: Path) -> None:
    store = _write(tmp_path / "store")
    args = ["--store", str(store), "--root", str(tmp_path / "run"), "--start", "2024-02-01"]
    result = _run(*args, "--end", "2024-01-15")
    _clean_failure(result)


def test_run_an_end_after_the_store_end_exits_1_cleanly(tmp_path: Path) -> None:
    store = _write(tmp_path / "store")
    args = ["--store", str(store), "--root", str(tmp_path / "run"), "--start", "2024-01-01"]
    result = _run(*args, "--end", "2024-02-10")
    _clean_failure(result)
    assert "outside the store" in result.stderr


def _corrupt_manifest(store: Path, corruption: str) -> None:
    """Break a store the manifest-key check and the digest check both let through."""
    path = store / MANIFEST
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if corruption == "coverage":
        manifest["coverage"] = {}
    elif corruption == "entry":
        del manifest["files"]["opens"]["sha256"]
    else:
        entry = manifest["files"]["facts"]
        body = b"not parquet"
        digest = hashlib.sha256(body).hexdigest()
        name = f"facts_{digest[:12]}.parquet"
        (store / name).write_bytes(body)
        entry.update(name=name, sha256=digest)
    path.write_text(json.dumps(manifest), encoding="utf-8")


@pytest.mark.parametrize("corruption", ["coverage", "entry", "parquet"])
def test_run_a_corrupt_store_exits_1_cleanly(
    stubbed: list[BacktestDeps], tmp_path: Path, corruption: str
) -> None:
    store = _write(tmp_path / "store")
    _corrupt_manifest(store, corruption)
    result = _run("--store", str(store), "--root", str(tmp_path / "run"))
    _clean_failure(result)
    assert "cannot start the backtest" in result.stderr
    assert stubbed == []


def test_run_a_start_before_the_floor_exits_1_from_the_real_guard(tmp_path: Path) -> None:
    store = _write(tmp_path / "store")
    result = _run("--store", str(store), "--root", str(tmp_path / "run"), "--start", "2007-12-31")
    assert result.exit_code == 1
    assert "2008-01-01" in result.stderr


def test_run_the_default_start_outside_the_store_exits_1(tmp_path: Path) -> None:
    store = _write(tmp_path / "store")
    result = _run("--store", str(store), "--root", str(tmp_path / "run"))
    assert result.exit_code == 1
    assert "outside the store" in result.stderr


def test_run_an_unexpected_failure_exits_1(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(deps: BacktestDeps, *, start: date, end: date) -> BacktestReport:
        raise RuntimeError("kaput")

    monkeypatch.setattr(cli, "run_backtest", boom)
    store = _write(tmp_path / "store")
    result = _run("--store", str(store), "--root", str(tmp_path / "run"), "--start", "2024-01-01")
    assert result.exit_code == 1
    assert "RuntimeError: kaput" in result.stderr


def test_build_backtest_deps_is_simulated_only_and_rooted_at_root(tmp_path: Path) -> None:
    data = make_frozen_data()
    deps = cli.build_backtest_deps(
        tmp_path, data, store_digest="d", cash=5000.0, costs=SimCosts(slippage_bps=1.0)
    )
    assert type(deps.broker) is SimulatedBroker
    assert not isinstance(deps.broker, AlpacaPaperBroker)
    assert deps.cash == 5000.0
    assert deps.costs == SimCosts(slippage_bps=1.0)
    assert isinstance(deps.snapshot_store, JsonSnapshotStore)
    assert isinstance(deps.report_store, JsonCycleReportStore)
    assert isinstance(deps.research_store, JsonResearchStore)
    assert isinstance(deps.gate, AutoApprovalGate)
    assert isinstance(deps.fundamentals, FrozenFundamentalsProvider)
    assert deps.params == CycleParams()
    day = date(2024, 1, 10)
    provider = deps.prices(day)
    assert isinstance(provider, FrozenDataProvider)
    assert provider.as_of == day
    shortlist = Shortlist(as_of=day, candidates=(), trimmed={})
    assert isinstance(deps.researcher(shortlist), NeutralResearcher)


def _replace_history(sources: SnapshotSources, history: object) -> SnapshotSources:
    return SnapshotSources(
        history=history,  # type: ignore[arg-type]
        splits=sources.splits,
        company_ciks=sources.company_ciks,
        sic=sources.sic,
        fundamentals=sources.fundamentals,
    )
