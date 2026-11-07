"""The backtest snapshot CLI: wiring, the summary, fail-closed roots and the source builder."""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner, Result

from hedge_fund import cli
from hedge_fund.adapters.data.edgar_provider import EdgarProvider
from hedge_fund.adapters.data.frozen_store import MANIFEST, load_store
from hedge_fund.adapters.data.sec_sources import SecLookups
from hedge_fund.adapters.data.yfinance_history import YFinanceHistory
from hedge_fund.domain.backtest import FlaggedSpan, MemberSpan
from hedge_fund.ports.data import DataProviderError
from hedge_fund.ports.fundamentals import FundamentalsProviderError
from hedge_fund.services.backtest_snapshot import SnapshotSources
from hedge_fund.universe import parse_universe
from tests.factories import make_facts
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


def _replace_history(sources: SnapshotSources, history: object) -> SnapshotSources:
    return SnapshotSources(
        history=history,  # type: ignore[arg-type]
        splits=sources.splits,
        company_ciks=sources.company_ciks,
        sic=sources.sic,
        fundamentals=sources.fundamentals,
    )
