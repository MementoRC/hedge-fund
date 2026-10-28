"""The settle CLI: flag wiring, the summary, exit codes and the adapter builder."""

from __future__ import annotations

import dataclasses
import json
import tomllib
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import click
import pandas as pd
import pytest
from typer.testing import CliRunner, Result

from hedge_fund import cli
from hedge_fund.adapters.broker import AlpacaPaperBroker
from hedge_fund.adapters.cycle import JsonCycleReportStore
from hedge_fund.adapters.ledger import JsonFillStore, JsonOpeningBalance, JsonSettlementReportStore
from hedge_fund.domain.ledger import (
    Fill,
    FillStatus,
    Lot,
    MarkedLot,
    PendingOrder,
    ReconciliationBreak,
    SettlementReport,
    SettleOutcome,
    UnsettledOrder,
)
from hedge_fund.domain.orders import Holding, OrderAction
from hedge_fund.services.settle import SettleDeps, SettlementSaveError
from tests.fakes import (
    FakeProvider,
    InMemoryFillStore,
    InMemoryReportStore,
    ScriptedBroker,
    StaticOpeningBalance,
)

REPO = Path(__file__).resolve().parents[1]
AS_OF = date(2026, 10, 1)
MORNING = datetime(2026, 10, 1, 14, 0, tzinfo=UTC)
LATE = datetime(2026, 10, 2, 2, 0, tzinfo=UTC)  # 2026-10-01 22:00 in New York
OPEN = datetime(2026, 10, 1, 13, 30, tzinfo=UTC)
runner = CliRunner()

BASE = SettlementReport(
    as_of=AS_OF,
    started_at=MORNING,
    outcome=SettleOutcome.CLEAN,
    failed_stage=None,
    error=None,
    new_fills=(),
    pending=(),
    unreadable_reports={},
    unsettled=(),
    broker_positions=(),
    open_lots=(),
    unmarked=(),
    closed_lots=(),
    breaks=(),
    realised_total=0.0,
    unrealised_total=0.0,
)


def _fill(cid: str, ticker: str, filled: int, requested: int, status: FillStatus) -> Fill:
    return Fill(
        client_order_id=cid,
        broker_order_id=None if filled == 0 else "b-1",
        cycle_as_of=date(2026, 9, 30),
        ticker=ticker,
        action=OrderAction.BUY,
        requested_shares=requested,
        filled_shares=filled,
        avg_price=None if filled == 0 else 101.0,
        filled_at=None if filled == 0 else OPEN,
        status=status,
        recorded_at=MORNING,
    )


FULL = dataclasses.replace(
    BASE,
    outcome=SettleOutcome.UNSETTLED,
    new_fills=(
        _fill("hf-a", "AAA", 10, 10, FillStatus.FILLED),
        _fill("hf-n", "BBB", 0, 5, FillStatus.NOT_SUBMITTED),
    ),
    pending=(PendingOrder("hf-p", "PPP", "open at broker: new"),),
    unreadable_reports={"bad.json": "ValueError: x"},
    unsettled=(
        UnsettledOrder("hf-u", "UUU", "unrecognised broker status 'held'"),
        UnsettledOrder("hf-g", "", "no planned order for id"),
    ),
    broker_positions=(Holding("AAA", 10),),
    open_lots=(
        MarkedLot(Lot("AAA", 10, 101.0, OPEN, "hf-a"), 102.0, AS_OF, 10.0),
        MarkedLot(Lot("BBB", -5, 50.0, OPEN, "hf-b"), 49.0, AS_OF, 5.0),
    ),
    unmarked=(Lot("MMM", 3, 20.0, OPEN, "opening"), Lot("MMM", 1, 21.0, OPEN, "opening")),
    breaks=(ReconciliationBreak("KKK", 5, 7),),
    realised_total=1234.5,
    unrealised_total=-15.0,
)


def _invoke(*args: str) -> Result:
    return runner.invoke(cli.app, ["settle", *args])


def _canned(
    monkeypatch: pytest.MonkeyPatch, outcome: SettlementReport | BaseException
) -> list[Any]:
    """Replace run_settle: record its arguments, then return ``outcome`` or raise it."""
    calls: list[Any] = []

    def run_settle(deps: SettleDeps, **kwargs: Any) -> SettlementReport:
        calls.append((deps, kwargs))
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    monkeypatch.setattr(cli, "run_settle", run_settle)
    monkeypatch.setattr(cli, "build_settle_deps", lambda root: root)
    monkeypatch.setattr(cli, "clock", lambda: LATE)
    return calls


# --- the command -------------------------------------------------------------------------


def test_settle_is_a_named_subcommand_with_its_flags() -> None:
    assert "settle" in runner.invoke(cli.app, ["--help"]).output
    # CI (GITHUB_ACTIONS, FORCE_COLOR) makes Rich style the help; compare the plain text.
    help_text = click.unstyle(_invoke("--help").output)
    for flag in ("--as-of", "--root"):
        assert flag in help_text


def test_flags_reach_run_settle(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls = _canned(monkeypatch, BASE)
    result = _invoke("--as-of", "2026-09-30", "--root", str(tmp_path))
    assert result.exit_code == 0, result.output
    [(deps, kwargs)] = calls
    assert deps == tmp_path
    assert kwargs["as_of"] == date(2026, 9, 30)
    assert kwargs["today"] == AS_OF  # 02:00 UTC on 10-02 is still 10-01 in New York
    assert kwargs["now"] is cli.clock


def test_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _canned(monkeypatch, BASE)
    assert _invoke().exit_code == 0
    [(deps, kwargs)] = calls
    assert deps == Path("data")
    assert (kwargs["as_of"], kwargs["today"]) == (AS_OF, AS_OF)


@pytest.mark.parametrize(
    ("outcome", "code"),
    [
        (SettleOutcome.CLEAN, 0),
        (SettleOutcome.PENDING, 0),
        (SettleOutcome.BREAK, 2),
        (SettleOutcome.UNSETTLED, 1),
    ],
)
def test_exit_codes(monkeypatch: pytest.MonkeyPatch, outcome: SettleOutcome, code: int) -> None:
    _canned(monkeypatch, dataclasses.replace(BASE, outcome=outcome))
    result = _invoke()
    assert result.exit_code == code
    assert result.stdout.startswith(f"outcome: {outcome.value}\n")


def test_the_summary(monkeypatch: pytest.MonkeyPatch) -> None:
    _canned(monkeypatch, FULL)
    assert _invoke().stdout.splitlines() == [
        "outcome: unsettled",
        "as_of: 2026-10-01",
        "new fills: 2",
        "  hf-a: filled 10/10 @ 101.00",
        "  hf-n: not-submitted 0/5",
        "pending: 1",
        "  hf-p (PPP): open at broker: new",
        "unsettled: 2",
        "  hf-u (UUU): unrecognised broker status 'held'",
        "  hf-g (?): no planned order for id",
        "unreadable run files: 1",
        "  bad.json: ValueError: x",
        "breaks: 1",
        "  KKK: ledger 5, broker 7",
        "open lots: 2",
        "  AAA +10 @ 101.00, close 102.00 on 2026-10-01: +10.00",
        "  BBB -5 @ 50.00, close 49.00 on 2026-10-01: +5.00",
        "unmarked: MMM",
        "realised: 1,234.50",
        "unrealised: -15.00",
        f"settlements: {Path('data') / 'settlements'}",
    ]


def test_an_empty_summary(monkeypatch: pytest.MonkeyPatch) -> None:
    _canned(monkeypatch, BASE)
    lines = _invoke().stdout.splitlines()
    assert "new fills: 0" in lines
    assert "unmarked: none" in lines


@pytest.mark.parametrize(
    ("error", "message"),
    [
        pytest.param(
            ValueError("as_of 2026-09-28 is not live"),
            f"ValueError: as_of 2026-09-28 is not live; the partial report is under "
            f"{Path('data') / 'settlements'}",
            id="error",
        ),
        pytest.param(KeyboardInterrupt(), "interrupted", id="interrupt"),
    ],
)
def test_a_failed_run_exits_1(
    monkeypatch: pytest.MonkeyPatch, error: BaseException, message: str
) -> None:
    _canned(monkeypatch, error)
    result = _invoke()
    assert result.exit_code == 1
    assert message in result.stderr


@pytest.mark.parametrize(
    ("outcome", "code"),
    [
        (SettleOutcome.CLEAN, 1),
        (SettleOutcome.PENDING, 1),
        (SettleOutcome.BREAK, 2),
        (SettleOutcome.UNSETTLED, 1),
    ],
)
def test_an_unsaved_report_still_prints_the_summary(
    monkeypatch: pytest.MonkeyPatch, outcome: SettleOutcome, code: int
) -> None:
    error = SettlementSaveError(dataclasses.replace(BASE, outcome=outcome))
    error.__cause__ = OSError("disk full")
    _canned(monkeypatch, error)
    result = _invoke()
    assert result.exit_code == code
    assert result.stdout.startswith(f"outcome: {outcome.value}\n")
    assert "the settlement report was not saved: OSError('disk full')" in result.stderr


def test_a_real_run_over_fakes_writes_a_settlement_report(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def build(root: Path) -> SettleDeps:
        return SettleDeps(
            broker=ScriptedBroker(),
            prices=FakeProvider(pd.DataFrame(), {}),
            cycle_reports=InMemoryReportStore(),
            fills=InMemoryFillStore(),
            opening=StaticOpeningBalance(),
            report_store=JsonSettlementReportStore(root),
        )

    monkeypatch.setattr(cli, "build_settle_deps", build)
    monkeypatch.setattr(cli, "clock", lambda: MORNING)
    result = _invoke("--root", str(tmp_path))
    assert result.exit_code == 0, result.output
    assert result.stdout.splitlines()[-1] == f"settlements: {tmp_path / 'settlements'}"
    [path] = (tmp_path / "settlements").iterdir()
    assert path.name == "2026-10-01_140000.json"
    assert json.loads(path.read_text(encoding="utf-8"))["outcome"] == "clean"


# --- building the real adapters ----------------------------------------------------------


@pytest.fixture
def credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APCA_API_KEY_ID", "key")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "secret")


@pytest.mark.usefixtures("credentials")
def test_build_settle_deps_wires_the_real_adapters(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    provider_calls: list[tuple[Path, str]] = []
    provider = object()

    def default_provider(cache_dir: Path, universe: str) -> object:
        provider_calls.append((cache_dir, universe))
        return provider

    monkeypatch.setattr(cli, "default_provider", default_provider)
    deps = cli.build_settle_deps(tmp_path)
    assert isinstance(deps.broker, AlpacaPaperBroker)
    assert deps.prices is provider
    assert provider_calls == [(tmp_path / "cache" / "prices", "sp500")]
    assert isinstance(deps.cycle_reports, JsonCycleReportStore)
    assert isinstance(deps.fills, JsonFillStore)
    assert isinstance(deps.opening, JsonOpeningBalance)
    assert isinstance(deps.report_store, JsonSettlementReportStore)


@pytest.mark.usefixtures("credentials")
@pytest.mark.parametrize("unset", ["APCA_API_KEY_ID", "APCA_API_SECRET_KEY"])
def test_missing_credentials_exit_1_before_any_read(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, unset: str
) -> None:
    monkeypatch.delenv(unset)
    calls: list[Any] = []
    monkeypatch.setattr(cli, "run_settle", lambda *args, **kwargs: calls.append(args))
    result = _invoke("--root", str(tmp_path))
    assert result.exit_code == 1
    assert f"cannot start settlement: {unset} is not set" in result.stderr
    assert calls == []
    assert not (tmp_path / "settlements").exists()


def test_the_pixi_task_is_declared() -> None:
    pixi = tomllib.loads((REPO / "pixi.toml").read_text(encoding="utf-8"))
    assert pixi["tasks"]["settle"] == "hedge-fund settle"
