"""The attribute CLI: flag wiring, the summary, exit codes and the adapter builder."""

from __future__ import annotations

import dataclasses
import json
import tomllib
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import click
import pytest
from typer.testing import CliRunner, Result

from hedge_fund import cli
from hedge_fund.adapters.attribution import JsonAttributionReportStore
from hedge_fund.adapters.broker import AlpacaPaperBroker
from hedge_fund.adapters.ledger import JsonFillStore, JsonOpeningBalance
from hedge_fund.domain.attribution import (
    NO_PNL,
    AttributeOutcome,
    AttributionReport,
    AttributionTotals,
)
from hedge_fund.domain.ledger import ReconciliationBreak
from hedge_fund.engine.portfolio.risk_model import RiskParams
from hedge_fund.services.attribute import AttributeDeps, AttributionSaveError
from hedge_fund.services.cycle import CycleParams
from hedge_fund.universe import Universe
from tests.factories import make_closes
from tests.fakes import FakeProvider, InMemoryFillStore, ScriptedBroker, StaticOpeningBalance

REPO = Path(__file__).resolve().parents[1]
NOW = datetime(2026, 9, 30, 22, 0, tzinfo=UTC)
runner = CliRunner()

BASE = AttributionReport(
    start=date(2026, 9, 25),
    end=date(2026, 9, 30),
    started_at=NOW,
    outcome=AttributeOutcome.CLEAN,
    failed_stage=None,
    error=None,
    days=(),
    by_ticker={},
    by_sector={},
    total=NO_PNL,
    breaks=(),
    no_beta={},
    unregressed_days=(),
    excluded={},
)
FULL = dataclasses.replace(
    BASE,
    outcome=AttributeOutcome.BREAK,
    by_sector={
        "Tech": AttributionTotals(15.0, 5.0, -2.5, 4.0, 21.5),
        "Unknown": AttributionTotals(0.0, 0.0, 1234.5, -10.0, 1224.5),
    },
    total=AttributionTotals(15.0, 5.0, 1232.0, -6.0, 1246.0),
    breaks=(ReconciliationBreak("KKK", 5, 7),),
    no_beta={"ZZZ": (date(2026, 9, 29), date(2026, 9, 30))},
    unregressed_days=(date(2026, 9, 28),),
    excluded={"GAP": "no close on 2026-09-30"},
)


def _invoke(*args: str) -> Result:
    return runner.invoke(cli.app, ["attribute", *args])


def _canned(
    monkeypatch: pytest.MonkeyPatch, outcome: AttributionReport | BaseException
) -> list[Any]:
    """Replace run_attribute: record its arguments, then return ``outcome`` or raise it."""
    calls: list[Any] = []

    def run_attribute(deps: AttributeDeps, **kwargs: Any) -> AttributionReport:
        calls.append((deps, kwargs))
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    monkeypatch.setattr(cli, "run_attribute", run_attribute)
    monkeypatch.setattr(cli, "build_attribute_deps", lambda root: root)
    return calls


# --- the command -------------------------------------------------------------------------


def test_attribute_is_a_named_subcommand_with_its_flags() -> None:
    assert "attribute" in runner.invoke(cli.app, ["--help"]).output
    # CI (GITHUB_ACTIONS, FORCE_COLOR) makes Rich style the help; compare the plain text.
    help_text = " ".join(click.unstyle(_invoke("--help").output).split())
    for flag in ("--start", "--end", "--root"):
        assert flag in help_text


def test_the_help_says_to_run_it_after_settle() -> None:
    help_text = " ".join(click.unstyle(_invoke("--help").output).split())
    assert "run it after settle" in help_text


def test_flags_reach_run_attribute(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls = _canned(monkeypatch, BASE)
    result = _invoke("--start", "2026-09-25", "--end", "2026-09-30", "--root", str(tmp_path))
    assert result.exit_code == 0, result.output
    [(deps, kwargs)] = calls
    assert deps == tmp_path
    assert kwargs == {"start": date(2026, 9, 25), "end": date(2026, 9, 30), "now": cli.clock}


def test_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _canned(monkeypatch, BASE)
    assert _invoke().exit_code == 0
    [(deps, kwargs)] = calls
    assert deps == Path("data")
    assert kwargs == {"start": None, "end": None, "now": cli.clock}


@pytest.mark.parametrize(
    ("outcome", "code"), [(AttributeOutcome.CLEAN, 0), (AttributeOutcome.BREAK, 2)]
)
def test_exit_codes(monkeypatch: pytest.MonkeyPatch, outcome: AttributeOutcome, code: int) -> None:
    _canned(monkeypatch, dataclasses.replace(BASE, outcome=outcome))
    result = _invoke()
    assert result.exit_code == code
    assert result.stdout.startswith(f"outcome: {outcome.value}\n")


def test_the_summary(monkeypatch: pytest.MonkeyPatch) -> None:
    _canned(monkeypatch, FULL)
    assert _invoke().stdout.splitlines() == [
        "outcome: break",
        "window: 2026-09-25 to 2026-09-30",
        "total: market +15.00, sector +5.00, residual +1,232.00, trading -6.00, total +1,246.00",
        "sectors: 2",
        "  Tech: market +15.00, sector +5.00, residual -2.50, trading +4.00, total +21.50",
        "  Unknown: market +0.00, sector +0.00, residual +1,234.50, trading -10.00, total +1,224.50",
        "breaks: 1",
        "  KKK: ledger 5, broker 7",
        "excluded: 1",
        "  GAP: no close on 2026-09-30",
        "no beta: 1",
        "  ZZZ: 2026-09-29, 2026-09-30",
        "unregressed days: 1",
        "  2026-09-28",
        "note: closes are dividend-adjusted and fill prices are not, so dollar P&L drifts "
        "slightly around ex-dividend dates",
        f"attributions: {Path('data') / 'attributions'}",
    ]


def test_an_empty_summary(monkeypatch: pytest.MonkeyPatch) -> None:
    _canned(monkeypatch, BASE)
    lines = _invoke().stdout.splitlines()
    for line in ("sectors: 0", "breaks: 0", "excluded: 0", "no beta: 0", "unregressed days: 0"):
        assert line in lines


@pytest.mark.parametrize(
    ("error", "message"),
    [
        pytest.param(
            ValueError("no SPY closes up to 2026-09-30"),
            f"ValueError: no SPY closes up to 2026-09-30; the partial report is under "
            f"{Path('data') / 'attributions'}",
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
    ("outcome", "code"), [(AttributeOutcome.CLEAN, 1), (AttributeOutcome.BREAK, 2)]
)
def test_an_unsaved_report_still_prints_the_summary(
    monkeypatch: pytest.MonkeyPatch, outcome: AttributeOutcome, code: int
) -> None:
    error = AttributionSaveError(dataclasses.replace(BASE, outcome=outcome))
    error.__cause__ = OSError("disk full")
    _canned(monkeypatch, error)
    result = _invoke()
    assert result.exit_code == code
    assert result.stdout.startswith(f"outcome: {outcome.value}\n")
    assert "the attribution report was not saved: OSError('disk full')" in result.stderr


def test_a_real_run_over_fakes_writes_an_attribution_report(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def build(root: Path) -> AttributeDeps:
        return AttributeDeps(
            broker=ScriptedBroker(),
            prices=FakeProvider(make_closes(["SPY"], n_days=80), {}),
            universe=Universe(name="test", list_date=date(2026, 9, 1), tickers=(), sectors={}),
            risk=RiskParams(lookback=30, min_obs=20),
            fills=InMemoryFillStore(),
            opening=StaticOpeningBalance(),
            report_store=JsonAttributionReportStore(root),
        )

    monkeypatch.setattr(cli, "build_attribute_deps", build)
    monkeypatch.setattr(cli, "clock", lambda: NOW)
    result = _invoke("--root", str(tmp_path))
    assert result.exit_code == 0, result.output
    assert result.stdout.splitlines()[-1] == f"attributions: {tmp_path / 'attributions'}"
    [path] = (tmp_path / "attributions").iterdir()
    assert path.name == "2026-09-30_2026-09-30_220000.json"
    body = json.loads(path.read_text(encoding="utf-8"))
    assert (body["outcome"], body["days"]) == ("clean", [])


# --- building the real adapters ----------------------------------------------------------


@pytest.fixture
def credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APCA_API_KEY_ID", "key")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "secret")


@pytest.mark.usefixtures("credentials")
def test_build_attribute_deps_wires_the_real_adapters(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    provider_calls: list[tuple[Path, str]] = []
    provider = object()

    def default_provider(cache_dir: Path, universe: str) -> object:
        provider_calls.append((cache_dir, universe))
        return provider

    monkeypatch.setattr(cli, "default_provider", default_provider)
    deps = cli.build_attribute_deps(tmp_path)
    assert isinstance(deps.broker, AlpacaPaperBroker)
    assert deps.prices is provider
    assert provider_calls == [(tmp_path / "cache" / "prices", "sp500")]
    assert deps.universe.name == "sp500"
    assert deps.risk == CycleParams().risk
    assert isinstance(deps.fills, JsonFillStore)
    assert isinstance(deps.opening, JsonOpeningBalance)
    assert isinstance(deps.report_store, JsonAttributionReportStore)


@pytest.mark.usefixtures("credentials")
@pytest.mark.parametrize("unset", ["APCA_API_KEY_ID", "APCA_API_SECRET_KEY"])
def test_missing_credentials_exit_1_before_any_read(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, unset: str
) -> None:
    monkeypatch.delenv(unset)
    calls: list[Any] = []
    monkeypatch.setattr(cli, "run_attribute", lambda *args, **kwargs: calls.append(args))
    result = _invoke("--root", str(tmp_path))
    assert result.exit_code == 1
    assert f"cannot start attribution: {unset} is not set" in result.stderr
    assert calls == []
    assert not (tmp_path / "attributions").exists()


def test_the_pixi_task_is_declared() -> None:
    pixi = tomllib.loads((REPO / "pixi.toml").read_text(encoding="utf-8"))
    assert pixi["tasks"]["attribute"] == "hedge-fund attribute"
