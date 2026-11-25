"""The verify CLI: flag wiring, the summary, exit codes and the adapter builder."""

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
from hedge_fund.adapters.ledger import JsonFillStore
from hedge_fund.adapters.research.store import JsonResearchStore
from hedge_fund.adapters.verification import JsonVerificationReportStore
from hedge_fund.domain.research import NameResearch, ResearchResult, Verdict
from hedge_fund.domain.shortlist import Candidate, Side
from hedge_fund.domain.verification import (
    HORIZONS,
    Bucket,
    HorizonSummary,
    VerificationReport,
    VerifyOutcome,
)
from hedge_fund.engine.verification import summarise
from hedge_fund.services.verify import VerificationSaveError, VerifyDeps
from tests.factories import make_closes
from tests.fakes import FakeProvider, InMemoryFillStore, InMemoryResearchStore

REPO = Path(__file__).resolve().parents[1]
NOW = datetime(2026, 9, 30, 22, 0, tzinfo=UTC)
runner = CliRunner()

ZEROED = tuple(summarise([], horizon) for horizon in HORIZONS)
BASE = VerificationReport(
    start=date(2026, 9, 1),
    end=date(2026, 9, 30),
    started_at=NOW,
    outcome=VerifyOutcome.CLEAN,
    failed_stage=None,
    error=None,
    observations=(),
    summaries=ZEROED,
)
FIVE = HorizonSummary(
    horizon=5,
    scored=4,
    pending=1,
    excluded=2,
    overall=Bucket("all", 4, 2, 2 / 3, 0.005, -0.01),
    by_verdict={
        "confirm": Bucket("confirm", 2, 1, 0.5, 0.03, 0.03),
        "neutral": Bucket("neutral", 1, 0, None, 0.02, 0.02),
        "contradict": Bucket("contradict", 1, 1, 1.0, -0.06, -0.06),
    },
    by_score={},
    by_sector={},
    executed=Bucket("executed", 2, 2, 1.0, 0.02, 0.02),
    ic=0.8,
    no_call_mean_abs_excess=0.02,
)
FULL = dataclasses.replace(BASE, outcome=VerifyOutcome.GAPS, summaries=(FIVE, ZEROED[1], ZEROED[2]))
EMPTY_HORIZON = [
    "{h}d: scored 0, pending 0, excluded 0; hit rate n/a, mean excess n/a, ic n/a",
    "  verdicts: none",
    "  executed: n=0 hit n/a excess n/a",
]


def _invoke(*args: str) -> Result:
    return runner.invoke(cli.app, ["verify", *args])


def _canned(
    monkeypatch: pytest.MonkeyPatch, outcome: VerificationReport | BaseException
) -> list[Any]:
    """Replace run_verify: record its arguments, then return ``outcome`` or raise it."""
    calls: list[Any] = []

    def run_verify(deps: VerifyDeps, **kwargs: Any) -> VerificationReport:
        calls.append((deps, kwargs))
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    monkeypatch.setattr(cli, "run_verify", run_verify)
    monkeypatch.setattr(cli, "build_verify_deps", lambda root: root)
    return calls


def _name(ticker: str) -> NameResearch:
    return NameResearch(
        candidate=Candidate(ticker, "Tech", Side.LONG, 1, 0.5),
        runs=(),
        valid=3,
        mean=1.0,
        std=0.0,
        agreement=1.0,
        verdict=Verdict.CONFIRM,
    )


def _over_fakes(monkeypatch: pytest.MonkeyPatch, research: InMemoryResearchStore) -> None:
    def build(root: Path) -> VerifyDeps:
        return VerifyDeps(
            research=research,
            fills=InMemoryFillStore(),
            data=FakeProvider(make_closes(["SPY"], n_days=80), {}),
            reports=JsonVerificationReportStore(root),
        )

    monkeypatch.setattr(cli, "build_verify_deps", build)
    monkeypatch.setattr(cli, "clock", lambda: NOW)


# --- the command -------------------------------------------------------------------------


def test_verify_is_a_named_subcommand_with_its_flags() -> None:
    assert "verify" in runner.invoke(cli.app, ["--help"]).output
    # CI (GITHUB_ACTIONS, FORCE_COLOR) makes Rich style the help; compare the plain text.
    help_text = " ".join(click.unstyle(_invoke("--help").output).split())
    for flag in ("--start", "--end", "--root"):
        assert flag in help_text


def test_flags_reach_run_verify(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls = _canned(monkeypatch, BASE)
    result = _invoke("--start", "2026-09-01", "--end", "2026-09-30", "--root", str(tmp_path))
    assert result.exit_code == 0, result.output
    [(deps, kwargs)] = calls
    assert deps == tmp_path
    assert kwargs == {"start": date(2026, 9, 1), "end": date(2026, 9, 30), "now": cli.clock}


def test_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _canned(monkeypatch, BASE)
    assert _invoke().exit_code == 0
    [(deps, kwargs)] = calls
    assert deps == Path("data")
    assert kwargs == {"start": None, "end": None, "now": cli.clock}


@pytest.mark.parametrize(
    ("outcome", "code"),
    [(VerifyOutcome.CLEAN, 0), (VerifyOutcome.GAPS, 2), (VerifyOutcome.FAILED, 1)],
)
def test_exit_codes(monkeypatch: pytest.MonkeyPatch, outcome: VerifyOutcome, code: int) -> None:
    _canned(monkeypatch, dataclasses.replace(BASE, outcome=outcome))
    result = _invoke()
    assert result.exit_code == code
    assert result.stdout.startswith(f"outcome: {outcome.value}\n")


def test_the_summary(monkeypatch: pytest.MonkeyPatch) -> None:
    _canned(monkeypatch, FULL)
    assert _invoke().stdout.splitlines() == [
        "outcome: gaps",
        "window: 2026-09-01 to 2026-09-30",
        "5d: scored 4, pending 1, excluded 2; hit rate 66.7%, mean excess +0.50%, ic +0.80",
        "  verdicts: confirm n=2 hit 50.0% excess +3.00%; neutral n=1 hit n/a excess +2.00%; "
        "contradict n=1 hit 100.0% excess -6.00%",
        "  executed: n=2 hit 100.0% excess +2.00%",
        *[line.format(h=21) for line in EMPTY_HORIZON],
        *[line.format(h=63) for line in EMPTY_HORIZON],
        f"verifications: {Path('data') / 'verifications'}",
    ]


def test_an_empty_summary_prints_every_horizon(monkeypatch: pytest.MonkeyPatch) -> None:
    _canned(monkeypatch, BASE)
    lines = _invoke().stdout.splitlines()
    for horizon in HORIZONS:
        assert EMPTY_HORIZON[0].format(h=horizon) in lines


@pytest.mark.parametrize(
    ("error", "message"),
    [
        pytest.param(
            ValueError("no SPY closes up to 2026-09-30"),
            f"ValueError: no SPY closes up to 2026-09-30; the partial report is under "
            f"{Path('data') / 'verifications'}",
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


@pytest.mark.parametrize(("outcome", "code"), [(VerifyOutcome.CLEAN, 1), (VerifyOutcome.GAPS, 2)])
def test_an_unsaved_report_still_prints_the_summary(
    monkeypatch: pytest.MonkeyPatch, outcome: VerifyOutcome, code: int
) -> None:
    error = VerificationSaveError(dataclasses.replace(BASE, outcome=outcome))
    error.__cause__ = OSError("disk full")
    _canned(monkeypatch, error)
    result = _invoke()
    assert result.exit_code == code
    assert result.stdout.startswith(f"outcome: {outcome.value}\n")
    assert "the verification report was not saved: OSError('disk full')" in result.stderr


def test_a_cannot_start_error_exits_1(monkeypatch: pytest.MonkeyPatch) -> None:
    def build(root: Path) -> VerifyDeps:
        raise ValueError("bad universe")

    monkeypatch.setattr(cli, "build_verify_deps", build)
    result = _invoke()
    assert result.exit_code == 1
    assert "cannot start verification: bad universe" in result.stderr


def test_a_bad_date_is_a_usage_error_and_runs_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _canned(monkeypatch, BASE)
    assert _invoke("--start", "09/01/2026").exit_code == 2
    assert _invoke("--end", "tomorrow").exit_code == 2
    assert calls == []


def test_a_real_run_with_no_research_writes_a_clean_report(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _over_fakes(monkeypatch, InMemoryResearchStore())
    result = _invoke("--root", str(tmp_path))
    assert result.exit_code == 0, result.output
    assert result.stdout.splitlines()[-1] == f"verifications: {tmp_path / 'verifications'}"
    [path] = (tmp_path / "verifications").iterdir()
    assert path.name == "none_2026-09-30_220000.json"
    body = json.loads(path.read_text(encoding="utf-8"))
    assert (body["outcome"], body["observations"]) == ("clean", [])


def test_a_real_run_with_a_name_without_a_price_exits_2(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    research = InMemoryResearchStore([ResearchResult(date(2026, 9, 1), (_name("ZZZ"),))])
    _over_fakes(monkeypatch, research)
    result = _invoke("--root", str(tmp_path))
    assert result.exit_code == 2, result.output
    assert result.stdout.startswith("outcome: gaps\n")
    [path] = (tmp_path / "verifications").iterdir()
    body = json.loads(path.read_text(encoding="utf-8"))
    assert [(row["horizon"], row["status"]) for row in body["observations"]] == [
        (5, "excluded"),
        (21, "excluded"),
        (63, "pending"),
    ]


def test_a_start_after_the_end_exits_1_and_saves_a_failed_report(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _over_fakes(monkeypatch, InMemoryResearchStore())
    result = _invoke("--start", "2026-09-30", "--end", "2026-09-01", "--root", str(tmp_path))
    assert result.exit_code == 1
    assert (
        "ValueError: start 2026-09-30 is after end 2026-09-01; the partial report is under "
        f"{tmp_path / 'verifications'}"
    ) in result.stderr
    [path] = (tmp_path / "verifications").iterdir()
    assert path.name == "2026-09-30_2026-09-01_220000.json"
    body = json.loads(path.read_text(encoding="utf-8"))
    assert (body["outcome"], body["failed_stage"]) == ("failed", "load")


# --- building the real adapters ----------------------------------------------------------


def test_build_verify_deps_wires_the_real_adapters(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    provider_calls: list[tuple[Path, str]] = []
    provider = object()

    def default_provider(cache_dir: Path, universe: str) -> object:
        provider_calls.append((cache_dir, universe))
        return provider

    monkeypatch.setattr(cli, "default_provider", default_provider)
    deps = cli.build_verify_deps(tmp_path)
    assert isinstance(deps.research, JsonResearchStore)
    assert isinstance(deps.fills, JsonFillStore)
    assert deps.data is provider
    assert provider_calls == [(tmp_path / "cache" / "prices", "sp500")]
    assert isinstance(deps.reports, JsonVerificationReportStore)


def test_the_pixi_task_is_declared() -> None:
    pixi = tomllib.loads((REPO / "pixi.toml").read_text(encoding="utf-8"))
    assert pixi["tasks"]["verify"] == "hedge-fund verify"
