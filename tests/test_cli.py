"""The run-once CLI: flag wiring, the summary, exit codes and the adapter builder."""

from __future__ import annotations

import dataclasses
import json
import tomllib
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import click
import pytest
from typer.testing import CliRunner, Result

from hedge_fund import cli
from hedge_fund.adapters.account import JsonSnapshotStore
from hedge_fund.adapters.approval import ConsoleApprovalGate
from hedge_fund.adapters.broker import AlpacaPaperBroker
from hedge_fund.adapters.cost import load_prices
from hedge_fund.adapters.cycle import JsonCycleReportStore
from hedge_fund.adapters.research.store import JsonResearchStore
from hedge_fund.adapters.research.tradingagents import TradingAgentsResearcher
from hedge_fund.domain.cost import BudgetPlan, Estimate, EstimateSource, PriceTable, Spend
from hedge_fund.domain.cycle import CycleOptions, CycleOutcome, CycleReport
from hedge_fund.domain.execution import ExecutionReport, OrderRequest, SubmitResult, SubmitStatus
from hedge_fund.domain.orders import Order, OrderAction
from hedge_fund.domain.research import NameResearch, ResearchResult, RunRecord, Verdict
from hedge_fund.domain.risk import BreakerState, RiskBreakdown, RiskVerdict, Violation
from hedge_fund.domain.shortlist import Candidate, Side
from hedge_fund.services.cycle import CycleDeps, ReportSaveError
from tests.cycle_world import AS_OF, EVENING, SESSION, Clock, World

REPO = Path(__file__).resolve().parents[1]
NIGHT = datetime(2026, 10, 1, 2, 0, tzinfo=UTC)  # 2026-09-30 22:00 in New York
runner = CliRunner()

BASE = CycleReport(
    as_of=AS_OF,
    started_at=EVENING,
    dry_run=False,
    outcome=CycleOutcome.HALTED_PREFLIGHT,
    failed_stage=None,
    error=None,
    account=None,
    price_unresolved={},
    fundamentals_unresolved={},
    shortlist=None,
    research=None,
    research_reused=False,
    target=None,
    verdict=None,
    order_prices=None,
    plan=None,
    decision=None,
    execution=None,
    execution_interrupted=False,
    planned_client_order_ids=(),
)


def _use(monkeypatch: pytest.MonkeyPatch, world: World, now: datetime = NIGHT) -> list[Path]:
    """Swap the adapter builder for ``world`` (reports as JSON under --root, the price table
    from --prices) and the clock."""
    roots: list[Path] = []

    def build(root: Path, prices: Path) -> CycleDeps:
        roots.append(root)
        return dataclasses.replace(
            world.deps, report_store=JsonCycleReportStore(root), price_table=load_prices(prices)
        )

    monkeypatch.setattr(cli, "build_deps", build)
    monkeypatch.setattr(cli, "clock", Clock(now))
    return roots


def _invoke(*args: str) -> Result:
    return runner.invoke(cli.app, ["cycle", *args])


def _leg(status: SubmitStatus, reason: str | None = None) -> SubmitResult:
    order = Order(ticker="AAA", action=OrderAction.BUY, shares=1, ref_price=1.0, notional=1.0)
    request = OrderRequest(order=order, client_order_id="hf-2026-09-30-AAA-buy-1")
    return SubmitResult(request=request, status=status, broker_order_id=None, reason=reason)


def _submitted(*statuses: SubmitStatus) -> CycleReport:
    execution = ExecutionReport(
        as_of=AS_OF,
        blocked=None,
        results=tuple(_leg(status) for status in statuses),
        opening_halted=False,
    )
    return dataclasses.replace(BASE, outcome=CycleOutcome.SUBMITTED, execution=execution)


def _canned(monkeypatch: pytest.MonkeyPatch, outcome: CycleReport | BaseException) -> list[Any]:
    """Replace run_cycle: record its arguments, then return ``outcome`` or raise it."""
    calls: list[Any] = []

    def run_cycle(deps: CycleDeps, **kwargs: Any) -> CycleReport:
        calls.append((deps, kwargs))
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    monkeypatch.setattr(cli, "run_cycle", run_cycle)
    monkeypatch.setattr(cli, "build_deps", lambda root, prices: (root, prices))
    monkeypatch.setattr(cli, "clock", Clock(NIGHT))
    return calls


# --- the command -------------------------------------------------------------------------


def test_cycle_is_a_named_subcommand() -> None:
    result = runner.invoke(cli.app, ["--help"])
    assert result.exit_code == 0
    assert "cycle" in result.output
    # CI (GITHUB_ACTIONS, FORCE_COLOR) makes Rich style the help; compare the plain text.
    help_text = click.unstyle(runner.invoke(cli.app, ["cycle", "--help"]).output)
    for flag in (
        "--as-of",
        "--dry-run",
        "--fresh-research",
        "--any-time",
        "--k",
        "--budget",
        "--prices",
        "--root",
    ):
        assert flag in help_text


def test_flags_reach_run_cycle(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls = _canned(monkeypatch, BASE)
    args = ["--as-of", "2026-09-29", "--dry-run", "--fresh-research", "--any-time", "-k", "2"]
    budget = ["--budget", "12.5", "--prices", str(tmp_path / "p.toml")]
    result = _invoke(*args, *budget, "--root", str(tmp_path))
    assert result.exit_code == 0, result.output
    [(deps, kwargs)] = calls
    assert deps == (tmp_path, tmp_path / "p.toml")
    assert kwargs["as_of"] == date(2026, 9, 29)
    assert kwargs["today"] == AS_OF  # 02:00 UTC on 10-01 is still 09-30 in New York
    assert kwargs["options"] == CycleOptions(
        dry_run=True, fresh_research=True, k=2, any_time=True, budget=12.5
    )
    assert kwargs["now"] is cli.clock


def test_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _canned(monkeypatch, BASE)
    assert _invoke().exit_code == 0
    [(deps, kwargs)] = calls
    assert deps == (Path("data"), Path("data") / "prices.toml")
    assert (kwargs["as_of"], kwargs["today"]) == (AS_OF, AS_OF)
    assert kwargs["options"] == CycleOptions()


def test_k_must_be_at_least_one(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _canned(monkeypatch, BASE)
    assert _invoke("-k", "0").exit_code == 2
    assert calls == []


# --- exit codes and output ---------------------------------------------------------------


def test_a_dry_run_exits_0_and_writes_a_report(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    world = World()
    roots = _use(monkeypatch, world)
    result = _invoke("--dry-run", "--root", str(tmp_path))
    assert result.exit_code == 0, result.output
    assert roots == [tmp_path]
    lines = result.stdout.splitlines()
    assert lines[:2] == ["outcome: dry-run", "as_of: 2026-09-30"]
    assert "account: equity 1,000,000.00, 0 holdings" in lines
    assert "shortlist: 6 candidates" in lines
    assert "research: 6 names, 18 runs, error runs: 0" in lines
    assert "spend: $0.00 over 18 runs, unknown cost: 18, over budget: n/a" in lines
    assert not any(line.startswith("budget:") for line in lines)
    assert "verdict: approved" in lines
    assert "orders: 6" in lines
    assert lines[-1] == f"reports: {tmp_path / 'runs'}"
    [path] = (tmp_path / "runs").iterdir()
    assert path.name == "2026-09-30_020000.json"
    assert json.loads(path.read_text(encoding="utf-8"))["outcome"] == "dry-run"
    assert world.snapshot_store.saved == []


def test_a_submitted_cycle_exits_0_and_lists_every_leg(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    world = World()
    _use(monkeypatch, world)
    result = _invoke("--root", str(tmp_path))
    assert result.exit_code == 0, result.output
    assert result.stdout.startswith("outcome: submitted\n")
    for client_id in world.broker.submitted:
        assert f"  {client_id}: accepted" in result.stdout
    assert len(world.broker.submitted) == 6


@pytest.mark.parametrize(
    ("statuses", "code"),
    [
        pytest.param((SubmitStatus.ACCEPTED, SubmitStatus.DUPLICATE), 0, id="accepted-duplicate"),
        pytest.param((SubmitStatus.ACCEPTED, SubmitStatus.DEFERRED), 0, id="deferred"),
        pytest.param((SubmitStatus.REJECTED, SubmitStatus.SKIPPED), 2, id="rejected"),
        pytest.param((SubmitStatus.ACCEPTED, SubmitStatus.ERROR), 2, id="error"),
    ],
)
def test_failed_legs_exit_2(
    monkeypatch: pytest.MonkeyPatch, statuses: tuple[SubmitStatus, ...], code: int
) -> None:
    _canned(monkeypatch, _submitted(*statuses))
    result = _invoke()
    assert result.exit_code == code
    assert f"  hf-2026-09-30-AAA-buy-1: {statuses[-1].value}" in result.stdout


def test_inside_the_window_exits_1_with_the_opg_message(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _use(monkeypatch, World(), now=SESSION)
    result = _invoke("--root", str(tmp_path))
    assert result.exit_code == 1
    assert "opg orders must be submitted before 09:25 ET or after the close" in result.stderr
    assert "--any-time overrides this check" in result.stderr
    [path] = (tmp_path / "runs").iterdir()
    assert json.loads(path.read_text(encoding="utf-8"))["failed_stage"] == "submission_window"


def test_a_stale_as_of_exits_1(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    world = World()
    _use(monkeypatch, world)
    result = _invoke("--as-of", "2026-09-28", "--root", str(tmp_path))
    assert result.exit_code == 1
    assert "ValueError: as_of 2026-09-28 is not live" in result.stderr
    assert f"partial report is under {tmp_path / 'runs'}" in result.stderr
    assert world.snapshot_store.saved == []


def test_an_interrupt_exits_1(monkeypatch: pytest.MonkeyPatch) -> None:
    _canned(monkeypatch, KeyboardInterrupt())
    result = _invoke()
    assert result.exit_code == 1
    assert "interrupted" in result.stderr


@pytest.mark.parametrize(
    ("report", "code"),
    [
        pytest.param(BASE, 1, id="good-cycle"),
        pytest.param(_submitted(SubmitStatus.REJECTED), 2, id="failed-legs"),
    ],
)
def test_an_unsaved_report_still_prints_the_summary(
    monkeypatch: pytest.MonkeyPatch, report: CycleReport, code: int
) -> None:
    error = ReportSaveError(report)
    error.__cause__ = OSError("disk full")
    _canned(monkeypatch, error)
    result = _invoke()
    assert result.exit_code == code
    assert result.stdout.startswith(f"outcome: {report.outcome.value}\n")
    assert "the cycle report was not saved: OSError('disk full')" in result.stderr


def test_the_summary_counts_error_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    candidate = Candidate(ticker="AAA", sector="Tech", side=Side.LONG, depth=1, composite=1.0)
    runs = tuple(
        RunRecord("AAA", AS_OF, index, None, "", {}, 0.0, error)
        for index, error in enumerate(["RuntimeError: x", None, "RuntimeError: y"])
    )
    name = NameResearch(candidate, runs, 0, None, None, None, Verdict.INSUFFICIENT)
    research = ResearchResult(as_of=AS_OF, names=(name,))
    _canned(monkeypatch, dataclasses.replace(BASE, research=research, research_reused=True))
    result = _invoke()
    assert "research: 1 names, 3 runs, error runs: 2 (reused)" in result.stdout


@pytest.mark.parametrize(
    ("approved", "halt", "line"),
    [
        pytest.param(True, False, "verdict: approved", id="approved"),
        pytest.param(False, False, "verdict: vetoed (1 violations)", id="vetoed"),
        pytest.param(False, True, "verdict: halted", id="halted"),
    ],
)
def test_the_summary_names_the_verdict(
    monkeypatch: pytest.MonkeyPatch, approved: bool, halt: bool, line: str
) -> None:
    breakdown = RiskBreakdown(0.0, 0.0, 0.0, 0.0, {}, {}, ())
    breakers = BreakerState(0.0, 0.0, daily_tripped=halt, drawdown_tripped=False)
    violations = () if approved or halt else (Violation("gross", 2.5, 2.0, "sum of |weight|"),)
    verdict = RiskVerdict(AS_OF, approved, halt, violations, breakdown, breakers)
    _canned(monkeypatch, dataclasses.replace(BASE, verdict=verdict))
    assert line in _invoke().stdout.splitlines()


def test_a_budget_prints_the_plan_and_the_spend(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / "prices.toml").write_text("fallback_run_cost = 1.0\n", encoding="utf-8")
    _use(monkeypatch, World())
    result = _invoke("--dry-run", "--budget", "12", "--root", str(tmp_path))
    assert result.exit_code == 0, result.output
    lines = result.stdout.splitlines()
    assert "research: 6 names, 12 runs, error runs: 0" in lines
    assert (
        "budget: $12.00, estimate $1.00/run (fallback, 0 samples), N×K 6×3 → 6×2, projected $12.00"
    ) in lines
    assert "  k 3→2: 6×3×$1.00=$18.00 > $12.00" in lines
    assert "spend: $0.00 over 12 runs, unknown cost: 12, over budget: no" in lines


def test_a_budget_skip_ends_nothing_to_do_and_prints_why(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / "prices.toml").write_text("fallback_run_cost = 1.0\n", encoding="utf-8")
    world = World()
    _use(monkeypatch, world)
    result = _invoke("--budget", "1", "--root", str(tmp_path))
    assert result.exit_code == 0, result.output  # NOTHING_TO_DO exits 0, as it always has
    lines = result.stdout.splitlines()
    assert lines[0] == "outcome: nothing-to-do"
    assert (
        "budget: $1.00, estimate $1.00/run (fallback, 0 samples), N×K 6×3 → 0×2, "
        "projected $0.00, research skipped"
    ) in lines
    assert "  n 6→0: 6×2×$1.00=$12.00 > $1.00" in lines
    assert "spend: $0.00 over 0 runs, unknown cost: 0, over budget: no" in lines
    assert not any(line.startswith(("research:", "verdict:", "orders:")) for line in lines)
    assert world.broker.submitted == []
    assert world.gate.summaries == []


@pytest.mark.parametrize(("value", "code"), [("nan", 1), ("-1", 1), ("abc", 2)])
def test_a_bad_budget_fails_before_the_cycle_reads_the_broker(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, value: str, code: int
) -> None:
    world = World()
    _use(monkeypatch, world)
    result = _invoke("--dry-run", f"--budget={value}", "--root", str(tmp_path))
    assert result.exit_code == code, result.output
    if code == 1:  # check_budget fails at the inputs stage
        assert "BudgetError" in result.stderr
    assert world.broker.submitted == []


def test_a_budget_without_an_estimate_exits_1_with_a_budget_stage_report(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _use(monkeypatch, World())
    result = _invoke("--dry-run", "--budget", "5", "--root", str(tmp_path))
    assert result.exit_code == 1
    assert "BudgetError: no run cost history and no fallback_run_cost" in result.stderr
    [path] = (tmp_path / "runs").iterdir()
    assert json.loads(path.read_text(encoding="utf-8"))["failed_stage"] == "budget"


@pytest.mark.parametrize(
    ("budget", "spend", "lines"),
    [
        pytest.param(
            BudgetPlan(
                1.0,
                Estimate(0.8, EstimateSource.HISTORY, 30),
                40,
                3,
                0,
                2,
                0.0,
                True,
                ("k 3→2: 40×3×$0.80=$96.00 > $1.00", "n 40→0: 40×2×$0.80=$64.00 > $1.00"),
            ),
            Spend(0.0, 0, 0, (), False),
            [
                "budget: $1.00, estimate $0.80/run (history, 30 samples), N×K 40×3 → 0×2, "
                "projected $0.00, research skipped",
                "  k 3→2: 40×3×$0.80=$96.00 > $1.00",
                "  n 40→0: 40×2×$0.80=$64.00 > $1.00",
                "spend: $0.00 over 0 runs, unknown cost: 0, over budget: no",
            ],
            id="skipped",
        ),
        pytest.param(
            None,
            Spend(1234.5, 120, 3, (), True),
            ["spend: $1,234.50 over 120 runs, unknown cost: 3, over budget: yes"],
            id="over",
        ),
    ],
)
def test_the_summary_shows_the_plan_and_the_spend(
    monkeypatch: pytest.MonkeyPatch,
    budget: BudgetPlan | None,
    spend: Spend,
    lines: list[str],
) -> None:
    _canned(monkeypatch, dataclasses.replace(BASE, budget=budget, spend=spend))
    output = _invoke().stdout.splitlines()
    for line in lines:
        assert line in output


def test_the_clock_is_utc_now() -> None:
    before = datetime.now(UTC)
    now = cli.clock()
    assert before <= now <= datetime.now(UTC)
    assert now.utcoffset() == timedelta(0)


# --- building the real adapters ----------------------------------------------------------


@pytest.fixture
def credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APCA_API_KEY_ID", "key")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "secret")
    monkeypatch.setenv("SEC_USER_AGENT", "Test test@example.com")


@pytest.mark.usefixtures("credentials")
def test_build_deps_wires_the_real_adapters(tmp_path: Path) -> None:
    deps = cli.build_deps(tmp_path, tmp_path / "prices.toml")
    assert deps.price_table == PriceTable()
    assert deps.universe.name == "sp500"
    assert isinstance(deps.broker, AlpacaPaperBroker)
    assert isinstance(deps.researcher, TradingAgentsResearcher)
    assert isinstance(deps.research_store, JsonResearchStore)
    assert isinstance(deps.snapshot_store, JsonSnapshotStore)
    assert isinstance(deps.gate, ConsoleApprovalGate)
    assert isinstance(deps.report_store, JsonCycleReportStore)


@pytest.mark.usefixtures("credentials")
@pytest.mark.parametrize(
    ("unset", "message"),
    [
        pytest.param("APCA_API_KEY_ID", "APCA_API_KEY_ID is not set", id="alpaca-key"),
        pytest.param("APCA_API_SECRET_KEY", "APCA_API_SECRET_KEY is not set", id="alpaca-secret"),
        pytest.param("SEC_USER_AGENT", "SEC_USER_AGENT is not set", id="sec-user-agent"),
    ],
)
def test_missing_credentials_exit_1_before_any_read(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, unset: str, message: str
) -> None:
    monkeypatch.delenv(unset)
    calls = []
    monkeypatch.setattr(cli, "run_cycle", lambda *args, **kwargs: calls.append(args))
    result = _invoke("--root", str(tmp_path))
    assert result.exit_code == 1
    assert f"cannot start the cycle: {message}" in result.stderr
    assert calls == []
    assert not (tmp_path / "runs").exists()


@pytest.mark.usefixtures("credentials")
def test_a_malformed_price_table_exits_1_before_the_cycle(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / "prices.toml").write_text("not = [toml", encoding="utf-8")
    calls = []
    monkeypatch.setattr(cli, "run_cycle", lambda *args, **kwargs: calls.append(args))
    result = _invoke("--root", str(tmp_path))
    assert result.exit_code == 1
    assert "cannot start the cycle: " in result.stderr
    assert "malformed price file" in result.stderr
    assert calls == []
    assert not (tmp_path / "runs").exists()


def test_the_script_and_the_pixi_task_are_declared() -> None:
    pyproject = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    assert pyproject["project"]["scripts"] == {"hedge-fund": "hedge_fund.cli:app"}
    pixi = tomllib.loads((REPO / "pixi.toml").read_text(encoding="utf-8"))
    assert pixi["tasks"]["cycle"] == "hedge-fund cycle"
