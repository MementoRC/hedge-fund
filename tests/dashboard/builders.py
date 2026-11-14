"""Hand-built roots, states and AppTest helpers for the dashboard tests."""

from __future__ import annotations

import dataclasses
import json
import shutil
from collections.abc import Callable
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from streamlit.testing.v1 import AppTest

from hedge_fund.adapters.demo_marker import DEMO_MARKER
from hedge_fund.dashboard.data import DEFAULT_ROOT, RootState
from hedge_fund.domain.cycle import CycleReport
from hedge_fund.viz.load import Workspace
from hedge_fund.viz.views import BacktestRun
from tests.viz.builders import (
    DAY0,
    make_cycle,
    make_nav,
    make_position,
    make_snapshots,
    make_target,
)

SYNTHETIC = Path("synthetic")
NAV_VALUES = (100.0, 120.0, 90.0, 115.0)  # 2024-01-02 .. 2024-01-05


def make_workspace(**changes: Any) -> Workspace:
    base = Workspace(
        root=SYNTHETIC,
        demo=None,
        snapshots=(),
        cycles=(),
        research=(),
        settlements=(),
        attributions=(),
        verifications=(),
        backtests=(),
        benchmark=None,
        unreadable=(),
    )
    return dataclasses.replace(base, **changes)


def make_state(**changes: Any) -> RootState:
    """An existing, unclassified root holding what ``changes`` give the workspace."""
    return RootState(
        root=SYNTHETIC, workspace=make_workspace(**changes), kind=None, kind_error=None
    )


def missing_state() -> RootState:
    return RootState(root=Path(DEFAULT_ROOT), workspace=None, kind=None, kind_error=None)


def benchmark_state(*closes: float, cycles: tuple[CycleReport, ...] = ()) -> RootState:
    """NAV 100, 120, 90, 115 from 2024-01-02, and SPY ``closes`` from the second NAV date
    (no benchmark when there are none).
    """
    benchmark = make_nav(*closes, start=DAY0 + timedelta(days=1)) if closes else None
    return make_state(snapshots=make_snapshots(*NAV_VALUES), benchmark=benchmark, cycles=cycles)


def targeted_state(*closes: float) -> RootState:
    """``benchmark_state`` plus two cycles with a target each (AAA long, BBB short)."""
    first = make_target(
        DAY0, make_position("AAA", "Tech", 0.30), make_position("BBB", "Energy", -0.10)
    )
    later = DAY0 + timedelta(days=2)
    second = make_target(
        later, make_position("AAA", "Tech", 0.20), make_position("BBB", "Energy", -0.10)
    )
    cycles = (make_cycle(DAY0, target=first), make_cycle(later, target=second))
    return benchmark_state(*closes, cycles=cycles)


def flat_state() -> RootState:
    """A cycle whose target holds no position: exposure points, no sectors, no positions."""
    return benchmark_state(cycles=(make_cycle(DAY0, target=make_target(DAY0)),))


def make_run(**changes: Any) -> BacktestRun:
    base = BacktestRun(
        run_id="run-1",
        start=date(2024, 1, 10),
        end=date(2024, 3, 8),
        trading_days=40,
        rebalances=3,
        total_return=0.10,
        benchmark_return=0.05,
        excess_return=0.05,
        max_drawdown=0.20,
        sharpe=1.5,
        beta=0.25,
        breaker_days=2,
        biases=(),
    )
    return dataclasses.replace(base, **changes)


def copy_root(demo_root: Path, tmp_path: Path, name: str) -> Path:
    root = tmp_path / name
    shutil.copytree(demo_root, root)
    return root


def backtest_root(demo_root: Path, tmp_path: Path) -> Path:
    """The demo run without its marker: a plain backtest root."""
    root = copy_root(demo_root, tmp_path, "backtest")
    (root / DEMO_MARKER).unlink()
    return root


def mismatched_root(demo_root: Path, tmp_path: Path) -> Path:
    """A root whose valid-format marker names another run: ``Workspace.demo`` is set, yet
    ``classify_root`` refuses it.
    """
    root = copy_root(demo_root, tmp_path, "mismatched")
    marker = json.loads((root / DEMO_MARKER).read_text(encoding="utf-8"))
    marker["run_id"] = "other-run"
    (root / DEMO_MARKER).write_text(json.dumps(marker), encoding="utf-8")
    return root


def open_page(script: Callable[..., None], root: Path | None = None, *args: Any) -> AppTest:
    """Run ``script`` (a function from ``tests/dashboard/scripts.py``) once, then point the
    sidebar root picker at ``root`` and run again.
    """
    at = AppTest.from_function(script, default_timeout=60, args=args or None).run()
    if root is not None:
        at.sidebar.text_input(key="root").set_value(str(root)).run()
    return at
