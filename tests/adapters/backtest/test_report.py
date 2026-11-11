"""The backtest summary writer."""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import pytest

from hedge_fund.adapters.backtest.report import save_backtest_report
from hedge_fund.domain.backtest import RESIDUAL_BIASES, BacktestReport, SimCosts
from hedge_fund.domain.risk import NavPoint
from hedge_fund.engine.backtest.metrics import backtest_metrics

D0 = date(2024, 1, 1)


def _report() -> BacktestReport:
    curve = tuple(
        NavPoint(date=D0 + timedelta(days=i), nav=v)
        for i, v in enumerate((100.0, 120.0, 96.0, 115.2))
    )
    return BacktestReport(
        run_id="2024-01-01_2024-01-04_0123456789ab",
        store_digest="d" * 64,
        start=curve[0].date,
        end=curve[-1].date,
        cash=100_000.0,
        costs=SimCosts(),
        trading_days=4,
        rebalances=1,
        metrics=backtest_metrics(curve, curve, (), (), flagged=()),
    )


def test_save_writes_the_report_under_backtest(tmp_path: Path) -> None:
    report = _report()

    path = save_backtest_report(tmp_path, report)

    assert path == tmp_path / "backtest" / f"{report.run_id}.json"
    body = json.loads(path.read_text(encoding="utf-8"))
    assert body["run_id"] == report.run_id
    assert body["start"] == "2024-01-01"
    assert body["metrics"]["strategy"]["cagr"] == pytest.approx(report.metrics.strategy.cagr)
    assert body["biases"] == list(RESIDUAL_BIASES)


def test_a_second_save_of_the_same_run_raises(tmp_path: Path) -> None:
    report = _report()
    save_backtest_report(tmp_path, report)

    with pytest.raises(FileExistsError, match=report.run_id):
        save_backtest_report(tmp_path, report)
