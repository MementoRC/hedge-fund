"""The backtest summary writer."""

from __future__ import annotations

import dataclasses
import json
from datetime import date, timedelta
from pathlib import Path

import pytest

from hedge_fund.adapters.backtest.report import load_backtest_reports, save_backtest_report
from hedge_fund.domain.backtest import RESIDUAL_BIASES, BacktestReport, SimCosts
from hedge_fund.domain.risk import NavPoint
from hedge_fund.engine.backtest.metrics import backtest_metrics

D0 = date(2024, 1, 1)


def _curve(*values: float) -> tuple[NavPoint, ...]:
    return tuple(NavPoint(date=D0 + timedelta(days=i), nav=v) for i, v in enumerate(values))


def _report() -> BacktestReport:
    """Strategy and benchmark curves differ and the Honesty mappings are non-empty, so a
    swapped or dropped field cannot round-trip by accident.
    """
    curve = _curve(100.0, 120.0, 96.0, 115.2)
    benchmark = _curve(50.0, 51.0, 49.5, 52.0)
    metrics = backtest_metrics(curve, benchmark, (), (), flagged=())
    honesty = dataclasses.replace(
        metrics.honesty,
        price_unresolved={"no close": 2, "stale": 1},
        rejects={"gross": 3},
        outcomes={"submitted": 4, "blocked": 1},
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
        metrics=dataclasses.replace(metrics, honesty=honesty),
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


def test_load_backtest_reports_round_trips_a_saved_summary(tmp_path: Path) -> None:
    report = _report()
    save_backtest_report(tmp_path, report)

    loaded = load_backtest_reports(tmp_path)

    assert loaded.items == (report,)
    assert loaded.unreadable == {}


def test_the_round_trip_keeps_distinct_curves_and_the_honesty_mappings(tmp_path: Path) -> None:
    report = _report()
    save_backtest_report(tmp_path, report)

    [loaded] = load_backtest_reports(tmp_path).items

    assert loaded.metrics.strategy != loaded.metrics.benchmark
    assert loaded.metrics.honesty.price_unresolved == {"no close": 2, "stale": 1}
    assert loaded.metrics.honesty.rejects == {"gross": 3}
    assert loaded.metrics.honesty.outcomes == {"submitted": 4, "blocked": 1}
    assert loaded == report


def test_the_round_trip_keeps_a_missing_sharpe_and_beta(tmp_path: Path) -> None:
    report = _report()
    metrics = dataclasses.replace(
        report.metrics,
        strategy=dataclasses.replace(report.metrics.strategy, sharpe=None),
        benchmark=dataclasses.replace(report.metrics.benchmark, sharpe=None),
        beta=None,
    )
    report = dataclasses.replace(report, metrics=metrics)
    save_backtest_report(tmp_path, report)

    [loaded] = load_backtest_reports(tmp_path).items

    assert loaded.metrics.strategy.sharpe is None
    assert loaded.metrics.beta is None
    assert loaded == report


def test_a_truncated_summary_is_unreadable_and_the_good_one_still_loads(tmp_path: Path) -> None:
    good = save_backtest_report(tmp_path, _report())
    text = good.read_text(encoding="utf-8")
    (tmp_path / "backtest" / "cut.json").write_text(text[: len(text) // 2], encoding="utf-8")

    loaded = load_backtest_reports(tmp_path)

    assert len(loaded.items) == 1
    assert list(loaded.unreadable) == ["cut.json"]


@pytest.mark.parametrize("key", ["run_id", "metrics", "biases"])
def test_a_summary_missing_a_key_is_unreadable_and_the_good_one_still_loads(
    tmp_path: Path, key: str
) -> None:
    good = save_backtest_report(tmp_path, _report())
    body = json.loads(good.read_text(encoding="utf-8"))
    del body[key]
    (tmp_path / "backtest" / "gap.json").write_text(json.dumps(body), encoding="utf-8")

    loaded = load_backtest_reports(tmp_path)

    assert len(loaded.items) == 1
    assert list(loaded.unreadable) == ["gap.json"]
    assert key in loaded.unreadable["gap.json"]


def test_load_backtest_reports_of_a_root_without_a_run_is_empty(tmp_path: Path) -> None:
    loaded = load_backtest_reports(tmp_path)

    assert loaded.items == ()
    assert loaded.unreadable == {}


INT_FIELDS = [
    ("trading_days",),
    ("rebalances",),
    ("metrics", "honesty", "fundamentals_unresolved"),
    ("metrics", "honesty", "flagged_excluded"),
    ("metrics", "honesty", "stopped_trading"),
    ("metrics", "honesty", "daily_breaker_days"),
    ("metrics", "honesty", "drawdown_breaker_days"),
]


@pytest.mark.parametrize("bad", [True, 1.5], ids=["bool", "float"])
@pytest.mark.parametrize("path", INT_FIELDS, ids=lambda p: ".".join(p))
def test_an_int_field_that_is_not_an_int_makes_the_summary_unreadable(
    tmp_path: Path, path: tuple[str, ...], bad: object
) -> None:
    good = save_backtest_report(tmp_path, _report())
    body = json.loads(good.read_text(encoding="utf-8"))
    node = body
    for step in path[:-1]:
        node = node[step]
    node[path[-1]] = bad
    (tmp_path / "backtest" / "bad.json").write_text(json.dumps(body), encoding="utf-8")

    loaded = load_backtest_reports(tmp_path)

    assert len(loaded.items) == 1
    assert list(loaded.unreadable) == ["bad.json"]


def test_load_backtest_reports_names_a_summary_that_does_not_decode(tmp_path: Path) -> None:
    (tmp_path / "backtest").mkdir()
    (tmp_path / "backtest" / "x.json").write_text('{"run_id": "x"}', encoding="utf-8")

    loaded = load_backtest_reports(tmp_path)

    assert loaded.items == ()
    assert list(loaded.unreadable) == ["x.json"]
