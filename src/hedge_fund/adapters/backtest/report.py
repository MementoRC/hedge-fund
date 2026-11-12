"""Write the backtest summary to ``<root>/backtest/<run-id>.json`` (write-once)."""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import date
from pathlib import Path
from typing import Any

from hedge_fund.adapters._json import decode_float, decode_int, decode_optional, jsonable
from hedge_fund.adapters._load import load_dir
from hedge_fund.adapters.data._atomic import atomic_create
from hedge_fund.domain.backtest import (
    BacktestMetrics,
    BacktestReport,
    Honesty,
    Performance,
    SimCosts,
    TradingCosts,
)
from hedge_fund.domain.loaded import Loaded

BACKTEST_DIR = "backtest"


def save_backtest_report(root: Path, report: BacktestReport) -> Path:
    """Write ``report`` atomically; raises FileExistsError if this run id already exists."""
    path = root / BACKTEST_DIR / f"{report.run_id}.json"
    body = json.dumps(jsonable(report), indent=2, allow_nan=False)
    try:
        atomic_create(path, lambda tmp: tmp.write_text(body, encoding="utf-8"))
    except FileExistsError as exc:
        raise FileExistsError(f"{path}: a report for this run id exists") from exc
    return path


def load_backtest_reports(root: Path) -> Loaded[BacktestReport]:
    """Decode every ``<root>/backtest/*.json``; a file that does not decode is in ``unreadable``."""
    return load_dir(Path(root) / BACKTEST_DIR, decode_backtest_report)


def decode_backtest_report(obj: Mapping[str, Any]) -> BacktestReport:
    """The BacktestReport whose ``jsonable`` form is ``obj``."""
    costs = obj["costs"]
    return BacktestReport(
        run_id=obj["run_id"],
        store_digest=obj["store_digest"],
        start=date.fromisoformat(obj["start"]),
        end=date.fromisoformat(obj["end"]),
        cash=decode_float(obj["cash"]),
        costs=SimCosts(
            slippage_bps=decode_float(costs["slippage_bps"]),
            borrow_rate=decode_float(costs["borrow_rate"]),
        ),
        trading_days=decode_int(obj["trading_days"]),
        rebalances=decode_int(obj["rebalances"]),
        metrics=_metrics(obj["metrics"]),
        biases=tuple(obj["biases"]),
    )


def _metrics(obj: Mapping[str, Any]) -> BacktestMetrics:
    return BacktestMetrics(
        strategy=_performance(obj["strategy"]),
        benchmark=_performance(obj["benchmark"]),
        beta=decode_optional(obj["beta"], decode_float),
        costs=_trading_costs(obj["costs"]),
        honesty=_honesty(obj["honesty"]),
    )


def _performance(item: Mapping[str, Any]) -> Performance:
    return Performance(
        start=date.fromisoformat(item["start"]),
        end=date.fromisoformat(item["end"]),
        total_return=decode_float(item["total_return"]),
        cagr=decode_float(item["cagr"]),
        volatility=decode_float(item["volatility"]),
        sharpe=decode_optional(item["sharpe"], decode_float),
        max_drawdown=decode_float(item["max_drawdown"]),
        peak=date.fromisoformat(item["peak"]),
        trough=date.fromisoformat(item["trough"]),
    )


def _trading_costs(item: Mapping[str, Any]) -> TradingCosts:
    return TradingCosts(
        traded_notional=decode_float(item["traded_notional"]),
        turnover=decode_float(item["turnover"]),
        slippage=decode_float(item["slippage"]),
        borrow=decode_float(item["borrow"]),
    )


def _honesty(item: Mapping[str, Any]) -> Honesty:
    return Honesty(
        price_unresolved=dict(item["price_unresolved"]),
        fundamentals_unresolved=decode_int(item["fundamentals_unresolved"]),
        flagged_excluded=decode_int(item["flagged_excluded"]),
        stopped_trading=decode_int(item["stopped_trading"]),
        rejects=dict(item["rejects"]),
        daily_breaker_days=decode_int(item["daily_breaker_days"]),
        drawdown_breaker_days=decode_int(item["drawdown_breaker_days"]),
        outcomes=dict(item["outcomes"]),
    )
