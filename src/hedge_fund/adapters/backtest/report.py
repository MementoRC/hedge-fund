"""Write the backtest summary to ``<root>/backtest/<run-id>.json`` (write-once)."""

from __future__ import annotations

import json
from pathlib import Path

from hedge_fund.adapters._json import jsonable
from hedge_fund.adapters.data._atomic import atomic_create
from hedge_fund.domain.backtest import BacktestReport

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
