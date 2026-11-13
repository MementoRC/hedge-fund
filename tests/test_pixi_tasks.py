"""The pixi tasks the docs and the dashboard's empty state name."""

from __future__ import annotations

import tomllib
from pathlib import Path

PIXI = Path(__file__).resolve().parents[1] / "pixi.toml"


def test_the_dev_feature_has_a_demo_task_that_builds_data_demo() -> None:
    tasks = tomllib.loads(PIXI.read_text(encoding="utf-8"))["feature"]["dev"]["tasks"]

    assert tasks["demo"] == "hedge-fund demo build --root data/demo"


def test_the_test_task_fails_below_full_coverage() -> None:
    tasks = tomllib.loads(PIXI.read_text(encoding="utf-8"))["feature"]["ci"]["tasks"]

    assert tasks["test"] == (
        "pytest tests/ --cov=hedge_fund --cov-report=term-missing --cov-fail-under=100"
    )
