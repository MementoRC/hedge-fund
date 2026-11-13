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


def _manifest() -> dict:
    return tomllib.loads(PIXI.read_text(encoding="utf-8"))


def test_the_site_feature_adds_mkdocs_material() -> None:
    deps = _manifest()["feature"]["site"]["dependencies"]

    assert "mkdocs-material" in deps


def test_the_site_environment_is_ci_dev_and_site_in_the_default_solve_group() -> None:
    env = _manifest()["environments"]["site"]

    assert env == {"features": ["ci", "dev", "site"], "solve-group": "default"}


def test_the_site_task_builds_the_site_strictly_into_the_repo_root() -> None:
    tasks = _manifest()["feature"]["site"]["tasks"]

    assert tasks["site"] == "mkdocs build --strict -f website/mkdocs.yml -d ../_site"


def test_dev_and_ci_stay_free_of_mkdocs() -> None:
    features = _manifest()["feature"]

    for name in ("dev", "ci"):
        assert not any("mkdocs" in dep for dep in features[name].get("dependencies", {}))
    assert not any("mkdocs" in dep for dep in _manifest()["dependencies"])
