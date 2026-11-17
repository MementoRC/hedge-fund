"""The Pages workflow: PR builds, the two checkouts, the recorded test run, gated deploy."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

WORKFLOWS = Path(__file__).resolve().parents[1] / ".github" / "workflows"
NOT_PR = "github.event_name != 'pull_request'"


def _load(name: str) -> dict[Any, Any]:
    loaded = yaml.safe_load((WORKFLOWS / name).read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def _triggers(workflow: dict[Any, Any]) -> dict[str, Any]:
    # YAML 1.1 reads the key `on` as the boolean True.
    return workflow.get("on", workflow.get(True))


def _uses(job: dict[str, Any]) -> list[str]:
    return [step["uses"].split("@")[0] for step in job["steps"] if "uses" in step]


def _step(job: dict[str, Any], uses: str) -> dict[str, Any]:
    return next(s for s in job["steps"] if s.get("uses", "").startswith(uses))


def _build() -> dict[str, Any]:
    return _load("pages.yml")["jobs"]["build"]


def test_it_runs_on_development_pushes_prs_into_development_and_by_hand() -> None:
    triggers = _triggers(_load("pages.yml"))

    assert triggers["push"] == {"branches": ["development"]}
    assert triggers["pull_request"] == {"branches": ["development"]}
    assert "workflow_dispatch" in triggers


def test_permissions_are_read_only_except_the_deploy_job() -> None:
    workflow = _load("pages.yml")

    assert workflow["permissions"] == {"contents": "read"}
    assert workflow["jobs"]["build"]["permissions"] == {"contents": "read", "pages": "read"}
    assert workflow["jobs"]["deploy"]["permissions"] == {"pages": "write", "id-token": "write"}


def test_only_pr_runs_cancel_and_deploys_share_one_group() -> None:
    assert _load("pages.yml")["concurrency"] == {
        "group": "${{ github.event_name == 'pull_request' && "
        "format('pages-pr-{0}', github.ref) || 'pages' }}",
        "cancel-in-progress": "${{ github.event_name == 'pull_request' }}",
    }


def test_the_build_checks_out_hedge_fund_and_the_tradingagents_fork_side_by_side() -> None:
    checkouts = [s for s in _build()["steps"] if s.get("uses", "").startswith("actions/checkout")]

    assert [c["with"] for c in checkouts] == [
        {"path": "hedge-fund"},
        {"repository": "MementoRC/TradingAgents", "ref": "hedge-fund", "path": "TradingAgents"},
    ]


def test_the_build_runs_in_hedge_fund_with_the_site_env() -> None:
    build = _build()
    setup = _step(build, "prefix-dev/setup-pixi")

    assert build["defaults"] == {"run": {"working-directory": "hedge-fund"}}
    assert setup["with"]["environments"] == "site"
    assert setup["with"]["manifest-path"] == "hedge-fund/pixi.toml"


def test_the_build_installs_tests_builds_the_demo_then_the_site() -> None:
    runs = [s for s in _build()["steps"] if "run" in s]

    assert [s["run"] for s in runs] == [
        "pixi run -e site setup-dev",
        "pixi run -e site test --junitxml=junit.xml --cov-report=json:coverage.json",
        "pixi run -e site demo",
        "pixi run -e site site",
    ]
    assert [s.get("continue-on-error", False) for s in runs] == [False, True, False, False]


def test_the_build_configures_pages_off_prs_and_uploads_the_site() -> None:
    build = _build()

    assert _uses(build) == [
        "actions/checkout",
        "actions/checkout",
        "prefix-dev/setup-pixi",
        "actions/configure-pages",
        "actions/upload-pages-artifact",
    ]
    assert _step(build, "actions/configure-pages")["if"] == NOT_PR
    assert _step(build, "actions/upload-pages-artifact")["with"]["path"] == "hedge-fund/_site"


def test_the_build_installs_the_same_pixi_as_ci() -> None:
    setup = _step(_build(), "prefix-dev/setup-pixi")

    assert setup["with"]["pixi-version"] == _load("ci.yml")["jobs"]["ci"]["with"]["pixi-version"]


def test_the_deploy_job_runs_after_the_build_and_never_on_a_pr() -> None:
    deploy = _load("pages.yml")["jobs"]["deploy"]

    assert deploy["needs"] == "build"
    assert deploy["if"] == NOT_PR
    assert deploy["environment"]["name"] == "github-pages"
    assert _uses(deploy) == ["actions/deploy-pages"]


def test_the_header_names_both_one_time_repo_settings() -> None:
    text = (WORKFLOWS / "pages.yml").read_text(encoding="utf-8")

    assert 'Settings > Pages > Source = "GitHub Actions"' in text
    assert "Settings > Environments > github-pages > Deployment branches" in text
