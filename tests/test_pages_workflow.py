"""The Pages workflow: triggers, permissions, concurrency and the build and deploy steps."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

WORKFLOWS = Path(__file__).resolve().parents[1] / ".github" / "workflows"


def _load(name: str) -> dict[Any, Any]:
    loaded = yaml.safe_load((WORKFLOWS / name).read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def _triggers(workflow: dict[Any, Any]) -> dict[str, Any]:
    # YAML 1.1 reads the key `on` as the boolean True.
    return workflow.get("on", workflow.get(True))


def _uses(job: dict[str, Any]) -> list[str]:
    return [step["uses"].split("@")[0] for step in job["steps"] if "uses" in step]


def test_it_runs_on_a_push_to_main_and_by_hand() -> None:
    triggers = _triggers(_load("pages.yml"))

    assert triggers["push"] == {"branches": ["main"]}
    assert "workflow_dispatch" in triggers


def test_it_asks_for_only_the_pages_permissions() -> None:
    assert _load("pages.yml")["permissions"] == {
        "contents": "read",
        "pages": "write",
        "id-token": "write",
    }


def test_deploys_queue_instead_of_cancelling() -> None:
    assert _load("pages.yml")["concurrency"] == {"group": "pages", "cancel-in-progress": False}


def test_the_build_job_installs_the_site_env_builds_and_uploads_the_site() -> None:
    build = _load("pages.yml")["jobs"]["build"]
    steps = build["steps"]

    assert _uses(build) == [
        "actions/checkout",
        "prefix-dev/setup-pixi",
        "actions/configure-pages",
        "actions/upload-pages-artifact",
    ]
    setup = next(s for s in steps if s.get("uses", "").startswith("prefix-dev/setup-pixi"))
    assert setup["with"]["environments"] == "site"
    assert [s["run"] for s in steps if "run" in s] == ["pixi run -e site site"]
    upload = next(s for s in steps if s.get("uses", "").startswith("actions/upload-pages"))
    assert upload["with"]["path"] == "_site"


def test_the_build_installs_the_same_pixi_as_ci() -> None:
    workflow = _load("pages.yml")
    setup = next(s for s in workflow["jobs"]["build"]["steps"] if "setup-pixi" in s.get("uses", ""))

    assert setup["with"]["pixi-version"] == _load("ci.yml")["jobs"]["ci"]["with"]["pixi-version"]


def test_the_deploy_job_runs_after_the_build_in_the_github_pages_environment() -> None:
    deploy = _load("pages.yml")["jobs"]["deploy"]

    assert deploy["needs"] == "build"
    assert deploy["environment"]["name"] == "github-pages"
    assert _uses(deploy) == ["actions/deploy-pages"]
