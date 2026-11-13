"""One real demo root per test session, built through the CLI."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from hedge_fund import cli
from hedge_fund.viz.load import Workspace, load_workspace


@pytest.fixture(scope="session")
def demo_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("viz") / "demo"
    result = CliRunner().invoke(cli.app, ["demo", "build", "--root", str(root)])
    assert result.exit_code == 0, result.output
    return root


@pytest.fixture(scope="session")
def workspace(demo_root: Path) -> Workspace:
    return load_workspace(demo_root)
