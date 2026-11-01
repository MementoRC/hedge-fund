"""Live ``cycle --dry-run`` against Alpaca paper and real data. Excluded by default; run with:
pytest -m network.

Needs APCA_API_KEY_ID and APCA_API_SECRET_KEY for a paper account and SEC_USER_AGENT;
skipped when any is unset. The researcher is scripted (no LLM calls). A dry run never calls
the approval gate, never submits and never saves an account snapshot.
"""

import dataclasses
import json
import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

from hedge_fund import cli
from hedge_fund.services.cycle import CycleDeps
from tests.fakes import ScriptedResearcher

_NEEDED = ("APCA_API_KEY_ID", "APCA_API_SECRET_KEY", "SEC_USER_AGENT")

pytestmark = [
    pytest.mark.network,
    pytest.mark.skipif(
        not all(os.environ.get(name) for name in _NEEDED), reason=f"{', '.join(_NEEDED)} unset"
    ),
]


def test_a_live_dry_run_writes_a_report_and_no_snapshot(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    real = cli.build_deps

    def scripted(root: Path, prices: Path) -> CycleDeps:
        return dataclasses.replace(real(root, prices), researcher=ScriptedResearcher())

    monkeypatch.setattr(cli, "build_deps", scripted)
    result = CliRunner().invoke(cli.app, ["cycle", "--dry-run", "-k", "1", "--root", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert result.stdout.startswith("outcome: dry-run\n")
    [path] = (tmp_path / "runs").iterdir()
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["outcome"] == "dry-run"
    assert report["account"]["equity"] > 0
    assert report["plan"] is not None
    assert not (tmp_path / "account").exists()
