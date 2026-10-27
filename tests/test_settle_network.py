"""Live ``settle`` against Alpaca paper, read-only. Excluded by default; run with:
pytest -m network.

Needs APCA_API_KEY_ID and APCA_API_SECRET_KEY for a paper account; skipped when either is
unset. Never submits, cancels or modifies orders.
"""

import json
import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

from hedge_fund import cli
from hedge_fund.adapters.broker import AlpacaPaperBroker
from hedge_fund.domain.ledger import OrderState

pytestmark = [
    pytest.mark.network,
    pytest.mark.skipif(
        not (os.environ.get("APCA_API_KEY_ID") and os.environ.get("APCA_API_SECRET_KEY")),
        reason="APCA_API_KEY_ID / APCA_API_SECRET_KEY unset",
    ),
]


def test_an_id_never_submitted_is_not_found() -> None:
    status = AlpacaPaperBroker().order_status("hf-smoke-never-submitted")
    assert status.state is OrderState.NOT_FOUND


def test_a_live_settle_over_an_empty_root_saves_a_report(tmp_path: Path) -> None:
    result = CliRunner().invoke(cli.app, ["settle", "--root", str(tmp_path)])
    # 0: no paper positions; 2: positions that no opening.json seeds are breaks.
    assert result.exit_code in (0, 2), result.output
    [path] = (tmp_path / "settlements").iterdir()
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["outcome"] == ("clean" if result.exit_code == 0 else "break")
    assert report["new_fills"] == []
    assert isinstance(report["broker_positions"], list)
