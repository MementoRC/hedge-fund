"""Live ``verify`` against the real research store format and the price vendor, read-only.
Excluded by default; run with: pytest -m network.

Saves one research result for AAPL in a temporary root and verifies it with real closes
(yfinance through the real provider cache). Needs no broker keys, never writes outside the
temporary root and never touches an order.
"""

import json
from datetime import date
from pathlib import Path

import pytest
from typer.testing import CliRunner

from hedge_fund import cli
from hedge_fund.adapters.research.store import save_research
from hedge_fund.domain.research import NameResearch, ResearchResult, Verdict
from hedge_fund.domain.shortlist import Candidate, Side

pytestmark = pytest.mark.network

AS_OF = date(2026, 9, 14)


def test_a_live_verify_of_one_stored_rating_saves_a_report(tmp_path: Path) -> None:
    name = NameResearch(
        candidate=Candidate("AAPL", "Information Technology", Side.LONG, 1, 1.0),
        runs=(),
        valid=3,
        mean=1.0,
        std=0.0,
        agreement=1.0,
        verdict=Verdict.CONFIRM,
    )
    save_research(ResearchResult(as_of=AS_OF, names=(name,)), tmp_path)
    result = CliRunner().invoke(cli.app, ["verify", "--root", str(tmp_path)])
    # 0: every horizon priced or pending; 2: the vendor lacked a close.
    assert result.exit_code in (0, 2), result.output
    [path] = (tmp_path / "verifications").iterdir()
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["start"] == AS_OF.isoformat()
    assert [row["horizon"] for row in report["observations"]] == [5, 21, 63]
    assert report["observations"][0]["status"] != "pending"  # 5 sessions have long passed
    assert [summary["horizon"] for summary in report["summaries"]] == [5, 21, 63]
