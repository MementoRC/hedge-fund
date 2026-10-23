"""Live smoke test: one real TradingAgents run (network, real LLM spend; opt-in)."""

import os
from datetime import date, timedelta
from pathlib import Path

import pytest

from hedge_fund.adapters.research.tradingagents import ResearchConfig, TradingAgentsResearcher
from hedge_fund.domain.research import Verdict
from hedge_fund.domain.shortlist import Candidate, Shortlist, Side
from hedge_fund.services.research import research_shortlist

pytestmark = [
    pytest.mark.network,
    pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY unset"),
]


def _last_weekday_before(today: date) -> date:
    day = today - timedelta(days=1)
    while day.weekday() >= 5:
        day -= timedelta(days=1)
    return day


def test_live_single_run(tmp_path: Path) -> None:
    pytest.importorskip("tradingagents.graph.trading_graph")
    today = date.today()
    as_of = _last_weekday_before(today)
    shortlist = Shortlist(
        as_of=as_of,
        candidates=(
            Candidate(ticker="AAPL", sector="Tech", side=Side.LONG, depth=1, composite=1.0),
        ),
        trimmed={},
    )
    researcher = TradingAgentsResearcher(ResearchConfig(root=tmp_path / "research"))
    out = research_shortlist(researcher, shortlist, k=1, max_workers=1, today=today)
    name = out.names[0]
    run = name.runs[0]
    assert run.error is None, run.error
    assert run.rating is not None
    assert run.decision
    assert set(run.reports) == {
        "market_report",
        "sentiment_report",
        "news_report",
        "fundamentals_report",
    }
    assert name.verdict is Verdict.INSUFFICIENT  # K=1 is below min_valid=2
    assert list((tmp_path / "research").iterdir()) == []  # scratch dir cleaned up
