"""Offline end-to-end: research service -> JSON store -> load (characterization test)."""

from datetime import date
from pathlib import Path

from hedge_fund.adapters.research.store import load_research, save_research
from hedge_fund.domain.research import Rating, Verdict
from hedge_fund.domain.shortlist import Candidate, Shortlist, Side
from hedge_fund.services.research import research_shortlist
from tests.fakes import ScriptedResearcher

AS_OF = date(2026, 9, 30)


def test_service_to_store_round_trip(tmp_path: Path) -> None:
    long = Candidate(ticker="AAA", sector="Tech", side=Side.LONG, depth=1, composite=1.5)
    short = Candidate(ticker="BBB", sector="Energy", side=Side.SHORT, depth=1, composite=-0.5)
    researcher = ScriptedResearcher(
        {
            ("AAA", 0): Rating.BUY,
            ("AAA", 1): Rating.BUY,
            ("AAA", 2): RuntimeError("boom"),
            ("BBB", 0): Rating.BUY,
            ("BBB", 1): Rating.BUY,
            ("BBB", 2): Rating.BUY,
        }
    )
    shortlist = Shortlist(as_of=AS_OF, candidates=(long, short), trimmed={})

    result = research_shortlist(researcher, shortlist, k=3, today=AS_OF)
    path = save_research(result, tmp_path)
    loaded = load_research(path)

    assert loaded == result
    assert [name.candidate.ticker for name in loaded.names] == ["AAA", "BBB"]
    assert [name.verdict for name in loaded.names] == [Verdict.CONFIRM, Verdict.CONTRADICT]
    assert [name.valid for name in loaded.names] == [2, 3]
    assert loaded.names[0].runs[2].error == "RuntimeError: boom"
