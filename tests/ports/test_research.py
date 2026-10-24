"""Tests for the Researcher and ResearchStore ports, exercised through the fakes."""

from datetime import date

import pytest

from hedge_fund.domain.research import Rating, ResearchResult
from hedge_fund.ports.research import Researcher, ResearchStore
from tests.fakes import InMemoryResearchStore, ScriptedResearcher

AS_OF = date(2026, 9, 30)


def test_scripted_researcher_satisfies_the_port() -> None:
    fake = ScriptedResearcher({("A", 1): Rating.SELL})
    researcher: Researcher = fake
    scripted = researcher.run("A", AS_OF, 1)
    default = researcher.run("A", AS_OF, 0)
    assert scripted.rating is Rating.SELL
    assert default.rating is Rating.HOLD
    assert scripted.error is None
    assert fake.calls == [("A", AS_OF, 1), ("A", AS_OF, 0)]


def test_scripted_researcher_raises_scripted_exceptions() -> None:
    researcher = ScriptedResearcher({("A", 0): RuntimeError("boom")})
    with pytest.raises(RuntimeError, match="boom"):
        researcher.run("A", AS_OF, 0)


def test_in_memory_research_store_satisfies_the_port() -> None:
    older = ResearchResult(as_of=AS_OF, names=())
    other_day = ResearchResult(as_of=date(2026, 9, 29), names=())
    fake = InMemoryResearchStore([older, other_day])
    store: ResearchStore = fake
    newer = ResearchResult(as_of=AS_OF, names=())
    store.save(newer)
    assert store.load(AS_OF) is newer
    assert store.load(date(2026, 9, 29)) is other_day
    assert store.load(date(2026, 9, 28)) is None
    assert fake.saved == [older, other_day, newer]
    assert fake.loaded == [AS_OF, date(2026, 9, 29), date(2026, 9, 28)]


def test_in_memory_research_store_starts_empty() -> None:
    assert InMemoryResearchStore().load(AS_OF) is None
