"""Tests for the Researcher port, exercised through the scripted fake."""

from datetime import date

import pytest

from hedge_fund.domain.research import Rating
from hedge_fund.ports.research import Researcher
from tests.fakes import ScriptedResearcher

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
