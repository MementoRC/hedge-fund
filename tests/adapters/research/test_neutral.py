"""NeutralResearcher: the backtest's deterministic research stand-in."""

from datetime import date

import pytest

from hedge_fund.adapters.research.neutral import NeutralResearcher
from hedge_fund.domain.research import Rating, Verdict
from hedge_fund.domain.shortlist import Candidate, Shortlist, Side
from hedge_fund.engine.research.conviction import conviction
from hedge_fund.ports.research import Researcher

AS_OF = date(2024, 3, 1)
LONG = Candidate("AAA", "Tech", Side.LONG, 1, 1.5)
SHORT = Candidate("ZZZ", "Tech", Side.SHORT, 1, -1.5)
SHORTLIST = Shortlist(AS_OF, (LONG, SHORT), {})


def test_neutral_researcher_satisfies_the_port():
    researcher: Researcher = NeutralResearcher(SHORTLIST)
    assert researcher.run("AAA", AS_OF, 0).ticker == "AAA"


@pytest.mark.parametrize(("ticker", "rating"), [("AAA", Rating.BUY), ("ZZZ", Rating.SELL)])
def test_it_rates_each_name_with_its_shortlist_side(ticker, rating):
    record = NeutralResearcher(SHORTLIST).run(ticker, AS_OF, 2)
    assert (record.ticker, record.as_of, record.run_index) == (ticker, AS_OF, 2)
    assert record.rating is rating
    assert record.valid
    assert record.usage == ()
    assert record.seconds == 0.0
    assert record.reports == {}


def test_it_records_a_fixed_neutral_decision():
    assert NeutralResearcher(SHORTLIST).run("ZZZ", AS_OF, 0).decision == "neutral:short"


def test_a_name_off_the_shortlist_raises():
    with pytest.raises(ValueError, match="BBB: not on the 2024-03-01 shortlist"):
        NeutralResearcher(SHORTLIST).run("BBB", AS_OF, 0)


def test_another_as_of_raises():
    with pytest.raises(ValueError, match="2024-03-04: shortlist is as of 2024-03-01"):
        NeutralResearcher(SHORTLIST).run("AAA", date(2024, 3, 4), 0)


@pytest.mark.parametrize("candidate", [LONG, SHORT])
def test_every_name_is_confirmed_by_the_real_conviction(candidate):
    researcher = NeutralResearcher(SHORTLIST)
    runs = [researcher.run(candidate.ticker, AS_OF, i) for i in range(3)]
    result = conviction(candidate, runs)
    assert result.verdict is Verdict.CONFIRM
    assert result.std == 0.0
