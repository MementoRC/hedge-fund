"""The NeutralResearcher through a real dry ``run_cycle``."""

from __future__ import annotations

import dataclasses

from hedge_fund.adapters.research.neutral import NeutralResearcher
from hedge_fund.domain.cycle import CycleOptions
from hedge_fund.domain.research import Verdict
from hedge_fund.services import cycle
from hedge_fund.services.cycle import run_cycle
from tests.cycle_world import AS_OF, EVENING, TODAY, Clock, World


def test_a_dry_cycle_with_the_neutral_researcher_confirms_every_name() -> None:
    world = World()
    planned = cycle.plan_shortlist(world.deps, AS_OF)
    assert planned.candidates
    deps = dataclasses.replace(world.deps, researcher=NeutralResearcher(planned))
    report = run_cycle(
        deps,
        as_of=AS_OF,
        today=TODAY,
        options=CycleOptions(dry_run=True),
        now=Clock(EVENING),
    )
    assert report.shortlist == planned
    assert report.research is not None
    assert {name.candidate.ticker for name in report.research.names} == {
        candidate.ticker for candidate in planned.candidates
    }
    assert {name.verdict for name in report.research.names} == {Verdict.CONFIRM}
    assert report.spend is not None
    assert report.spend.total == 0
