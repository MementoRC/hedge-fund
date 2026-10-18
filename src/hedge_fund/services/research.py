"""Run the researcher K times per shortlist candidate and score the results."""

from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta

from hedge_fund.domain.research import NameResearch, ResearchResult, RunRecord
from hedge_fund.domain.shortlist import Shortlist
from hedge_fund.engine.research.conviction import DEFAULT_CONVICTION, ConvictionParams, conviction
from hedge_fund.ports.research import Researcher

_WEEKEND_DAYS_BACK = {0: 3, 5: 1, 6: 2}  # Monday, Saturday, Sunday -> previous Friday


def research_shortlist(
    researcher: Researcher,
    shortlist: Shortlist,
    *,
    k: int = 3,
    max_workers: int = 4,
    params: ConvictionParams = DEFAULT_CONVICTION,
    today: date,
) -> ResearchResult:
    """Research every candidate ``k`` times (live only) and return per-name verdicts.

    ``today`` is injected for tests. Raises ValueError unless ``k`` and ``max_workers``
    are ints >= 1, or unless ``previous weekday(today) <= shortlist.as_of <= today``
    (holidays ignored): TradingAgents researches the present, not history. An empty
    shortlist returns an empty result without calling the researcher. A run raising
    ``Exception`` becomes an error record; ``BaseException`` (e.g. KeyboardInterrupt)
    propagates after cancelling queued runs (only in-flight runs are left to finish). Names
    follow the shortlist's candidate order. Tickers are assumed unique within a shortlist
    (the shortlist guarantees it); runs are grouped by ticker.
    """
    _validate_count("k", k)
    _validate_count("max_workers", max_workers)
    as_of = shortlist.as_of
    if not _previous_weekday(today) <= as_of <= today:
        raise ValueError(
            f"as_of {as_of} is not live: must be between {_previous_weekday(today)} and {today}"
        )
    if not shortlist.candidates:
        return ResearchResult(as_of=as_of, names=())
    pool = ThreadPoolExecutor(max_workers=max_workers)
    try:
        futures = [
            pool.submit(_run_one, researcher, candidate.ticker, as_of, index)
            for candidate in shortlist.candidates
            for index in range(k)
        ]
        records = [future.result() for future in futures]
    except BaseException:
        # Ctrl-C (or any non-Exception) must not wait for every queued LLM run.
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    else:
        pool.shutdown(wait=True)
    by_ticker: dict[str, list[RunRecord]] = defaultdict(list)
    for record in records:
        by_ticker[record.ticker].append(record)
    names: tuple[NameResearch, ...] = tuple(
        conviction(candidate, by_ticker[candidate.ticker], params=params)
        for candidate in shortlist.candidates
    )
    return ResearchResult(as_of=as_of, names=names)


def _run_one(researcher: Researcher, ticker: str, as_of: date, run_index: int) -> RunRecord:
    try:
        return researcher.run(ticker, as_of, run_index)
    except Exception as exc:
        return RunRecord(
            ticker=ticker,
            as_of=as_of,
            run_index=run_index,
            rating=None,
            decision="",
            reports={},
            seconds=0.0,
            error=f"{type(exc).__name__}: {exc}",
        )


def _previous_weekday(day: date) -> date:
    """Previous Friday for Saturday, Sunday and Monday, otherwise the day before."""
    return day - timedelta(days=_WEEKEND_DAYS_BACK.get(day.weekday(), 1))


def _validate_count(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be an int >= 1, got {value!r}")
