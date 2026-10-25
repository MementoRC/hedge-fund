"""Run the researcher K times per shortlist candidate and score the results."""

from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import date

from hedge_fund.domain.research import NameResearch, ResearchResult, RunRecord
from hedge_fund.domain.shortlist import Shortlist
from hedge_fund.engine.dates import previous_weekday
from hedge_fund.engine.research.conviction import DEFAULT_CONVICTION, ConvictionParams, conviction
from hedge_fund.ports.research import Researcher


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
    follow the shortlist's candidate order. Runs are grouped by the candidate they were
    submitted for; a record whose ticker, as_of or run_index differs from the request
    becomes an error record for the requested run.
    """
    _validate_count("k", k)
    _validate_count("max_workers", max_workers)
    as_of = shortlist.as_of
    if not previous_weekday(today) <= as_of <= today:
        raise ValueError(
            f"as_of {as_of} is not live: must be between {previous_weekday(today)} and {today}"
        )
    if not shortlist.candidates:
        return ResearchResult(as_of=as_of, names=())
    pool = ThreadPoolExecutor(max_workers=max_workers)
    try:
        submitted = [
            (
                candidate,
                pool.submit(_run_one, researcher, candidate.ticker, as_of, index),
            )
            for candidate in shortlist.candidates
            for index in range(k)
        ]
        results = [(candidate, future.result()) for candidate, future in submitted]
    except BaseException:
        # Ctrl-C (or any non-Exception) must not wait for every queued LLM run.
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    else:
        pool.shutdown(wait=True)
    by_ticker: dict[str, list[RunRecord]] = defaultdict(list)
    for candidate, record in results:
        by_ticker[candidate.ticker].append(record)
    names: tuple[NameResearch, ...] = tuple(
        conviction(candidate, by_ticker[candidate.ticker], params=params)
        for candidate in shortlist.candidates
    )
    return ResearchResult(as_of=as_of, names=names)


def _error_record(ticker: str, as_of: date, run_index: int, error: str) -> RunRecord:
    return RunRecord(
        ticker=ticker,
        as_of=as_of,
        run_index=run_index,
        rating=None,
        decision="",
        reports={},
        seconds=0.0,
        error=error,
    )


def _run_one(researcher: Researcher, ticker: str, as_of: date, run_index: int) -> RunRecord:
    try:
        record = researcher.run(ticker, as_of, run_index)
    except Exception as exc:
        return _error_record(ticker, as_of, run_index, f"{type(exc).__name__}: {exc}")
    if (record.ticker, record.as_of, record.run_index) != (ticker, as_of, run_index):
        return _error_record(
            ticker,
            as_of,
            run_index,
            f"ValueError: researcher returned {record.ticker!r} {record.as_of} run "
            f"{record.run_index}, expected {ticker!r} {as_of} run {run_index}",
        )
    return record


def _validate_count(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be an int >= 1, got {value!r}")
