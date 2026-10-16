"""Tests for the candidate shortlist."""

import dataclasses
from datetime import date

import pytest

from hedge_fund.domain.scores import FactorScore, ScoreResult
from hedge_fund.domain.shortlist import Shortlist, Side
from hedge_fund.engine.factors.score import score
from hedge_fund.engine.shortlist.select import DEFAULT_SHORTLIST, ShortlistParams, shortlist
from tests.factories import make_closes, snapshot_from

AS_OF = date(2026, 9, 30)
LONG = Side.LONG
SHORT = Side.SHORT
NAN = float("nan")
WIDE = ShortlistParams(per_sector=2, max_names=40)


def _score(ticker: str, sector: str, composite: float) -> FactorScore:
    return FactorScore(
        ticker=ticker,
        sector=sector,
        raw={"f": composite},
        z={"f": composite},
        composite=composite,
        fallback_universe_z=False,
    )


def _result(*scores: FactorScore) -> ScoreResult:
    """ScoreResult over ``scores``, sorted like ``score()``."""
    ordered = sorted(scores, key=lambda s: (-s.composite, s.ticker))
    return ScoreResult(as_of=AS_OF, scores=tuple(ordered), excluded={})


def _picks(out: Shortlist) -> list[tuple[str, Side, int]]:
    return [(c.ticker, c.side, c.depth) for c in out.candidates]


# --- params ----------------------------------------------------------------


def test_default_params() -> None:
    assert ShortlistParams(per_sector=3, max_names=40) == DEFAULT_SHORTLIST


@pytest.mark.parametrize("field", ["per_sector", "max_names"])
@pytest.mark.parametrize("value", [0, -1, 1.5, NAN, True])
def test_rejects_invalid_params(field: str, value: object) -> None:
    params = dataclasses.replace(DEFAULT_SHORTLIST, **{field: value})
    with pytest.raises(ValueError, match=f"{field} must be >= 1"):
        shortlist(_result(), params=params)


# --- per-sector queues -------------------------------------------------------


def test_each_side_is_cut_to_per_sector_with_depth() -> None:
    result = _result(
        *(
            _score(t, "Tech", c)
            for t, c in [("A", 3.0), ("B", 2.0), ("C", 1.0), ("D", -1.0), ("E", -2.0), ("F", -3.0)]
        )
    )
    out = shortlist(result, params=WIDE)
    # depth 1: A (3.0) and F (-3.0) tie on |composite|, ticker breaks it; depth 2: B, E.
    assert _picks(out) == [("A", LONG, 1), ("F", SHORT, 1), ("B", LONG, 2), ("E", SHORT, 2)]
    # C and D never qualified (beyond per_sector), so they are not "trimmed".
    assert out.trimmed == {}
    assert out.as_of == AS_OF


def test_zero_and_nan_composites_are_on_neither_side() -> None:
    result = _result(_score("A", "Tech", 1.0), _score("Z", "Tech", 0.0), _score("N", "Tech", NAN))
    out = shortlist(result, params=WIDE)
    assert _picks(out) == [("A", LONG, 1)]
    assert out.trimmed == {}


def test_one_sided_sector_contributes_only_that_side() -> None:
    result = _result(_score("A", "Tech", 1.0), _score("B", "Tech", 0.5))
    out = shortlist(result, params=WIDE)
    assert _picks(out) == [("A", LONG, 1), ("B", LONG, 2)]


def test_candidates_carry_sector_and_composite() -> None:
    out = shortlist(_result(_score("A", "Tech", -0.7)), params=WIDE)
    (candidate,) = out.candidates
    assert (candidate.sector, candidate.composite) == ("Tech", -0.7)


# --- round-robin and cap -----------------------------------------------------

# Tech: A 1.0, B 0.5 (longs). Energy: C 2.0 (long), D -0.2 (short).
# depth 1 = {A, C, D} ordered by |composite|: C, A, D. depth 2 = {B}.
TWO_SECTORS = (
    _score("A", "Tech", 1.0),
    _score("B", "Tech", 0.5),
    _score("C", "Energy", 2.0),
    _score("D", "Energy", -0.2),
)


def test_round_robin_takes_every_depth_one_before_any_depth_two() -> None:
    out = shortlist(_result(*TWO_SECTORS), params=WIDE)
    # B (0.5) is stronger than D (-0.2) but comes after it: depth beats strength.
    assert _picks(out) == [("C", LONG, 1), ("A", LONG, 1), ("D", SHORT, 1), ("B", LONG, 2)]
    assert out.trimmed == {}


def test_cap_cuts_mid_depth_and_reports_trimmed() -> None:
    out = shortlist(_result(*TWO_SECTORS), params=ShortlistParams(per_sector=2, max_names=2))
    assert _picks(out) == [("C", LONG, 1), ("A", LONG, 1)]
    assert out.trimmed == {"B": "cap", "D": "cap"}
    assert list(out.trimmed) == ["B", "D"]


def test_equal_strength_across_sides_breaks_ties_by_ticker() -> None:
    result = _result(_score("B", "Tech", 1.0), _score("A", "Energy", -1.0))
    out = shortlist(result, params=ShortlistParams(per_sector=1, max_names=1))
    assert _picks(out) == [("A", SHORT, 1)]
    assert out.trimmed == {"B": "cap"}


# --- edges and integration ---------------------------------------------------


def test_empty_result_gives_empty_shortlist() -> None:
    result = ScoreResult(as_of=AS_OF, scores=(), excluded={"X": "missing_data"})
    assert shortlist(result) == Shortlist(as_of=AS_OF, candidates=(), trimmed={})


def test_shortlist_of_a_scored_universe() -> None:
    tickers = [f"{s}{i}" for s in "ABC" for i in range(6)]
    sectors = {t: {"A": "Tech", "B": "Energy", "C": "Health"}[t[0]] for t in tickers}
    result = score(snapshot_from(make_closes(tickers), sectors))
    out = shortlist(result, params=ShortlistParams(per_sector=2, max_names=5))
    scored = {s.ticker: s for s in result.scores}
    picked = [c.ticker for c in out.candidates]
    # Sector z-means average 0, so every sector has >= 1 long and >= 1 short: >= 6 qualify.
    assert len(picked) == 5
    assert len(set(picked)) == 5
    assert out.trimmed
    assert set(out.trimmed).isdisjoint(picked)
    assert set(out.trimmed.values()) == {"cap"}
    for candidate in out.candidates:
        source = scored[candidate.ticker]
        assert (candidate.sector, candidate.composite) == (source.sector, source.composite)
        assert (candidate.side is LONG) == (candidate.composite > 0)
        assert (candidate.side is SHORT) == (candidate.composite < 0)
