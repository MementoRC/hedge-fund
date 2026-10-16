"""Tests for the candidate shortlist."""

import dataclasses
from datetime import date

import pytest

from hedge_fund.domain.scores import FactorScore, ScoreResult
from hedge_fund.domain.shortlist import Shortlist, Side
from hedge_fund.engine.shortlist.select import DEFAULT_SHORTLIST, ShortlistParams, shortlist

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
    assert DEFAULT_SHORTLIST == ShortlistParams(per_sector=3, max_names=40)


@pytest.mark.parametrize("field", ["per_sector", "max_names"])
@pytest.mark.parametrize("value", [0, -1, 1.5, NAN, True])
def test_rejects_invalid_params(field: str, value: object) -> None:
    params = dataclasses.replace(DEFAULT_SHORTLIST, **{field: value})
    with pytest.raises(ValueError, match=f"{field} must be >= 1"):
        shortlist(_result(), params=params)
