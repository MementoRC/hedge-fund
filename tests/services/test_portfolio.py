"""Tests for the portfolio service: benchmark fetch and end-to-end portfolio build."""

import dataclasses
import math
from collections.abc import Sequence
from datetime import date, timedelta
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from hedge_fund.domain.portfolio import TargetPortfolio
from hedge_fund.domain.research import NameResearch, ResearchResult, Verdict
from hedge_fund.domain.shortlist import Candidate, Side
from hedge_fund.domain.snapshot import PriceSnapshot
from hedge_fund.engine.portfolio import optimize as optimize_module
from hedge_fund.engine.portfolio.alpha import AlphaParams
from hedge_fund.engine.portfolio.correlation import CorrelationParams
from hedge_fund.engine.portfolio.optimize import Asset, OptimizeParams, PortfolioInfeasibleError
from hedge_fund.engine.portfolio.risk_model import RiskParams
from hedge_fund.services.portfolio import (
    _assemble,
    build_portfolio,
    fetch_benchmark,
    naive_dates,
    validate_benchmark_end,
    validate_inputs,
)
from tests.fakes import FakeProvider

AS_OF = date(2026, 9, 30)  # Wednesday
NAN = float("nan")
LONG = Side.LONG
SHORT = Side.SHORT
CONFIRM = Verdict.CONFIRM


class _LeakyProvider(FakeProvider):
    """Ignores `end` and returns every row it has, including rows after as_of."""

    def closes(self, tickers: Sequence[str], start: date, end: date) -> pd.DataFrame:
        return self.frame.reindex(columns=list(tickers))


def _spy_frame(values: list[float], end: str = "2026-09-30") -> pd.DataFrame:
    index = pd.bdate_range(end=end, periods=len(values))
    return pd.DataFrame({"SPY": values, "QQQ": [1.0] * len(values)}, index=index)


# --- fetch_benchmark -------------------------------------------------------


def test_fetch_benchmark_requests_lookback_plus_one_rows_of_calendar() -> None:
    # 253 trading rows -> ceil(253 * 365 / 252) + 30 = 367 + 30 = 397 calendar days.
    fake = FakeProvider(_spy_frame([400.0, NAN, 402.0]), {})
    out = fetch_benchmark(fake, AS_OF, lookback_days=252)
    assert fake.closes_calls == [(("SPY",), AS_OF - timedelta(days=397), AS_OF)]
    assert out.tolist() == [400.0, 402.0]  # the NaN row is dropped
    assert out.index[-1] == pd.Timestamp(AS_OF)


def test_fetch_benchmark_uses_the_given_ticker() -> None:
    fake = FakeProvider(_spy_frame([400.0, 401.0]), {})
    out = fetch_benchmark(fake, AS_OF, lookback_days=10, ticker="QQQ")
    assert fake.closes_calls[0][0] == ("QQQ",)
    assert out.tolist() == [1.0, 1.0]


def test_fetch_benchmark_drops_rows_after_as_of() -> None:
    fake = _LeakyProvider(_spy_frame([400.0, 401.0, 402.0], end="2026-10-02"), {})
    out = fetch_benchmark(fake, AS_OF, lookback_days=10)
    assert out.tolist() == [400.0]  # 2026-10-01 and 2026-10-02 are after as_of


def test_fetch_benchmark_normalises_a_tz_aware_index_to_naive_dates() -> None:
    frame = _spy_frame([400.0, 401.0, 402.0])
    frame.index = frame.index.tz_localize("America/New_York")
    # _LeakyProvider skips FakeProvider's naive-Timestamp window slicing on the index.
    out = fetch_benchmark(_LeakyProvider(frame, {}), AS_OF, lookback_days=10)
    assert out.index.tz is None
    assert out.index[-1] == pd.Timestamp(AS_OF)
    assert out.tolist() == [400.0, 401.0, 402.0]


def test_fetch_benchmark_rejects_a_missing_ticker() -> None:
    # The provider returns an all-NaN column for an unknown ticker, not an empty frame.
    fake = FakeProvider(_spy_frame([400.0, 401.0]), {})
    with pytest.raises(ValueError, match="no 'XYZ' closes"):
        fetch_benchmark(fake, AS_OF, lookback_days=10, ticker="XYZ")


# --- build_portfolio -------------------------------------------------------
#
# Eight daily returns of +-1% following orthogonal, zero-mean sign patterns (Walsh
# functions), so every sample variance is 1e-4 and every cross-pattern covariance is 0.

SPY_SIGNS = [1, -1, 1, -1, 1, -1, 1, -1]
F1 = [1, 1, -1, -1, 1, 1, -1, -1]
F2 = [1, -1, -1, 1, 1, -1, -1, 1]
F3 = [1, 1, 1, 1, -1, -1, -1, -1]
RISK = RiskParams(lookback=8, min_obs=6)
INDEX = pd.bdate_range(end="2026-09-30", periods=9)

# ticker: (sector, side, sign pattern, verdict, mean, std)
UNIVERSE: dict[str, tuple[str, Side, list[int], Verdict, float, float]] = {
    "AAA": ("Tech", LONG, F1, CONFIRM, 2.0, 0.0),  # score (2 / 2) / 1 = 1.0
    "BBB": ("Health", LONG, F2, CONFIRM, 1.5, 0.5),  # (1.5 / 2) / 1.5 = 0.5
    "CCC": ("Energy", LONG, F3, CONFIRM, 1.0, 0.0),  # 0.5
    "DDD": ("Tech", LONG, F1, CONFIRM, 1.0, 0.0),  # 0.5; same returns as AAA
    "NNN": ("Health", LONG, F2, Verdict.NEUTRAL, 0.0, 0.5),
    "HHH": ("Energy", SHORT, F3, CONFIRM, 2.0, 0.0),  # NaN closes: short history
    "XXX": ("Tech", SHORT, F2, CONFIRM, 2.0, 0.0),  # 1.0
    "YYY": ("Health", SHORT, F3, CONFIRM, 1.0, 0.0),  # 0.5
    "ZZZ": ("Energy", SHORT, F1, CONFIRM, 1.0, 0.0),  # 0.5; AAA's returns, other side
}
VOL = math.sqrt(252 * 1e-4)  # every name: see the end-to-end test


def _path(signs: list[int]) -> list[float]:
    prices = [100.0]
    for sign in signs:
        prices.append(prices[-1] * (1 + 0.01 * sign))
    return prices


def _snapshot() -> PriceSnapshot:
    closes = {ticker: _path(spec[2]) for ticker, spec in UNIVERSE.items()}
    closes["HHH"] = [NAN if 2 <= row <= 5 else p for row, p in enumerate(closes["HHH"])]
    sectors = {ticker: spec[0] for ticker, spec in UNIVERSE.items()}
    return PriceSnapshot(as_of=AS_OF, closes=pd.DataFrame(closes, index=INDEX), sectors=sectors)


def _benchmark() -> pd.Series:
    return pd.Series(_path(SPY_SIGNS), index=INDEX)


def _research(*tickers: str, as_of: date = AS_OF) -> ResearchResult:
    names = []
    for ticker in tickers or tuple(UNIVERSE):
        sector, side, _, verdict, mean, std = UNIVERSE[ticker]
        candidate = Candidate(ticker=ticker, sector=sector, side=side, depth=1, composite=1.0)
        names.append(
            NameResearch(
                candidate=candidate,
                runs=(),
                valid=3,
                mean=mean,
                std=std,
                agreement=1.0,
                verdict=verdict,
            )
        )
    return ResearchResult(as_of=as_of, names=tuple(names))


def _build(research: ResearchResult, **kwargs: object) -> TargetPortfolio:
    return build_portfolio(research, _snapshot(), _benchmark(), risk=RISK, **kwargs)  # type: ignore[arg-type]


def test_end_to_end_sizes_or_explains_every_name() -> None:
    # Risk model on the 8 CONFIRM names: HHH has 3 returns (< 6) -> history. The other 7
    # fall in three identical-return groups {AAA, DDD, ZZZ}, {BBB, XXX}, {CCC, YYY}; in 1e-4
    # units S is 1 within a group and 0 across, so m = 1, d2 = 10 off-diagonal ones / 7,
    # b2_bar = 8 rows * 32 cross-group entries / (7 * 64) = 4/7, delta = 0.4 and every
    # diagonal of sigma is 252e-4 * (0.4 + 0.6 * 1): vol = sqrt(0.0252) for every name.
    # DDD duplicates AAA (rho 1) with a lower score -> correlated:AAA. 3 + 3 names in paired
    # sectors -> every weight at its 0.10 bound. The book's return series is exactly 0, so
    # w' sigma w = 252e-4 * 0.4 * (6 * 0.01) = 6.048e-4.
    research = _research()
    out = _build(research)
    assert out.as_of == AS_OF
    assert out.excluded == {"DDD": "correlated:AAA", "HHH": "history", "NNN": "verdict:neutral"}
    # Reasons arrive as NNN (gate), HHH (risk model), DDD (correlation): sorted on output.
    assert list(out.excluded) == ["DDD", "HHH", "NNN"]
    assert [(p.ticker, p.side) for p in out.positions] == [
        ("AAA", LONG),
        ("BBB", LONG),
        ("CCC", LONG),
        ("XXX", SHORT),
        ("YYY", SHORT),
        ("ZZZ", SHORT),
    ]
    assert [p.weight for p in out.positions] == pytest.approx([0.1] * 3 + [-0.1] * 3, abs=1e-6)
    assert [p.sector for p in out.positions] == ["Tech", "Health", "Energy"] * 2
    # alpha = side * 0.05 * vol * score
    expected_alpha = [0.05 * VOL * s for s in (1.0, 0.5, 0.5, -1.0, -0.5, -0.5)]
    assert [p.alpha for p in out.positions] == pytest.approx(expected_alpha)
    assert [p.beta for p in out.positions] == pytest.approx([0.0] * 6, abs=1e-9)
    assert out.gross == pytest.approx(0.6, abs=1e-6)
    assert out.net == pytest.approx(0.0, abs=1e-6)
    assert out.beta == pytest.approx(0.0, abs=1e-9)
    assert out.expected_vol == pytest.approx(math.sqrt(6.048e-4))
    sized = {p.ticker for p in out.positions}
    researched = {name.candidate.ticker for name in research.names}
    assert sized | set(out.excluded) == researched
    assert not sized & set(out.excluded)


def _empty(excluded: dict[str, str]) -> TargetPortfolio:
    return TargetPortfolio(
        as_of=AS_OF, positions=(), excluded=excluded, gross=0.0, net=0.0, beta=0.0, expected_vol=0.0
    )


def test_empty_research_gives_an_empty_portfolio() -> None:
    out = build_portfolio(ResearchResult(as_of=AS_OF, names=()), _snapshot(), _benchmark())
    assert out == _empty({})


def test_no_confirm_names_gives_an_empty_portfolio() -> None:
    assert _build(_research("NNN")) == _empty({"NNN": "verdict:neutral"})


def test_no_history_left_gives_an_empty_portfolio() -> None:
    assert _build(_research("HHH", "NNN")) == _empty({"HHH": "history", "NNN": "verdict:neutral"})


def test_thin_book_gives_an_empty_portfolio() -> None:
    # Without CCC only AAA and BBB are left long (DDD is correlated): 2 < 3.
    out = _build(_research("AAA", "BBB", "DDD", "XXX", "YYY", "ZZZ"))
    assert out == _empty(
        {
            "AAA": "thin-book",
            "BBB": "thin-book",
            "DDD": "correlated:AAA",
            "XXX": "thin-book",
            "YYY": "thin-book",
            "ZZZ": "thin-book",
        }
    )


def test_params_are_passed_to_every_step() -> None:
    # max_rho 1 keeps DDD; min_names_per_side 4 then makes the 4-long / 3-short book thin.
    out = _build(
        _research(),
        correlation=CorrelationParams(max_rho=1),
        optimize=OptimizeParams(min_names_per_side=4, max_weight=0.25),
    )
    assert out.excluded["DDD"] == "thin-book"
    assert out.positions == ()
    doubled = _build(_research(), alpha=AlphaParams(ic=0.1))
    assert doubled.positions[0].alpha == pytest.approx(0.1 * VOL)  # AAA, score 1


def test_optimizer_infeasibility_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    # The full universe reaches SLSQP (6 sized names); a failed solve must not be swallowed.
    # Real inputs only reach this raise through a solver failure, so the solve is scripted:
    # an LP-infeasible start returns an all-"infeasible" Allocation instead of raising.
    def failed(*args: object, **kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(success=False, status=9, message="scripted", x=np.zeros(6))

    monkeypatch.setattr(optimize_module, "minimize", failed)
    with pytest.raises(PortfolioInfeasibleError, match="did not converge"):
        _build(_research())


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"alpha": AlphaParams(ic=0)}, "ic must be > 0"),
        ({"risk": RiskParams(lookback=1, min_obs=1)}, "2 <= min_obs <= lookback"),
        ({"correlation": CorrelationParams(max_rho=2)}, "-1 < max_rho <= 1"),
        ({"optimize": OptimizeParams(gross=-1)}, r"0 < max_weight <= gross / 2"),
    ],
    ids=["alpha", "risk", "correlation", "optimize"],
)
def test_invalid_params_are_rejected_even_for_empty_research(
    params: dict[str, object], message: str
) -> None:
    research = ResearchResult(as_of=AS_OF, names=())
    with pytest.raises(ValueError, match=message):
        build_portfolio(research, _snapshot(), _benchmark(), **params)  # type: ignore[arg-type]


def test_positions_are_ordered_longs_then_shorts_by_size_then_ticker() -> None:
    # Sigma is 0.04 on the diagonal with cov(A, B) = 0.01, cov(X, Z) = 0.01 and, for the
    # dropped Q, cov(Q, A) = cov(Q, B) = 0.02; its index order differs from `assets`.
    # w' sigma w over the five positions (Q has no weight, so it is not one):
    # 0.04 * 0.0275 = 0.0011 diagonal, + 2 * 0.01 * (0.05 * 0.10) = 0.0001 for A-B,
    # + 2 * 0.01 * (-0.05 * -0.10) = 0.0001 for X-Z: 0.0013.
    assets = [
        Asset(ticker="A", sector="Tech", side=LONG, alpha=0.01, beta=1.0),
        Asset(ticker="B", sector="Tech", side=LONG, alpha=0.02, beta=1.0),
        Asset(ticker="Q", sector="Tech", side=LONG, alpha=0.001, beta=1.0),
        Asset(ticker="Y", sector="Energy", side=SHORT, alpha=-0.01, beta=1.0),
        Asset(ticker="X", sector="Energy", side=SHORT, alpha=-0.01, beta=1.0),
        Asset(ticker="Z", sector="Energy", side=SHORT, alpha=-0.02, beta=1.0),
    ]
    weights = {"A": 0.05, "B": 0.10, "Y": -0.05, "X": -0.05, "Z": -0.10}
    tickers = ["Z", "Q", "X", "A", "Y", "B"]
    sigma = pd.DataFrame(0.04 * np.eye(6), index=tickers, columns=tickers)
    for left, right, cov in [
        ("A", "B", 0.01),
        ("X", "Z", 0.01),
        ("Q", "A", 0.02),
        ("Q", "B", 0.02),
    ]:
        sigma.loc[left, right] = sigma.loc[right, left] = cov
    out = _assemble(AS_OF, assets, weights, sigma, {"Q": "optimizer:zero"})
    assert [p.ticker for p in out.positions] == ["B", "A", "Z", "X", "Y"]
    assert out.excluded == {"Q": "optimizer:zero"}
    assert out.gross == pytest.approx(0.35)  # 0.05 + 0.1 + 0.05 + 0.05 + 0.1
    assert out.net == pytest.approx(-0.05)  # 0.15 - 0.2
    assert out.beta == pytest.approx(-0.05)  # every beta 1
    assert out.expected_vol == pytest.approx(math.sqrt(0.0013))


def test_rejects_as_of_mismatch() -> None:
    with pytest.raises(ValueError, match="as_of"):
        _build(_research(as_of=date(2026, 9, 29)))


@pytest.mark.parametrize(
    "benchmark",
    [_benchmark().iloc[:-1], _benchmark().iloc[:0]],
    ids=["ends-early", "empty"],
)
def test_rejects_benchmark_not_ending_with_the_snapshot(benchmark: pd.Series) -> None:
    with pytest.raises(ValueError, match="benchmark must end on"):
        build_portfolio(_research(), _snapshot(), benchmark)


def test_rejects_benchmark_with_a_row_after_the_snapshot() -> None:
    extra = pd.Series([1.0], index=[INDEX[-1] + pd.offsets.BDay(1)])
    benchmark = pd.concat([_benchmark(), extra])
    with pytest.raises(ValueError, match="benchmark must end on"):
        build_portfolio(_research(), _snapshot(), benchmark)


def test_validate_inputs_rejects_a_benchmark_date_mismatch() -> None:
    with pytest.raises(ValueError, match="benchmark must end on"):
        validate_inputs(_research(), _snapshot(), _benchmark().iloc[:-1])


def test_tz_aware_benchmark_aligned_with_a_naive_snapshot_builds() -> None:
    aware = _benchmark()
    aware.index = aware.index.tz_localize("America/New_York")
    validate_inputs(_research(), _snapshot(), aware)
    out = build_portfolio(_research(), _snapshot(), aware, risk=RISK)
    assert out == _build(_research())


def test_validate_inputs_accepts_consistent_inputs() -> None:
    validate_inputs(_research(), _snapshot(), _benchmark())


def test_rejects_duplicate_research_tickers() -> None:
    with pytest.raises(ValueError, match=r"duplicate.*'AAA', 'BBB'"):
        validate_inputs(_research("BBB", "AAA", "BBB", "CCC", "AAA"), _snapshot(), _benchmark())


def _with_candidate(ticker: str, sector: str) -> ResearchResult:
    candidate = Candidate(ticker=ticker, sector=sector, side=LONG, depth=1, composite=1.0)
    name = dataclasses.replace(_research("AAA").names[0], candidate=candidate)
    return ResearchResult(as_of=AS_OF, names=(name,))


def test_rejects_ticker_missing_from_snapshot() -> None:
    with pytest.raises(ValueError, match="'QQQ' is not in the snapshot"):
        _build(_with_candidate("QQQ", "Tech"))


def test_rejects_sector_mismatch() -> None:
    with pytest.raises(ValueError, match="'AAA' sector 'Energy' differs"):
        _build(_with_candidate("AAA", "Energy"))


def test_naive_dates_keeps_the_wall_clock_date() -> None:
    aware = _benchmark()
    aware.index = aware.index.tz_localize("America/New_York")
    out = naive_dates(aware)
    assert out.index.tz is None
    assert out.index.equals(INDEX)
    plain = _benchmark()
    assert naive_dates(plain) is plain


def test_validate_benchmark_end_normalises_and_names_what_it_got() -> None:
    aware = _benchmark()
    aware.index = aware.index.tz_localize("America/New_York")
    validate_benchmark_end(_snapshot(), aware)
    with pytest.raises(ValueError, match="but ends on nothing"):
        validate_benchmark_end(_snapshot(), _benchmark().iloc[:0])
