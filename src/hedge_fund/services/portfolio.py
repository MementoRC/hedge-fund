"""Fetch the benchmark and wire research, prices and risk into a target portfolio."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from datetime import date

import numpy as np
import pandas as pd

from hedge_fund.domain.portfolio import Position, TargetPortfolio
from hedge_fund.domain.research import ResearchResult
from hedge_fund.domain.shortlist import Side
from hedge_fund.domain.snapshot import PriceSnapshot
from hedge_fund.engine.portfolio.alpha import DEFAULT_ALPHA, AlphaParams, alphas, gate, score
from hedge_fund.engine.portfolio.alpha import _validate_params as _validate_alpha
from hedge_fund.engine.portfolio.correlation import (
    DEFAULT_CORRELATION,
    CorrelationParams,
    correlation_filter,
)
from hedge_fund.engine.portfolio.correlation import _validate_params as _validate_correlation
from hedge_fund.engine.portfolio.optimize import (
    DEFAULT_OPTIMIZE,
    Asset,
    OptimizeParams,
    optimize_weights,
    validate_params,
)
from hedge_fund.engine.portfolio.risk_model import DEFAULT_RISK, RiskParams, risk_model
from hedge_fund.engine.portfolio.risk_model import _validate_params as _validate_risk
from hedge_fund.ports.data import DataProvider
from hedge_fund.services.snapshots import _calendar_window

BENCHMARK = "SPY"


def fetch_benchmark(
    provider: DataProvider, as_of: date, *, lookback_days: int, ticker: str = BENCHMARK
) -> pd.Series:
    """Adjusted closes of ``ticker`` up to ``as_of``, NaN rows dropped.

    The calendar start covers ``lookback_days + 1`` trading rows (one extra close for the
    first return). Raises ValueError when no close is left; the provider returns an all-NaN
    column for an unknown ticker rather than an empty frame.
    """
    frame = provider.closes([ticker], as_of - _calendar_window(lookback_days + 1), as_of)
    closes = frame[ticker].dropna()
    closes = closes.loc[closes.index <= pd.Timestamp(as_of)]
    if closes.empty:
        raise ValueError(f"no {ticker!r} closes up to {as_of}")
    return closes


def build_portfolio(
    research: ResearchResult,
    snapshot: PriceSnapshot,
    benchmark: pd.Series,
    *,
    alpha: AlphaParams = DEFAULT_ALPHA,
    risk: RiskParams = DEFAULT_RISK,
    correlation: CorrelationParams = DEFAULT_CORRELATION,
    optimize: OptimizeParams = DEFAULT_OPTIMIZE,
) -> TargetPortfolio:
    """Gate -> risk model -> alpha -> correlation filter -> optimizer (pure orchestration).

    Every researched ticker ends up either in ``positions`` or in ``excluded`` with a
    reason, never both. When no name is left after a step the rest are skipped and the
    portfolio is empty (all statistics 0.0). Raises ValueError when ``research`` and
    ``snapshot`` disagree on ``as_of``, when ``benchmark`` is empty or does not end on the
    snapshot's last date, when a researched ticker is missing from the snapshot or has
    another sector there, or when any of the four param sets is invalid (all are checked
    up front, even for empty research); engine errors (short or flat benchmark,
    PortfolioInfeasibleError) propagate.
    """
    validate_inputs(research, snapshot, benchmark)
    _validate_alpha(alpha)
    _validate_risk(risk)
    _validate_correlation(correlation)
    validate_params(optimize)
    eligible, excluded = gate(research.names)
    if not eligible:
        return _assemble(research.as_of, [], {}, pd.DataFrame(), excluded)
    model = risk_model(
        snapshot.closes[[name.candidate.ticker for name in eligible]], benchmark, params=risk
    )
    excluded |= model.excluded
    modelled = [name for name in eligible if name.candidate.ticker in model.vols]
    if not modelled:
        return _assemble(research.as_of, [], {}, pd.DataFrame(), excluded)
    expected = alphas(modelled, model.vols, params=alpha)
    candidates = {name.candidate.ticker: name.candidate for name in modelled}
    kept, correlated = correlation_filter(
        {name.candidate.ticker: score(name) for name in modelled},
        {ticker: candidate.side for ticker, candidate in candidates.items()},
        model.corr,
        params=correlation,
    )
    excluded |= correlated
    assets = [
        Asset(
            ticker=ticker,
            sector=candidates[ticker].sector,
            side=candidates[ticker].side,
            alpha=expected[ticker],
            beta=model.betas[ticker],
        )
        for ticker in kept
    ]
    allocation = optimize_weights(assets, model.sigma, params=optimize)
    excluded |= allocation.excluded
    return _assemble(research.as_of, assets, allocation.weights, model.sigma, excluded)


def validate_inputs(
    research: ResearchResult, snapshot: PriceSnapshot, benchmark: pd.Series
) -> None:
    if research.as_of != snapshot.as_of:
        raise ValueError(
            f"research as_of {research.as_of} differs from snapshot as_of {snapshot.as_of}"
        )
    last = snapshot.closes.index[-1]
    if benchmark.empty or benchmark.index[-1] != last:
        got = "nothing" if benchmark.empty else str(benchmark.index[-1].date())
        raise ValueError(
            f"benchmark must end on the snapshot's last date {last.date()}, but ends on {got}"
        )
    tickers = [name.candidate.ticker for name in research.names]
    duplicates = sorted({ticker for ticker in tickers if tickers.count(ticker) > 1})
    if duplicates:
        raise ValueError(f"duplicate research tickers: {', '.join(map(repr, duplicates))}")
    for name in research.names:
        ticker, sector = name.candidate.ticker, name.candidate.sector
        if ticker not in snapshot.closes.columns:
            raise ValueError(f"{ticker!r} is not in the snapshot")
        if snapshot.sectors[ticker] != sector:
            raise ValueError(
                f"{ticker!r} sector {sector!r} differs from the snapshot's "
                f"{snapshot.sectors[ticker]!r}"
            )


def _assemble(
    as_of: date,
    assets: Sequence[Asset],
    weights: Mapping[str, float],
    sigma: pd.DataFrame,
    excluded: Mapping[str, str],
) -> TargetPortfolio:
    """Order positions and compute book statistics; no weights gives all-zero statistics."""
    positions = sorted(
        (
            Position(
                ticker=asset.ticker,
                sector=asset.sector,
                side=asset.side,
                weight=weights[asset.ticker],
                alpha=asset.alpha,
                beta=asset.beta,
            )
            for asset in assets
            if asset.ticker in weights
        ),
        key=lambda p: (p.side is not Side.LONG, -abs(p.weight), p.ticker),
    )
    tickers = [position.ticker for position in positions]
    w = np.array([position.weight for position in positions], dtype=float)
    cov = sigma.reindex(index=tickers, columns=tickers).to_numpy(dtype=float)
    return TargetPortfolio(
        as_of=as_of,
        positions=tuple(positions),
        excluded=dict(sorted(excluded.items())),
        gross=float(np.abs(w).sum()),
        net=float(w.sum()),
        beta=float(sum(position.beta * position.weight for position in positions)),
        expected_vol=math.sqrt(max(float(w @ cov @ w), 0.0)),
    )
