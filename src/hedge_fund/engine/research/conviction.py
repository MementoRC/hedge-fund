"""Side-aligned conviction verdict from K research ratings of one candidate."""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass

from hedge_fund.domain.research import NameResearch, RunRecord, Verdict
from hedge_fund.domain.shortlist import Candidate, Side


@dataclass(frozen=True)
class ConvictionParams:
    """Minimum valid runs, mean-score thresholds and the dispersion ceiling."""

    min_valid: int = 2
    confirm: float = 0.5
    contradict: float = -0.5
    max_std: float = 1.0


DEFAULT_CONVICTION = ConvictionParams()


def conviction(
    candidate: Candidate,
    runs: Sequence[RunRecord],
    *,
    params: ConvictionParams = DEFAULT_CONVICTION,
) -> NameResearch:
    """Side-aligned consensus and dispersion of ``runs`` for ``candidate``.

    A run counts only if ``RunRecord.valid``. Its score is multiplied by +1 for a long and
    -1 for a short, so a positive aligned score always supports the candidate's side.
    ``mean``, population ``std`` (ddof=0) and ``agreement`` (share of aligned scores > 0)
    are None when no run is valid. Verdict, first match: fewer than ``min_valid`` valid
    runs -> INSUFFICIENT; ``mean >= confirm`` and ``std <= max_std`` -> CONFIRM;
    ``mean <= contradict`` and ``std <= max_std`` -> CONTRADICT; else NEUTRAL. Raises
    ValueError on invalid params, on runs whose ticker or ``as_of`` differ from the
    candidate / the first run, and on a repeated ``run_index``. Empty ``runs`` is allowed.
    """
    _validate_params(params)
    ordered = _validated_runs(candidate, runs)
    sign = 1 if candidate.side is Side.LONG else -1
    scores = [run.rating.score for run in ordered if run.valid and run.rating is not None]
    aligned = [sign * score for score in scores if score is not None]
    mean: float | None = None
    std: float | None = None
    agreement: float | None = None
    if aligned:
        mean = statistics.fmean(aligned)
        std = statistics.pstdev(aligned)
        agreement = sum(1 for score in aligned if score > 0) / len(aligned)
    return NameResearch(
        candidate=candidate,
        runs=ordered,
        valid=len(aligned),
        mean=mean,
        std=std,
        agreement=agreement,
        verdict=_verdict(len(aligned), mean, std, params),
    )


def _verdict(
    valid: int, mean: float | None, std: float | None, params: ConvictionParams
) -> Verdict:
    if valid < params.min_valid or mean is None or std is None:
        return Verdict.INSUFFICIENT
    if std <= params.max_std:
        if mean >= params.confirm:
            return Verdict.CONFIRM
        if mean <= params.contradict:
            return Verdict.CONTRADICT
    return Verdict.NEUTRAL


def _validated_runs(candidate: Candidate, runs: Sequence[RunRecord]) -> tuple[RunRecord, ...]:
    """Runs sorted by ``run_index``; ValueError on foreign ticker, mixed as_of or repeats."""
    seen: set[int] = set()
    for run in runs:
        if run.ticker != candidate.ticker:
            raise ValueError(
                f"run {run.run_index} is for {run.ticker!r}, expected {candidate.ticker!r}"
            )
        if run.as_of != runs[0].as_of:
            raise ValueError(f"run {run.run_index} as_of {run.as_of} differs from {runs[0].as_of}")
        if run.run_index in seen:
            raise ValueError(f"duplicate run_index {run.run_index}")
        seen.add(run.run_index)
    return tuple(sorted(runs, key=lambda run: run.run_index))


def _validate_params(params: ConvictionParams) -> None:
    """Type check first: bool, float and NaN would slip past plain range comparisons."""
    min_valid = params.min_valid
    if isinstance(min_valid, bool) or not isinstance(min_valid, int) or min_valid < 1:
        raise ValueError(f"min_valid must be an int >= 1, got {min_valid!r}")
    for name in ("confirm", "contradict", "max_std"):
        value = getattr(params, name)
        if (
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(value)
        ):
            raise ValueError(f"{name} must be a finite number, got {value!r}")
    if not params.contradict < 0 < params.confirm:
        raise ValueError(
            "thresholds must satisfy contradict < 0 < confirm, "
            f"got contradict={params.contradict!r}, confirm={params.confirm!r}"
        )
    if not params.max_std > 0:
        raise ValueError(f"max_std must be > 0, got {params.max_std!r}")
