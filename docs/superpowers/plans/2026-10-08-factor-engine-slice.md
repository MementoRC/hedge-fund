# Factor Engine First Slice Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the pure factor-scoring slice from `docs/superpowers/specs/2026-10-08-factor-engine-slice-design.md`. It covers `PriceSnapshot`, `FactorScore` and `ScoreResult` domain types, the momentum_12_1 and low_vol factors, winsorization with sector-neutral z-scores, and `score()`.

**Architecture:** Pure-function pipeline. Factors are plain functions declared in a `FACTORS` tuple of frozen `FactorSpec`. `normalize.py` holds the cross-sectional math, and `score.py` orchestrates exclusion → fill → raw factors → post-exclusion → sign → winsorize → sector z → composite. There is no I/O anywhere.

**Tech Stack:** Python 3.11–3.13, pandas ≥ 3.0, numpy, pytest, ruff, mypy. All commands run through pixi.

---

## Conventions for every task

- Repo: `/home/memento/PycharmProjects/hedge-fund`, branch `development`. Do not create branches or worktrees.
- Run commands through pixi: `pixi run -e dev <cmd>`. Use the pixi-task MCP or shell-runner MCP; the Bash tool is blocked.
- Git goes through `mcp__git__execute_tool` (`git_add` with an explicit `files` list, then `git_commit`). Every commit message ends with the line:
  `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`
- Before each commit, `pixi run -e dev quality` (ruff check, ruff format --check, mypy src/) and `pixi run -e dev test` must pass. If ruff format complains, run `pixi run -e dev format` and re-check.
- Follow TDD: write the test, watch it fail, implement, watch it pass.
- Do not edit `../TradingAgents`.

## File map

| File | Responsibility |
|---|---|
| `src/hedge_fund/domain/snapshot.py` | `PriceSnapshot` + contract validation |
| `src/hedge_fund/domain/scores.py` | `FactorScore`, `ScoreResult` |
| `src/hedge_fund/engine/factors/definitions.py` | `FactorSpec`, `momentum_12_1`, `low_vol`, `FACTORS` |
| `src/hedge_fund/engine/factors/normalize.py` | `winsorize`, `sector_zscore` |
| `src/hedge_fund/engine/factors/score.py` | `score` orchestration + exclusions |
| `tests/__init__.py`, `tests/domain/__init__.py`, `tests/engine/__init__.py`, `tests/engine/factors/__init__.py` | Make tests a package so `tests.factories` imports and same-named test modules don't collide |
| `tests/factories.py` | Deterministic `make_closes`, `snapshot_from` |
| `tests/domain/test_snapshot.py`, `tests/domain/test_scores.py` | Domain tests |
| `tests/engine/factors/test_definitions.py`, `test_normalize.py`, `test_score.py` | Engine tests |
| `pyproject.toml` | mypy override for pandas (no bundled type stubs) |

---

### Task 1: PriceSnapshot, test package and fixtures

**Files:**
- Create: `src/hedge_fund/domain/snapshot.py`
- Create: `tests/__init__.py`, `tests/domain/__init__.py`, `tests/engine/__init__.py`, `tests/engine/factors/__init__.py` (each holds a one-line docstring)
- Create: `tests/factories.py`
- Test: `tests/domain/test_snapshot.py`
- Modify: `pyproject.toml` (append a mypy override after the `[tool.mypy]` table)

- [ ] **Step 0: Check the layout**

Confirm that `src/hedge_fund/domain/__init__.py`, `src/hedge_fund/engine/__init__.py` and `src/hedge_fund/engine/factors/__init__.py` exist as package files. Leave their docstrings as they are. If any is missing, create it with a one-line docstring. If `domain` or `factors` is a module file (`domain.py`) rather than a package, STOP and report.

- [ ] **Step 1: Write the failing test**

`tests/domain/test_snapshot.py`:

```python
"""Tests for PriceSnapshot contract validation."""

import dataclasses
from datetime import date

import pandas as pd
import pytest

from hedge_fund.domain.snapshot import PriceSnapshot

DATES = pd.DatetimeIndex(["2026-01-02", "2026-01-05", "2026-01-06"])
SECTORS = {"AAA": "Tech", "BBB": "Energy"}
AS_OF = date(2026, 1, 6)


def _closes(index: pd.Index = DATES) -> pd.DataFrame:
    return pd.DataFrame({"AAA": [1.0, 2.0, 3.0], "BBB": [4.0, 5.0, 6.0]}, index=index)


def test_valid_snapshot_keeps_fields() -> None:
    closes = _closes()
    snap = PriceSnapshot(as_of=AS_OF, closes=closes, sectors=SECTORS)
    assert snap.as_of == AS_OF
    assert snap.closes is closes
    assert snap.sectors == SECTORS


def test_as_of_after_last_row_is_allowed() -> None:
    snap = PriceSnapshot(as_of=date(2026, 3, 1), closes=_closes(), sectors=SECTORS)
    assert snap.as_of == date(2026, 3, 1)


def test_extra_sector_keys_are_allowed() -> None:
    sectors = {**SECTORS, "ZZZ": "Utilities"}
    snap = PriceSnapshot(as_of=AS_OF, closes=_closes(), sectors=sectors)
    assert snap.sectors == sectors


@pytest.mark.parametrize(
    "closes",
    [pd.DataFrame(index=pd.DatetimeIndex([])), pd.DataFrame(index=DATES)],
    ids=["no-rows", "no-columns"],
)
def test_empty_closes_rejected(closes: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="empty"):
        PriceSnapshot(as_of=AS_OF, closes=closes, sectors=SECTORS)


def test_non_datetime_index_rejected() -> None:
    with pytest.raises(ValueError, match="DatetimeIndex"):
        PriceSnapshot(as_of=AS_OF, closes=_closes(pd.Index([0, 1, 2])), sectors=SECTORS)


def test_unsorted_index_rejected() -> None:
    index = pd.DatetimeIndex(["2026-01-05", "2026-01-02", "2026-01-06"])
    with pytest.raises(ValueError, match="sorted"):
        PriceSnapshot(as_of=AS_OF, closes=_closes(index), sectors=SECTORS)


def test_duplicate_dates_rejected() -> None:
    index = pd.DatetimeIndex(["2026-01-02", "2026-01-02", "2026-01-06"])
    with pytest.raises(ValueError, match="duplicate"):
        PriceSnapshot(as_of=AS_OF, closes=_closes(index), sectors=SECTORS)


def test_rows_after_as_of_rejected() -> None:
    with pytest.raises(ValueError, match="after as_of"):
        PriceSnapshot(as_of=date(2026, 1, 5), closes=_closes(), sectors=SECTORS)


def test_ticker_without_sector_rejected() -> None:
    with pytest.raises(ValueError, match="BBB"):
        PriceSnapshot(as_of=AS_OF, closes=_closes(), sectors={"AAA": "Tech"})


def test_snapshot_is_frozen() -> None:
    snap = PriceSnapshot(as_of=AS_OF, closes=_closes(), sectors=SECTORS)
    with pytest.raises(dataclasses.FrozenInstanceError):
        snap.as_of = date(2026, 2, 1)  # type: ignore[misc]
```

Also create the four `__init__.py` files under `tests/`, each with a one-line docstring such as `"""Test package."""`.

- [ ] **Step 2: Run the test and confirm it fails**

Run: `pixi run -e dev pytest tests/domain/test_snapshot.py -v`
Expected: a collection error, `ModuleNotFoundError: No module named 'hedge_fund.domain.snapshot'`.

- [ ] **Step 3: Implement**

`src/hedge_fund/domain/snapshot.py`:

```python
"""Point-in-time price snapshot: the factor engine's only input."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date

import pandas as pd


@dataclass(frozen=True, eq=False)
class PriceSnapshot:
    """Close prices (dates x tickers) known as of ``as_of``, with a sector per ticker.

    Frozen only prevents field reassignment; the engine must never mutate ``closes``.
    """

    as_of: date
    closes: pd.DataFrame
    sectors: Mapping[str, str]

    def __post_init__(self) -> None:
        closes = self.closes
        if closes.empty:
            raise ValueError("closes is empty")
        if not isinstance(closes.index, pd.DatetimeIndex):
            raise ValueError("closes index must be a DatetimeIndex")
        if not closes.index.is_monotonic_increasing:
            raise ValueError("closes index must be sorted ascending")
        if closes.index.has_duplicates:
            raise ValueError("closes index has duplicate dates")
        if closes.index[-1].date() > self.as_of:
            raise ValueError("closes has rows after as_of")
        missing = sorted(str(t) for t in closes.columns if t not in self.sectors)
        if missing:
            raise ValueError(f"tickers without sector: {missing}")
```

`eq=False` is intentional. Comparing DataFrames for equality is ambiguous, so the dataclass gets identity equality.

Append to `pyproject.toml` directly after the `[tool.mypy]` table:

```toml
[[tool.mypy.overrides]]
module = ["pandas", "pandas.*"]
ignore_missing_imports = true
```

- [ ] **Step 4: Run the test and confirm it passes**

Run: `pixi run -e dev pytest tests/domain/test_snapshot.py -v`
Expected: 11 passed.

- [ ] **Step 5: Add the fixtures module**

`tests/factories.py`:

```python
"""Deterministic price fixtures for engine tests."""

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from hedge_fund.domain.snapshot import PriceSnapshot

END_DATE = "2026-09-30"


def make_closes(tickers: Sequence[str], n_days: int = 300, seed: int = 0) -> pd.DataFrame:
    """Geometric random-walk closes on business days ending END_DATE, one column per ticker.

    Each ticker gets its own drift and volatility, so ties are practically impossible.
    """
    rng = np.random.default_rng(seed)
    index = pd.bdate_range(end=END_DATE, periods=n_days)
    data = {}
    for ticker in tickers:
        drift = rng.uniform(-0.001, 0.001)
        vol = rng.uniform(0.005, 0.03)
        data[ticker] = 100.0 * np.exp(np.cumsum(rng.normal(drift, vol, n_days)))
    return pd.DataFrame(data, index=index)


def snapshot_from(closes: pd.DataFrame, sectors: Mapping[str, str]) -> PriceSnapshot:
    """Build a snapshot whose as_of is the date of the last row."""
    return PriceSnapshot(as_of=closes.index[-1].date(), closes=closes, sectors=dict(sectors))
```

- [ ] **Step 6: Run the quality gate and the full suite**

Run: `pixi run -e dev quality` and then `pixi run -e dev test`.
Expected: both pass, with 12 tests (11 new plus the smoke test).

- [ ] **Step 7: Commit**

Files: `src/hedge_fund/domain/snapshot.py`, `tests/__init__.py`, `tests/domain/__init__.py`, `tests/engine/__init__.py`, `tests/engine/factors/__init__.py`, `tests/factories.py`, `tests/domain/test_snapshot.py`, `pyproject.toml` (plus any domain/engine `__init__.py` created in Step 0).
Message: `feat(domain): add PriceSnapshot with contract validation`

---

### Task 2: FactorScore and ScoreResult

**Files:**
- Create: `src/hedge_fund/domain/scores.py`
- Test: `tests/domain/test_scores.py`

- [ ] **Step 1: Write the failing test**

`tests/domain/test_scores.py`:

```python
"""Tests for scoring result domain types."""

import dataclasses
from datetime import date

import pytest

from hedge_fund.domain.scores import FactorScore, ScoreResult


def _factor_score() -> FactorScore:
    return FactorScore(
        ticker="AAA",
        sector="Tech",
        raw={"momentum_12_1": 0.1},
        z={"momentum_12_1": 1.0},
        composite=1.0,
        fallback_universe_z=False,
    )


def test_score_result_holds_fields() -> None:
    result = ScoreResult(
        as_of=date(2026, 1, 6),
        scores=(_factor_score(),),
        excluded={"BBB": "missing_data"},
    )
    assert result.as_of == date(2026, 1, 6)
    assert result.scores[0].ticker == "AAA"
    assert result.excluded == {"BBB": "missing_data"}


def test_results_compare_by_value() -> None:
    assert _factor_score() == _factor_score()


def test_factor_score_is_frozen() -> None:
    score = _factor_score()
    with pytest.raises(dataclasses.FrozenInstanceError):
        score.composite = 2.0  # type: ignore[misc]
```

- [ ] **Step 2: Run the test and confirm it fails**

Run: `pixi run -e dev pytest tests/domain/test_scores.py -v`
Expected: `ModuleNotFoundError: No module named 'hedge_fund.domain.scores'`.

- [ ] **Step 3: Implement**

`src/hedge_fund/domain/scores.py`:

```python
"""Factor scoring results."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class FactorScore:
    """One ticker's factor values. ``raw`` and ``z`` are keyed by factor name."""

    ticker: str
    sector: str
    raw: Mapping[str, float]
    z: Mapping[str, float]
    composite: float
    fallback_universe_z: bool


@dataclass(frozen=True)
class ScoreResult:
    """Scores sorted by composite descending (ties by ticker), plus excluded tickers."""

    as_of: date
    scores: tuple[FactorScore, ...]
    excluded: Mapping[str, str]
```

- [ ] **Step 4: Run the test and confirm it passes**

Run: `pixi run -e dev pytest tests/domain/test_scores.py -v`
Expected: 3 passed.

- [ ] **Step 5: Quality gate, suite, commit**

Run `pixi run -e dev quality` and `pixi run -e dev test`; both must pass.
Files: `src/hedge_fund/domain/scores.py`, `tests/domain/test_scores.py`.
Message: `feat(domain): add FactorScore and ScoreResult`

---

### Task 3: Factor definitions

**Files:**
- Create: `src/hedge_fund/engine/factors/definitions.py`
- Test: `tests/engine/factors/test_definitions.py`

- [ ] **Step 1: Write the failing test**

`tests/engine/factors/test_definitions.py`:

```python
"""Closed-form checks for factor definitions."""

import numpy as np
import pandas as pd
import pytest

from hedge_fund.engine.factors.definitions import FACTORS, low_vol, momentum_12_1

WINDOW = 253  # 252 daily returns


def _constant_growth(rate: float) -> pd.DataFrame:
    return pd.DataFrame({"AAA": 100.0 * (1 + rate) ** np.arange(WINDOW)})


def test_momentum_constant_growth() -> None:
    result = momentum_12_1(_constant_growth(0.001))
    assert result["AAA"] == pytest.approx(1.001**231 - 1)


def test_momentum_uses_rows_t_minus_252_and_t_minus_21() -> None:
    prices = np.full(WINDOW, 7.0)
    prices[0] = 50.0
    prices[231] = 100.0
    result = momentum_12_1(pd.DataFrame({"AAA": prices}))
    assert result["AAA"] == pytest.approx(1.0)


def test_low_vol_is_zero_for_constant_growth() -> None:
    result = low_vol(_constant_growth(0.001))
    assert result["AAA"] == pytest.approx(0.0, abs=1e-12)


def test_low_vol_alternating_returns() -> None:
    step = 0.02
    log_prices = np.log(100.0) + step * (np.arange(WINDOW) % 2)
    result = low_vol(pd.DataFrame({"AAA": np.exp(log_prices)}))
    assert result["AAA"] == pytest.approx(-step * np.sqrt(252))


def test_factor_registry() -> None:
    assert [spec.name for spec in FACTORS] == ["momentum_12_1", "low_vol"]
    assert {spec.min_history for spec in FACTORS} == {WINDOW}
    assert {spec.sign for spec in FACTORS} == {1}
```

- [ ] **Step 2: Run the test and confirm it fails**

Run: `pixi run -e dev pytest tests/engine/factors/test_definitions.py -v`
Expected: `ModuleNotFoundError: No module named 'hedge_fund.engine.factors.definitions'`.

- [ ] **Step 3: Implement**

`src/hedge_fund/engine/factors/definitions.py`:

```python
"""Factor definitions: pure functions from a price window to a raw value per ticker."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS = 252
SKIP_DAYS = 21


@dataclass(frozen=True)
class FactorSpec:
    """A named factor. ``sign`` is +1 if higher is better, -1 if lower is better."""

    name: str
    fn: Callable[[pd.DataFrame], pd.Series]
    sign: int
    min_history: int


def momentum_12_1(window: pd.DataFrame) -> pd.Series:
    """Return from t-252 to t-21, skipping the most recent month."""
    return window.iloc[-1 - SKIP_DAYS] / window.iloc[-1 - TRADING_DAYS] - 1.0


def low_vol(window: pd.DataFrame) -> pd.Series:
    """Negated annualized volatility of daily log returns (higher is calmer)."""
    log_returns = np.log(window).diff().iloc[1:]
    return -log_returns.std(ddof=0) * np.sqrt(TRADING_DAYS)


FACTORS: tuple[FactorSpec, ...] = (
    FactorSpec("momentum_12_1", momentum_12_1, sign=1, min_history=TRADING_DAYS + 1),
    FactorSpec("low_vol", low_vol, sign=1, min_history=TRADING_DAYS + 1),
)
```

- [ ] **Step 4: Run the test and confirm it passes**

Run: `pixi run -e dev pytest tests/engine/factors/test_definitions.py -v`
Expected: 5 passed.

- [ ] **Step 5: Quality gate, suite, commit**

Run `pixi run -e dev quality` and `pixi run -e dev test`; both must pass.
Files: `src/hedge_fund/engine/factors/definitions.py`, `tests/engine/factors/test_definitions.py`.
Message: `feat(engine): add momentum_12_1 and low_vol factor definitions`

---

### Task 4: Normalization

**Files:**
- Create: `src/hedge_fund/engine/factors/normalize.py`
- Test: `tests/engine/factors/test_normalize.py`

- [ ] **Step 1: Write the failing test**

`tests/engine/factors/test_normalize.py`:

```python
"""Tests for winsorization and sector-neutral z-scores."""

import numpy as np
import pandas as pd
import pytest

from hedge_fund.engine.factors.normalize import sector_zscore, winsorize


def test_winsorize_clips_only_tails() -> None:
    values = pd.Series(np.arange(101, dtype=float))
    result = winsorize(values)
    assert result.iloc[0] == pytest.approx(1.0)
    assert result.iloc[100] == pytest.approx(99.0)
    pd.testing.assert_series_equal(result.iloc[1:100], values.iloc[1:100])


def test_zscore_standardizes_within_each_sector() -> None:
    values = pd.Series([1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 45.0], index=list("abcdefgh"))
    sectors = pd.Series(["X"] * 4 + ["Y"] * 4, index=values.index)
    z, fallback = sector_zscore(values, sectors)
    for name in ("X", "Y"):
        members = sectors.index[sectors == name]
        assert z[members].mean() == pytest.approx(0.0, abs=1e-12)
        assert z[members].std(ddof=0) == pytest.approx(1.0)
    assert not fallback.any()


def test_small_sector_falls_back_to_universe() -> None:
    values = pd.Series([1.0, 2.0, 3.0, 4.0, 100.0, 200.0], index=list("abcdef"))
    sectors = pd.Series(["X"] * 4 + ["Y"] * 2, index=values.index)
    z, fallback = sector_zscore(values, sectors)
    small = ["e", "f"]
    expected = (values[small] - values.mean()) / values.std(ddof=0)
    pd.testing.assert_series_equal(z[small], expected)
    assert fallback[small].all()
    assert not fallback[["a", "b", "c", "d"]].any()


def test_zero_variance_sector_falls_back_to_universe() -> None:
    values = pd.Series([5.0, 5.0, 5.0, 1.0, 2.0, 3.0], index=list("abcdef"))
    sectors = pd.Series(["X"] * 3 + ["Y"] * 3, index=values.index)
    z, fallback = sector_zscore(values, sectors)
    flat = ["a", "b", "c"]
    expected = (values[flat] - values.mean()) / values.std(ddof=0)
    pd.testing.assert_series_equal(z[flat], expected)
    assert fallback[flat].all()
    assert not fallback[["d", "e", "f"]].any()


def test_zero_universe_variance_gives_zero_z() -> None:
    values = pd.Series([7.0] * 4, index=list("abcd"))
    sectors = pd.Series(["X"] * 4, index=values.index)
    z, fallback = sector_zscore(values, sectors)
    assert (z == 0.0).all()
    assert fallback.all()
```

- [ ] **Step 2: Run the test and confirm it fails**

Run: `pixi run -e dev pytest tests/engine/factors/test_normalize.py -v`
Expected: `ModuleNotFoundError: No module named 'hedge_fund.engine.factors.normalize'`.

- [ ] **Step 3: Implement**

`src/hedge_fund/engine/factors/normalize.py`:

```python
"""Cross-sectional normalization: winsorization and sector-neutral z-scores."""

from __future__ import annotations

import pandas as pd

MIN_SECTOR_SIZE = 3


def winsorize(values: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    """Clip ``values`` to its own [lower, upper] cross-sectional quantiles."""
    return values.clip(lower=values.quantile(lower), upper=values.quantile(upper))


def sector_zscore(values: pd.Series, sectors: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Z-score ``values`` within each sector (population std).

    Sectors with fewer than MIN_SECTOR_SIZE names, or zero std, use universe mean/std
    instead. Returns ``(z, fallback)`` indexed like ``values``; ``fallback`` marks tickers
    scored against the universe. Where the applicable std is 0, z is 0.
    """
    grouped = values.groupby(sectors)
    mean = grouped.transform("mean")
    std = grouped.transform(lambda group: group.std(ddof=0))
    count = grouped.transform("count")
    fallback = (count < MIN_SECTOR_SIZE) | (std == 0)
    mean = mean.where(~fallback, values.mean())
    std = std.where(~fallback, values.std(ddof=0))
    z = ((values - mean) / std).where(std != 0, 0.0)
    return z, fallback
```

- [ ] **Step 4: Run the test and confirm it passes**

Run: `pixi run -e dev pytest tests/engine/factors/test_normalize.py -v`
Expected: 5 passed. If `assert_series_equal` fails only on the Series `name` attribute, pass `check_names=False`. Do not change the math.

- [ ] **Step 5: Quality gate, suite, commit**

Run `pixi run -e dev quality` and `pixi run -e dev test`; both must pass.
Files: `src/hedge_fund/engine/factors/normalize.py`, `tests/engine/factors/test_normalize.py`.
Message: `feat(engine): add winsorize and sector-neutral z-score`

---

### Task 5: score() orchestration

**Files:**
- Create: `src/hedge_fund/engine/factors/score.py`
- Test: `tests/engine/factors/test_score.py`

- [ ] **Step 1: Write the failing test**

`tests/engine/factors/test_score.py`:

```python
"""Tests for the factor scoring pipeline."""

import numpy as np
import pandas as pd
import pytest

from hedge_fund.domain.snapshot import PriceSnapshot
from hedge_fund.engine.factors.definitions import FACTORS, FactorSpec, low_vol, momentum_12_1
from hedge_fund.engine.factors.score import score
from tests.factories import make_closes, snapshot_from

SECTORS = {
    "A1": "Tech",
    "A2": "Tech",
    "A3": "Tech",
    "A4": "Tech",
    "B1": "Energy",
    "B2": "Energy",
    "B3": "Energy",
    "B4": "Energy",
}
TICKERS = list(SECTORS)
WINDOW = 253


def _snapshot(closes: pd.DataFrame) -> PriceSnapshot:
    return snapshot_from(closes, SECTORS)


def _last_price(window: pd.DataFrame) -> pd.Series:
    return window.iloc[-1]


# --- happy path -------------------------------------------------------------


def test_scores_every_clean_ticker() -> None:
    snap = _snapshot(make_closes(TICKERS))
    result = score(snap)
    assert result.as_of == snap.as_of
    assert result.excluded == {}
    assert sorted(s.ticker for s in result.scores) == sorted(TICKERS)
    composites = [s.composite for s in result.scores]
    assert composites == sorted(composites, reverse=True)
    for s in result.scores:
        assert set(s.raw) == {"momentum_12_1", "low_vol"}
        assert s.composite == pytest.approx(sum(s.z.values()) / 2)
        assert s.sector == SECTORS[s.ticker]
        assert s.fallback_universe_z is False


def test_raw_values_match_factor_definitions() -> None:
    closes = make_closes(TICKERS)
    result = score(_snapshot(closes))
    window = closes.iloc[-WINDOW:]
    momentum = momentum_12_1(window)
    vol = low_vol(window)
    for s in result.scores:
        assert s.raw["momentum_12_1"] == pytest.approx(momentum[s.ticker])
        assert s.raw["low_vol"] == pytest.approx(vol[s.ticker])


def test_history_before_window_is_ignored() -> None:
    closes = make_closes(TICKERS, n_days=400)
    assert score(_snapshot(closes)) == score(_snapshot(closes.iloc[-WINDOW:]))


# --- exclusions -------------------------------------------------------------


def test_short_history_excludes_everything() -> None:
    result = score(_snapshot(make_closes(TICKERS, n_days=200)))
    assert result.scores == ()
    assert result.excluded == dict.fromkeys(TICKERS, "insufficient_history")


def test_leading_nan_and_all_nan_are_insufficient_history() -> None:
    closes = make_closes(TICKERS)  # 300 rows; window starts at row 47
    closes.iloc[:60, 0] = np.nan  # A1: leading NaNs inside the window
    closes["A2"] = np.nan
    result = score(_snapshot(closes))
    assert result.excluded == {"A1": "insufficient_history", "A2": "insufficient_history"}


def test_too_many_nans_is_missing_data() -> None:
    closes = make_closes(TICKERS)
    closes.iloc[100:120, 0] = np.nan  # 20 / 253 > 5%
    result = score(_snapshot(closes))
    assert result.excluded == {"A1": "missing_data"}


def test_interior_and_trailing_nans_are_forward_filled() -> None:
    closes = make_closes(TICKERS)
    closes.iloc[[150, 151, 152], 0] = np.nan
    closes.iloc[-1, 0] = np.nan
    result = score(_snapshot(closes))
    assert result.excluded == {}
    scored = {s.ticker: s for s in result.scores}
    expected = low_vol(closes.iloc[-WINDOW:].ffill())["A1"]
    assert scored["A1"].raw["low_vol"] == pytest.approx(expected)


def test_nonpositive_price_excluded() -> None:
    closes = make_closes(TICKERS)
    closes.iloc[200, 0] = 0.0
    closes.iloc[210, 1] = -5.0
    result = score(_snapshot(closes))
    assert result.excluded == {"A1": "nonpositive_price", "A2": "nonpositive_price"}


def test_nonfinite_factor_excluded() -> None:
    closes = make_closes(TICKERS)
    closes.iloc[200, 0] = np.inf
    result = score(_snapshot(closes))
    assert result.excluded == {"A1": "nonfinite_factor"}
    assert "A1" not in {s.ticker for s in result.scores}


def test_first_failing_reason_wins() -> None:
    closes = make_closes(TICKERS)
    closes.iloc[:60, 0] = np.nan  # A1: insufficient_history ...
    closes.iloc[200, 0] = 0.0  # ... beats nonpositive_price
    closes.iloc[100:120, 1] = np.nan  # A2: missing_data ...
    closes.iloc[200, 1] = -1.0  # ... beats nonpositive_price
    closes.iloc[150, 2] = np.inf  # A3: nonpositive_price ...
    closes.iloc[160, 2] = 0.0  # ... beats nonfinite_factor
    result = score(_snapshot(closes))
    assert result.excluded == {
        "A1": "insufficient_history",
        "A2": "missing_data",
        "A3": "nonpositive_price",
    }


# --- contract errors --------------------------------------------------------


def test_duplicate_factor_names_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate factor"):
        score(_snapshot(make_closes(TICKERS)), factors=(FACTORS[0], FACTORS[0]))


def test_empty_factor_list_rejected() -> None:
    with pytest.raises(ValueError, match="at least one factor"):
        score(_snapshot(make_closes(TICKERS)), factors=())


def test_snapshot_with_future_rows_rejected() -> None:
    closes = make_closes(TICKERS)
    with pytest.raises(ValueError, match="after as_of"):
        PriceSnapshot(as_of=closes.index[-2].date(), closes=closes, sectors=SECTORS)


# --- determinism and invariants --------------------------------------------


def test_ties_broken_by_ticker() -> None:
    closes = make_closes(TICKERS)
    closes["A2"] = closes["A1"]
    closes = closes[closes.columns[::-1]]
    order = [s.ticker for s in score(_snapshot(closes)).scores]
    assert order.index("A2") == order.index("A1") + 1


def test_prices_after_as_of_cannot_affect_scores() -> None:
    full = make_closes(TICKERS, n_days=400)
    cutoff = full.index[300]
    perturbed = full.copy()
    perturbed.loc[perturbed.index > cutoff] *= 3.0
    base = score(_snapshot(full.loc[:cutoff]))
    moved = score(_snapshot(perturbed.loc[:cutoff]))
    assert base == moved


def test_column_order_does_not_matter() -> None:
    closes = make_closes(TICKERS)
    base = score(_snapshot(closes))
    shuffled = score(_snapshot(closes[list(reversed(TICKERS))]))
    assert [s.ticker for s in shuffled.scores] == [s.ticker for s in base.scores]
    for a, b in zip(base.scores, shuffled.scores, strict=True):
        assert b.composite == pytest.approx(a.composite)
        assert b.z == pytest.approx(a.z)


def test_negative_sign_inverts_ranking_but_keeps_raw() -> None:
    closes = make_closes(TICKERS)
    spec = FactorSpec("last_price", _last_price, sign=-1, min_history=WINDOW)
    result = score(_snapshot(closes), factors=(spec,))
    last = closes.iloc[-1]
    for s in result.scores:
        assert s.raw["last_price"] == pytest.approx(last[s.ticker])
    tech = [s for s in result.scores if s.sector == "Tech"]
    cheapest = min(tech, key=lambda s: s.raw["last_price"])
    assert cheapest.z["last_price"] == max(s.z["last_price"] for s in tech)


def test_small_sector_flags_fallback() -> None:
    sectors = {**SECTORS, "C1": "Utilities", "C2": "Utilities"}
    closes = make_closes([*TICKERS, "C1", "C2"])
    result = score(snapshot_from(closes, sectors))
    flags = {s.ticker: s.fallback_universe_z for s in result.scores}
    assert flags.pop("C1") is True
    assert flags.pop("C2") is True
    assert not any(flags.values())
```

- [ ] **Step 2: Run the test and confirm it fails**

Run: `pixi run -e dev pytest tests/engine/factors/test_score.py -v`
Expected: `ModuleNotFoundError: No module named 'hedge_fund.engine.factors.score'`.

- [ ] **Step 3: Implement**

`src/hedge_fund/engine/factors/score.py`:

```python
"""Sector-neutral factor scoring pipeline."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from hedge_fund.domain.scores import FactorScore, ScoreResult
from hedge_fund.domain.snapshot import PriceSnapshot
from hedge_fund.engine.factors.definitions import FACTORS, FactorSpec
from hedge_fund.engine.factors.normalize import sector_zscore, winsorize

MAX_MISSING_FRACTION = 0.05


def score(snapshot: PriceSnapshot, factors: Sequence[FactorSpec] = FACTORS) -> ScoreResult:
    """Score every ticker in ``snapshot``; unscorable tickers land in ``excluded``."""
    if not factors:
        raise ValueError("score needs at least one factor")
    names = [spec.name for spec in factors]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate factor names: {names}")

    window_size = max(spec.min_history for spec in factors)
    window = snapshot.closes.iloc[-window_size:]

    excluded: dict[str, str] = {}
    for ticker in window.columns:
        reason = _pre_exclusion_reason(window[ticker], window_size)
        if reason is not None:
            excluded[ticker] = reason

    scorable = [t for t in window.columns if t not in excluded]
    if not scorable:
        return _result(snapshot, (), excluded)

    filled = window[scorable].ffill()
    raw = pd.DataFrame({spec.name: spec.fn(filled) for spec in factors})
    finite = np.isfinite(raw).all(axis=1)
    for ticker in raw.index[~finite]:
        excluded[ticker] = "nonfinite_factor"
    raw = raw[finite]
    if raw.empty:
        return _result(snapshot, (), excluded)

    sectors = pd.Series({t: snapshot.sectors[t] for t in raw.index})
    z = pd.DataFrame(index=raw.index)
    fallback = pd.Series(False, index=raw.index)
    for spec in factors:
        z_factor, fallback_factor = sector_zscore(winsorize(raw[spec.name] * spec.sign), sectors)
        z[spec.name] = z_factor
        fallback = fallback | fallback_factor
    composite = z.mean(axis=1)

    scores = [
        FactorScore(
            ticker=t,
            sector=snapshot.sectors[t],
            raw={n: float(raw.at[t, n]) for n in names},
            z={n: float(z.at[t, n]) for n in names},
            composite=float(composite[t]),
            fallback_universe_z=bool(fallback[t]),
        )
        for t in raw.index
    ]
    scores.sort(key=lambda s: (-s.composite, s.ticker))
    return _result(snapshot, tuple(scores), excluded)


def _pre_exclusion_reason(prices: pd.Series, window_size: int) -> str | None:
    """First failing data-quality rule on the unfilled window, or None."""
    first_valid = prices.first_valid_index()
    if len(prices) < window_size or first_valid is None or first_valid > prices.index[0]:
        return "insufficient_history"
    if prices.isna().mean() > MAX_MISSING_FRACTION:
        return "missing_data"
    if (prices.dropna() <= 0).any():
        return "nonpositive_price"
    return None


def _result(
    snapshot: PriceSnapshot, scores: tuple[FactorScore, ...], excluded: dict[str, str]
) -> ScoreResult:
    return ScoreResult(as_of=snapshot.as_of, scores=scores, excluded=dict(sorted(excluded.items())))
```

- [ ] **Step 4: Run the test and confirm it passes**

Run: `pixi run -e dev pytest tests/engine/factors/test_score.py -v`
Expected: 18 passed. The inf-price tests may emit numpy `RuntimeWarning`s, which is acceptable. If a test fails, debug the implementation against the spec. Do not weaken the test. Report any test you believe is itself wrong instead of editing it silently.

- [ ] **Step 5: Quality gate, suite, commit**

Run `pixi run -e dev quality` and `pixi run -e dev test`; both must pass.
Files: `src/hedge_fund/engine/factors/score.py`, `tests/engine/factors/test_score.py`.
Message: `feat(engine): add sector-neutral factor scoring pipeline`

---

### Task 6: Cross-version verification

**Files:** none expected.

- [ ] **Step 1: Run every environment**

Run `pixi run -e py311 test`, `pixi run -e py312 test`, `pixi run -e dev test` and `pixi run -e dev quality`.
Expected: all pass. Coverage of `hedge_fund/domain/snapshot.py`, `hedge_fund/domain/scores.py` and `hedge_fund/engine/factors/*.py` is 100% in the term-missing report.

- [ ] **Step 2: Close coverage gaps if any**

If any line in those modules is uncovered, add the smallest test that exercises it, in the matching test file, then re-run. Commit with the message `test(engine): close factor engine coverage gaps`. If there are no gaps, skip the commit.
