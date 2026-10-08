# Factor Engine — First Slice (Design)

- **Date:** 2026-10-08
- **Status:** Approved design, pending implementation plan
- **Architecture stage:** 1 (Factor engine) of `docs/DIRECTION.md`

## Goal

Deliver the first pure, testable piece of the deterministic quant core: domain types for a
point-in-time price snapshot and scoring results, plus a sector-neutral factor scoring
pipeline with two price-only factors. This fixes the data shapes that the `DataProvider`
port, the composite/shortlist stage and the evaluation harness will build on.

## Scope

In scope:

- `PriceSnapshot` and score result domain types (frozen dataclasses).
- Two factors: 12-1 momentum and low volatility.
- Cross-sectional winsorization, sector-neutral z-scores, equal-weight composite.
- Explicit exclusion of unscorable tickers with reasons.

Out of scope (later specs): crowding check, Piotroski/Altman quality gates, fundamentals,
remaining factors of the eventual 8, shortlist/top-N selection, non-equal factor weights,
the `DataProvider` port and any adapter or I/O.

## Constraints

- `engine/` and `domain/` are pure: no I/O, no network, no LLM, no global mutable state.
- Inputs use pandas; outputs are frozen dataclasses.
- Deterministic: identical inputs produce identical outputs, including ordering.
- No look-ahead: only data at or before `as_of` can influence a score.
- Bad data never raises; contract violations raise `ValueError`.

## Approach

Pure-function pipeline. Each factor is a plain function `(window: pd.DataFrame) -> pd.Series`
returning a raw value per ticker, declared in a module-level tuple of `FactorSpec`. A single
`score()` function owns exclusion, winsorization, z-scoring and the composite.

Rejected alternatives:

- *Factor protocol + decorator registry*: registry is global mutable state in a pure module;
  unnecessary ceremony for two factors.
- *Single monolithic function*: couples factor definitions to normalization; would need
  splitting by the third factor.

## Components

```
src/hedge_fund/domain/
  snapshot.py        PriceSnapshot
  scores.py          FactorScore, ScoreResult
src/hedge_fund/engine/factors/
  definitions.py     FactorSpec, momentum_12_1, low_vol, FACTORS
  normalize.py       winsorize, sector_zscore
  score.py           score
```

### `PriceSnapshot` (frozen)

| Field     | Type                 | Meaning                                  |
|-----------|----------------------|------------------------------------------|
| `as_of`   | `datetime.date`      | Decision date                            |
| `closes`  | `pd.DataFrame`       | Close prices, index = dates, columns = tickers |
| `sectors` | `Mapping[str, str]`  | Ticker → sector label                    |

`__post_init__` raises `ValueError` when:

- `closes` is empty (no rows or no columns);
- the index is not a `DatetimeIndex`, is not sorted ascending, or has duplicates;
- any index date is after `as_of`;
- any column ticker has no entry in `sectors` (extra `sectors` keys are allowed).

Staleness is not checked in this slice. An `as_of` later than the last row is valid, and
the last row is treated as `t`.

### `FactorScore` (frozen)

`ticker: str`, `sector: str`, `raw: Mapping[str, float]`, `z: Mapping[str, float]`,
`composite: float`, `fallback_universe_z: bool`.

`raw` and `z` are keyed by factor name. `fallback_universe_z` is `True` when the ticker's
z-scores used universe rather than sector statistics.

### `ScoreResult` (frozen)

`as_of: date`, `scores: tuple[FactorScore, ...]`, `excluded: Mapping[str, str]`
(ticker → reason).

### `FactorSpec` (frozen)

`name: str`, `fn: Callable[[pd.DataFrame], pd.Series]`, `sign: int` (+1 higher-is-better,
−1 lower-is-better; applied before normalization), `min_history: int` (rows required).

`FACTORS = (FactorSpec("momentum_12_1", momentum_12_1, +1, 253),
FactorSpec("low_vol", low_vol, +1, 253))`.

## Data flow — `score(snapshot, factors=FACTORS) -> ScoreResult`

1. **Validate factors.** Duplicate factor names → `ValueError`.
2. **Window.** Take the last `max(spec.min_history)` rows (253: 252 daily returns).
3. **Pre-exclude** on the *unfilled* window. For each ticker, record the first failing
   reason, checked in this order:
   1. `insufficient_history`: the snapshot has fewer than 253 rows, or the ticker's first
      valid price comes after the window's first row. An all-NaN column counts as this case. The window size governs this check,
      not each factor's own `min_history`.
   2. `missing_data`: the ticker's NaN count divided by the window row count is above 0.05.
   3. `nonpositive_price`: any non-NaN close is ≤ 0.
4. **Fill.** For the remaining tickers, forward-fill interior and trailing NaNs within the
   window. Never back-fill. (A leading NaN has already been excluded by rule 3.1.)
5. **Raw factors** on the filled window, where `t` is the last row:
   - `momentum_12_1 = P[t−21] / P[t−252] − 1`
   - `low_vol = −std(log returns over the window, ddof=0) × √252`. It is negated so that
     higher is better. Zero variance gives 0, which is a valid value.

   **Post-exclude:** a ticker with any raw factor that is NaN or ±inf is excluded with
   reason `nonfinite_factor` and removed from the scorable universe before step 6.
6. **Sign.** Multiply each raw series by its `spec.sign` for normalization (`raw` in the
   output keeps the unsigned value).
7. **Winsorize** each factor cross-sectionally at the 1st and 99th percentiles of the
   scorable universe.
8. **Sector z-score.** `z = (x − mean_sector) / std_sector` with `ddof=0`.
   - If a sector has fewer than 3 scorable tickers, or its std is 0, its tickers use the
     universe mean/std and get `fallback_universe_z=True`.
   - If the universe std is 0, z = 0 for every ticker.
9. **Composite** = equal-weight mean of the ticker's z values.
10. **Order** `scores` by composite descending, ties by ticker ascending.

If every ticker is excluded, return `scores=()` with all reasons populated. This is not an
error.

## Error handling

| Condition                                   | Behaviour            |
|---------------------------------------------|----------------------|
| Snapshot contract violation                 | `ValueError` at construction |
| Duplicate factor names                      | `ValueError` from `score` |
| Short history / NaNs / bad prices / NaN factor | Ticker in `excluded` with reason |
| Small or zero-variance sector               | Universe fallback, flagged |
| All tickers excluded                        | Empty `scores`, no error |

## Testing

TDD; hand-built, seeded fixtures; no I/O.

- `tests/factories.py`: `make_snapshot(...)` builds geometric price paths per ticker with
  noise from `numpy.random.default_rng(0)`.
- `tests/domain/test_snapshot.py`: one test per contract violation; valid construction;
  field reassignment raises `FrozenInstanceError`. Frozen dataclasses do not deep-freeze
  the `DataFrame` or `Mapping` they hold, and the engine must not mutate its inputs.
- `tests/engine/factors/test_definitions.py`: closed-form checks. Constant daily growth `r`
  gives momentum `(1+r)^231 − 1` and `low_vol` 0; an alternating series gives an exact std.
- `tests/engine/factors/test_normalize.py`: winsorize clips only the tails; per-sector z has
  mean 0, std 1; small or zero-std sector falls back and flags; universe std 0 gives z = 0.
- `tests/engine/factors/test_score.py`:
  - each exclusion reason, and the order when several apply;
  - an interior or trailing NaN is forward-filled and the ticker is scored; a leading NaN
    is excluded as `insufficient_history` (never back-filled);
  - all-excluded returns empty scores;
  - deterministic tie-break;
  - duplicate factor names raise;
  - look-ahead guard: a snapshot with rows after `as_of` is rejected. Separately, build a
    long frame, perturb every price after date `T`, then truncate both versions with
    `.loc[:T]`; their scores at `as_of = T` must be identical;
  - shuffling ticker column order does not change the result;
  - a synthetic factor with `sign = −1` is negated before normalization while `raw` keeps
    the unsigned value (covers the sign path; neither shipped factor uses −1).
- Coverage: 100% of `domain/` and `engine/factors/`. Tests pass in the `dev`, `py311` and
  `py312` environments.

## Open questions (deferred, not blocking)

- Universe definition and point-in-time S&P 500 membership (survivorship source).
- Sector classification scheme (GICS or other).
- The remaining six factors and fundamentals lag handling.
- Rebalance cadence; shortlist N; research K and LLM budget.
