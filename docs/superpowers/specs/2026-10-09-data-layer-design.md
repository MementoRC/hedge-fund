# Data Layer — First Slice (Design)

- **Date:** 2026-10-09
- **Status:** Approved design, pending implementation plan
- **Architecture stage:** 0 (Data layer) of `docs/DIRECTION.md`
- **Depends on:** factor engine slice (`2026-10-08-factor-engine-slice-design.md`)

## Goal

Feed the factor engine real data. Provide a `DataProvider` port, a yfinance adapter, a
per-ticker point-in-time price cache, sector resolution, a shipped S&P 500 universe file,
frozen snapshot storage for evaluation runs, and `build_snapshot()` that turns
`(tickers, as_of)` into a validated `PriceSnapshot`. After this slice,
`score(build_snapshot(...).snapshot)` runs end to end on real prices.

## Scope

In scope:

- `DataProvider` protocol (prices + sectors).
- `YFinanceProvider` with injectable vendor calls.
- `CachingProvider` decorator with a per-ticker parquet cache.
- `StaticSectorProvider` decorator backed by the shipped universe file.
- `load_universe("sp500")` plus a dev-only refresh script.
- Frozen snapshot save/load.
- `build_snapshot()` service.

Out of scope (later specs): point-in-time index membership (survivorship-free universe),
fundamentals, FRED, SEC EDGAR, other vendors (Polygon, Alpha Vantage), intraday data,
retries/rate limiting beyond what yfinance does itself.

Known bias, accepted for this slice: the universe is a static current list, so backtests
over past dates carry survivorship bias.

## Constraints

- `engine/` and `domain/` stay pure; all I/O lives in `adapters/`. `services/` may call
  ports but performs no I/O itself.
- The default test suite and CI make no network calls.
- Prices are dividend- and split-adjusted closes (`auto_adjust=True`).
- Paths are passed in explicitly; nothing writes outside a caller-provided directory.

## Approach

One port with decorators. `DataProvider` is a `typing.Protocol`; caching and static sector
lookup are decorators that implement the same protocol and wrap an inner provider. This
keeps vendor code, caching and sector policy independently replaceable, and lets tests use
an in-memory fake.

Rejected alternatives:

- *Single yfinance provider doing fetch, cache and snapshot*: welds caching and sector
  policy to one vendor; breaks when a second vendor arrives.
- *Separate price and sector ports*: extra ceremony while one vendor serves both; easy to
  split later if needed.

## Components

```
src/hedge_fund/
  ports/data.py                    DataProvider, DataProviderError
  adapters/data/
    yfinance_provider.py           YFinanceProvider, YAHOO_TO_GICS
    caching.py                     CachingProvider
    static_sectors.py              StaticSectorProvider
    frozen.py                      save_snapshot, load_snapshot
  universe.py                      Universe, load_universe, GICS_SECTORS
  services/snapshots.py            SnapshotBuild, build_snapshot
  resources/universes/sp500.csv    shipped universe (package data)
scripts/refresh_sp500.py           dev-only regeneration of sp500.csv
```

The universe file lives under `resources/` because the `.gitignore` pattern `data/` would
also match `src/hedge_fund/data/`.

### `DataProvider` (Protocol)

```python
def closes(self, tickers: Sequence[str], start: date, end: date) -> pd.DataFrame: ...
def sectors(self, tickers: Sequence[str]) -> dict[str, str | None]: ...
```

- `closes` returns adjusted closes with a tz-naive `DatetimeIndex` (sorted, unique) and
  one column per requested ticker, in request order, covering `start..end` inclusive.
  A ticker with no data is an all-NaN column, not an error.
- `sectors` returns a GICS sector name per requested ticker, or `None` when unknown.
- Transport or vendor failures raise `DataProviderError`.

### `YFinanceProvider(download=yf.download, info_fetcher=lambda t: yf.Ticker(t).info)`

- `closes`: calls `download(list(tickers), start=start, end=end + 1 day,
  auto_adjust=True, actions=False, progress=False)` and takes the `Close` field.
  Normalizes yfinance's shapes (MultiIndex columns, the single-ticker frame, empty
  results) into the protocol's frame; strips timezones; reindexes columns to the request.
- `sectors`: `info_fetcher(ticker)` returns the Yahoo `info` mapping; its `"sector"` value
  is mapped through `YAHOO_TO_GICS` to one of the 11 GICS sectors. Unmapped or missing
  values become `None`.
- Any exception from `download` or `info_fetcher` is re-raised as `DataProviderError`
  (with the original as `__cause__`). yfinance reports per-ticker failures as empty data
  rather than raising; those surface as all-NaN columns, which `CachingProvider` does not
  cache.

`YAHOO_TO_GICS`: Technology → Information Technology; Financial Services → Financials;
Healthcare → Health Care; Consumer Cyclical → Consumer Discretionary; Consumer Defensive
→ Consumer Staples; Communication Services → Communication Services; Industrials →
Industrials; Energy → Energy; Utilities → Utilities; Real Estate → Real Estate; Basic
Materials → Materials.

### `CachingProvider(inner, cache_dir, clock=date.today)`

Layout under `cache_dir`:

- `prices/<TICKER>.parquet`: one column `close`, date index.
- `prices/<TICKER>.json`: `{"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}`, the covered
  range. Coverage is recorded explicitly because holidays and halts make it impossible
  to infer from the data.
- `sectors.json`: ticker → sector, for tickers the inner provider resolved to a non-`None`
  sector.

`closes(tickers, start, end)`:

1. For each ticker, the request is *covered* if a cache exists and
   `cov_start <= start` and `end <= cov_end`. Covered tickers are served from the cache
   with no `inner` call.
2. Every uncovered ticker is refetched over its full union range
   `min(start, cov_start) .. max(end, cov_end)` (just `start..end` if there is no cache),
   and the fetched series *replaces* the cached one. Splicing is never done: adjusted
   closes are restated after dividends and splits, so mixing fetches taken at different
   times would create artificial jumps. Uncovered tickers that share a union range are
   fetched in one `inner.closes` call.
3. Recorded coverage is `fetch_start .. min(fetch_end, clock() − 1 day)`. Today's
   partial bar and future dates are never marked covered, so any request with
   `end >= clock()` refetches; this is intended. If the capped end falls before
   `fetch_start`, nothing is written to the cache for that ticker.
4. A ticker whose fetch returned zero non-NaN rows writes nothing: an existing cache and
   coverage record are kept unchanged, and the response serves whatever cached rows fall
   in `start..end` (all-NaN if there are none). The ticker is retried on the next call.
   yfinance reports per-ticker failures (bad symbol, throttling) as empty data rather
   than exceptions.
5. The response is assembled in request order and sliced to `start..end`.

`sectors(tickers)`: serve tickers known in `sectors.json`; ask `inner` for the rest;
persist only non-`None` answers, so throttled or empty `info` responses are retried.

All cache writes are atomic: write to a temporary file in the same directory, then
`os.replace`. The parquet is written before its coverage JSON, so an interrupted run
leaves at worst a parquet file with no coverage record, which is treated as uncached.

### `StaticSectorProvider(inner, universe)`

- `sectors`: universe sectors first; `inner.sectors()` only for tickers not in the
  universe. Returns the merged result in request order.
- `closes`: delegates to `inner` unchanged.

### Universe

`resources/universes/sp500.csv`: first line `# list_date: YYYY-MM-DD`, then a header
`ticker,gics_sector`, one row per constituent. Tickers use Yahoo's symbol format (e.g.
`BRK-B`).

`load_universe(name: str) -> Universe` reads `resources/universes/<name>.csv` through
`importlib.resources`. `Universe` is frozen: `name`, `list_date: date`,
`tickers: tuple[str, ...]` (file order), `sectors: Mapping[str, str]`. Unknown names
raise `ValueError`. `GICS_SECTORS` is the frozenset of the 11 sector names.

`scripts/refresh_sp500.py` (dev only, network) regenerates the CSV from a public
constituent list and maps sectors to GICS names. It is not imported by the package and is
excluded from coverage.

### Frozen snapshots

- `save_snapshot(snapshot, root) -> Path` writes `<as_of>_<sha12>.parquet` (the closes
  frame) and `<as_of>_<sha12>.json` (`as_of`, `columns` in order, `sectors` for those
  columns, `created_at`). `sha12` is the first 12 hex chars of SHA-256 over `as_of`, the
  column list, the sectors of those columns, and the frame's values and index
  (deterministic serialization). `created_at` is excluded from the hash.
- Files are content-addressed: the name is derived from the content, so if the target
  pair already exists the call is a no-op returning the existing path. Files are never
  overwritten. Both files are written atomically (temp file + `os.replace`).
- `load_snapshot(path) -> PriceSnapshot` reads the pair and constructs a
  `PriceSnapshot`, so its contract is revalidated.

### `build_snapshot(provider, tickers, as_of, lookback=300) -> SnapshotBuild`

1. Normalize `tickers`: strip, upper-case, de-duplicate preserving first occurrence.
   Empty after normalization → `ValueError`.
2. `closes = provider.closes(tickers, as_of − 450 calendar days, as_of)`; drop any rows
   after `as_of` (look-ahead guard); keep the last `lookback` rows.
3. `sectors = provider.sectors(tickers)`. Tickers whose sector is `None` or missing are
   dropped from the frame and recorded in `unresolved` with reason `"no_sector"`.
4. Tickers with no price data stay as all-NaN columns; the factor engine excludes them as
   `insufficient_history`. `build_snapshot` does not duplicate that rule.
5. If the frame has no rows or no columns left, raise
   `ValueError("no price data for the requested tickers at as_of")` before constructing
   the snapshot. Otherwise return
   `SnapshotBuild(snapshot=PriceSnapshot(as_of, closes, sectors_of_kept), unresolved)`.

`SnapshotBuild` is frozen: `snapshot: PriceSnapshot`, `unresolved: Mapping[str, str]`
(sorted by ticker).

## Error handling

| Condition | Behaviour |
|---|---|
| Vendor/transport failure | `DataProviderError` propagates |
| Ticker with no prices | All-NaN column (engine excludes it) |
| Ticker with no sector | Dropped, `unresolved[ticker] = "no_sector"` |
| Empty ticker list / nothing left | `ValueError` from `build_snapshot` |
| Vendor rows after `as_of` | Dropped before building |
| Request not fully covered by cache | Full union-range refetch, cache replaced |
| Ticker fetch returned no data | NaN column, not cached, retried next call |
| Unknown universe name | `ValueError` |

## Testing

Offline by default; vendor calls and providers are injected fakes; `tmp_path` for all
directories.

- `tests/fakes.py`: `FakeProvider(closes_by_ticker, sectors)` serving canned data and
  recording every `closes`/`sectors` call; `fake_download(...)` reproducing yfinance's
  multi-ticker, single-ticker and empty return shapes.
- `tests/adapters/data/test_yfinance_provider.py`: shape normalization; missing ticker →
  NaN column; `end + 1 day` request; timezone stripping; Yahoo→GICS mapping and unknown →
  `None`; vendor exception → `DataProviderError` with cause.
- `tests/adapters/data/test_caching.py`: cold fetch writes parquet and coverage; covered
  request makes no inner call; forward extension and backward extension each refetch the
  full union range and replace the cache; a restated series (all earlier prices scaled)
  replaces the old one with no seam; coverage end is capped at `clock() − 1 day` (fixed
  clock); an all-NaN fetch is not cached and is retried; `None` sectors are not cached;
  a new instance on the same directory reuses the cache; a parquet without its JSON is
  treated as uncached.
- `tests/adapters/data/test_static_sectors.py`: universe first, misses delegated,
  `closes` passes through.
- `tests/test_universe.py`: shipped `sp500.csv` has a list date, unique tickers, and only
  `GICS_SECTORS` values; unknown name raises.
- `tests/adapters/data/test_frozen.py`: save→load round trip (frame, sectors, `as_of`);
  repeat save is a no-op returning the same path; different content (prices or sectors)
  yields a different file; deterministic name.
- `tests/services/test_snapshots.py`: normalization and de-duplication; look-ahead rows
  dropped; `lookback` trim; missing sector → `unresolved` and column dropped; no-price
  ticker kept as NaN column; empty/all-dropped raises; end-to-end
  `build_snapshot` → `score()` on fake data yields a non-empty ranking.
- `tests/test_network_smoke.py` (`@pytest.mark.network`): three real tickers through
  `CachingProvider(YFinanceProvider())` → `build_snapshot` → `score()`. Excluded by
  default via `addopts = "-m 'not network'"`; the `network` marker is registered.
- Coverage: 100% of `ports/data.py`, `adapters/data/`, `universe.py` and `services/`.
  Tests pass in `dev`, `py311` and `py312`.

## Dependencies

- Add `pyarrow` (conda-forge) for parquet.
- `yfinance` is already a dependency.
- `pyproject.toml`: extend the mypy `ignore_missing_imports` override to `yfinance` and
  `pyarrow`; register the `network` pytest marker and add `addopts = "-m 'not network'"`.

## Open questions (deferred, not blocking)

- Survivorship-free point-in-time S&P 500 membership source.
- Whether to keep Yahoo symbols or move to a vendor-neutral ticker scheme when a second
  vendor arrives.
- Rate limiting and retries for large universes.
