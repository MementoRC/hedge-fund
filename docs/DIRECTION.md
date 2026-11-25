# Hedge Fund — Direction (2026-10-07)

## Sources
| Source | Contribution |
|---|---|
| TauricResearch/TradingAgents v0.6.0 (Apache-2.0, LangGraph) | Multi-agent single-ticker qualitative research: 4 analysts → bull/bear debate → trader → risk debate → portfolio manager. Claude-capable; memory log with settlement/reflection; `run_backtest()` grid. Forked to MementoRC/TradingAgents (branch `hedge-fund`). |
| Video w-8XSCgwzHw "MIT Researchers Built An AI Hedge Fund" (AI Pathways, 2026-09-24), 18 claims | TradingAgents in practice: ~8 min and $0.05–$1 per ticker-run; non-deterministic; best as supplementary long-horizon research; strongest model on the research/portfolio managers; historical-date runs used as a "backtest". |
| Video ANUXcTgrpg0 "Turn Claude Into Your Personal Hedge Fund" (AI Pathways, 2026-05-03), 27 claims | Quant-first long/short equity: 8-factor sector-neutral S&P 500 scoring, crowding check, Piotroski/Altman screens, mean-variance sizing, factor-vs-specific risk decomposition, correlation flags, beta/sector/factor/residual attribution, circuit breakers (2.5% daily loss, 8% drawdown), human-in-the-loop approvals, Alpaca execution. |

Claims are in the financial-analyst ledger. Most claims are methodology or promotion rather than checkable facts. Performance claims are unverified and short-window, and the creator promotes a paid community, so discount them.

## Gap analysis
TradingAgents answers "what do I think about NVDA today?" It lacks:
1. Universe selection. It runs one ticker at a time, and 500 names would be cost-prohibitive.
2. Portfolio construction. It has no cross-asset optimization or exposure and correlation limits.
3. Hard risk controls. Its "risk debate" is LLM talk.
4. Trustworthy evaluation. It is non-deterministic, and historical-date runs hit live vendor data (look-ahead and survivorship bias). It has no cost accounting.
5. Execution.

## Architecture: deterministic quant core, LLM research layer
0. Data layer: point-in-time cache (yfinance first, Polygon/Alpha Vantage later), FRED, SEC EDGAR.
1. Factor engine (deterministic): 8 sector-neutral factors, a composite score, a crowding check, and Piotroski/Altman quality gates.
2. Candidate shortlist: top/bottom N per sector, bounded by cost.
3. Research layer: forked TradingAgents, Claude-routed, with the strong model only on the research and portfolio managers. Run it K times per name. Consensus plus dispersion gives conviction.
4. Portfolio construction: mean-variance or risk-parity with sector, beta, gross/net and pairwise-ρ constraints.
5. Risk engine (hard, non-LLM, veto power): circuit breakers, factor/specific decomposition, exposure limits.
6. Human approval gate, then Alpaca paper execution.
7. Ledger and evaluation: decisions, theses, costs and settlements; attribution; thesis claims verified after the holding period, as financial-analyst does.
8. Backtest harness (offline): `hedge-fund backtest snapshot` freezes point-in-time prices, splits, dividends, facts and sectors into a store, online and once. `hedge-fund backtest run --store S --root R` replays the unchanged cycle over that store with a simulated paper broker and a neutral research stand-in: no network, no LLM, no broker credentials. Cycles run on the first trading day of each month, with no LLM budget, and the run's summary is written to `R/backtest/<run_id>.json`.

## Differentiators
- Ensemble agent runs, where dispersion becomes a conviction signal.
- A point-in-time evaluation harness: frozen snapshots, with no live data on historical dates.
- A cost governor: a per-cycle LLM budget sets N and K.
- Thesis verification feedback loop.

## Read-only views: demo world, dashboard, site

### Demo world
`pixi run -e dev demo` runs `hedge-fund demo build --root data/demo`. It writes a seeded synthetic frozen store to the sibling `data/demo.store`, runs the real backtest over it into `data/demo`, and writes the marker `data/demo/DEMO.json` last. Its numbers are synthetic, not a result.

### Dashboard
`pixi run -e dev app` starts a read-only Streamlit app (`app.py` only calls `hedge_fund.dashboard.shell.main`). Cycles, approvals and backtests stay in the CLI.

- **Pages:** Overview, Portfolio & NAV, Cycles, Research & attribution, Backtests.
- **Sidebar on every page:** the store root (default `data/demo`), what kind of root it is, and a warning that lists unreadable files (up to 10, then a count).
- **Data:** read only through `viz.load.load_workspace`; the dashboard imports no adapter. The state is cached per root and refreshed when a file under the root changes; a change only in a demo's sibling `<root>.store` is not seen.
- **Root states:**
  - Demo vs real is decided only by `viz.load.classify_root`: a demo root gets a DEMO badge, a backtest root the caption "Backtest root.".
  - `classify_root` refuses every ordinary live or paper root, and that refusal is never shown as an error. A root with snapshots, no marker, no backtest summary and nothing unreadable gets the neutral caption "Live root"; any other refusal gets "Not a demo or backtest root: <reason>". The pages show the root's data either way.
  - A root that cannot be read fails closed: the page shows an error and no data, and the sidebar a warning.
  - Overview and Portfolio & NAV need account snapshots. Cycles, Research & attribution and Backtests only need a readable root.
  - A missing root, and a page whose input the root does not hold, name `pixi run -e dev demo`; most also name the command that writes the input. Research's verification and attribution gaps are the exception, because the demo cannot fill them: on a demo or backtest root the page explains that these reports cannot exist there, and on any other root it names `hedge-fund verify` or `hedge-fund attribute`.
- **n/a:** the views pass NaN and infinities through. Every float, percentage and money value a page prints goes through `viz/format.py` (shared with the site), which shows "n/a" for a missing, NaN or infinite value. Whole-number counts (trading days, breaker days, funnel counts) are printed as they are.
- **Text:** where the dashboard renders text as markdown, it escapes the text that comes from stores or errors: thesis text, risk vetoes, residual biases and a cycle's error; the backtest run id in its heading and the ticker in a thesis label; and, in the sidebar and the page error, error messages and unreadable-file lines. That text shows as written. Selector options and table cells are plain text and are not escaped.
- **Charts:** built by `viz/charts.py` (no Streamlit) in one fixed dark palette, `viz.theme.DARK`, chosen in `dashboard/palette.py`. Streamlit's chart theme is turned off for them.

### Site
`website/` holds hand-written MkDocs Material pages and a generated "Results" section (Portfolio, Cycles, Backtest, Stats). `pixi run -e site site` first runs `hedge-fund site build --root data/demo`, which renders those pages from the demo root with the dashboard's views and charts into `website/docs/generated/`. It then builds everything into `_site/`. The builder accepts only demo and backtest roots, and it rewrites its output dir only when that dir is empty or carries its `.hf-site-generated` marker. Charts are drawn in the browser from JSON in both palettes, matching Material's light or dark scheme. `.github/workflows/pages.yml` builds the demo in CI, records a test run for the Stats page (`ci.yml` stays the blocking gate), builds PRs into `development` as a check, and deploys on every push to `development` and on manual dispatch.

## Out of scope (v1)
Live-money trading, intraday strategies, non-US markets.
