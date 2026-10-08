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

## Differentiators
- Ensemble agent runs, where dispersion becomes a conviction signal.
- A point-in-time evaluation harness: frozen snapshots, with no live data on historical dates.
- A cost governor: a per-cycle LLM budget sets N and K.
- Thesis verification feedback loop.

## Out of scope (v1)
Live-money trading, intraday strategies, non-US markets.
