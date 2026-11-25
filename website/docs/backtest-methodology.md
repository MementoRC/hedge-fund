# Backtest methodology

The backtest asks one question: does the deterministic quant core, factors to shortlist to portfolio to risk engine to orders, have an edge after realistic costs? It is repeatable and offline.

## What it does not test

The LLM research layer is out of scope. It cannot be backtested honestly: its models have seen the historical period and its tools fetch live vendor data. In the replay, research is replaced by a neutral stand-in that rates every long a buy and every short a sell with equal conviction, so every name is confirmed and the research layer neither adds nor removes anything. A good result therefore says something about the factors, the portfolio construction and the risk engine, and nothing about the LLM layer.

Also out of scope in v1: settlement, attribution and thesis verification (they report on real research and real fills), same-day liquidation when a breaker trips, resuming an interrupted run, paid data for delisted names, and size-based slippage.

## How the replay works

The replay runs the same cycle code as live trading, with simulated parts injected: a frozen data provider, a point-in-time universe, the neutral researcher, an approval gate that always says yes, and a simulated broker. For each trading day in the window:

1. The simulated broker closes out any held name that has stopped trading, applies splits and dividends dated that day, then fills pending orders at the day's open, with slippage.
2. At the close it accrues short borrow cost, marks the book to market and saves an account snapshot. A snapshot is saved every trading day, so the circuit breakers see daily moves.
3. The breakers are evaluated on those snapshots: the report counts the days they would have tripped, and, as in live trading, a tripped breaker halts planning when the next cycle runs.
4. On the first trading day of each calendar month, after the close, a cycle runs with data up to that close. Its orders fill at the next trading day's open, as they would live. A window that starts mid-month first rebalances the following month.

When the window ends, the harness computes the metrics and writes a summary.

## Costs

- Fills at the next session's open, with 5 basis points of adverse slippage.
- Short borrow at 0.5% a year on the short's market value, accrued at each close over the calendar days since the previous close.
- Zero commission.
- Slippage and the borrow rate are configurable.

## Point-in-time rules

- **Prices** are seen as of the modelled date. Every split dated after that date is undone. Dividends adjust prices only where the ex-date is on or before that date.
- **Fundamentals** count only when filed before the modelled date, and the latest filing for each period and concept wins, so a restatement replaces the original only after its own filing date.
- **Membership** comes from a dated S&P 500 constituents file. A security is identified by ticker and membership span, not by ticker alone. A packaged file flags the spans where a reused ticker belongs to another company, and those spans are excluded and counted.
- **Universe** on each date: the members that day, minus flagged spans. Members that are unpriced or have no sector are then dropped from that cycle's data. Every drop is counted in the report.
- The broker trades on raw, split-unwound prices.
- The window may not start before 2008-01-01, because membership is treated as reliable only from then. The default start is 2012-01-01.

## The data is frozen

A one-off online step builds a content-addressed store: prices from yfinance with split and dividend events, company facts from SEC EDGAR with every filing's row, and sectors (a current static list, with the SEC industry code as a fallback). The store carries a manifest of digests. The replay is offline and refuses to run if the store does not match its manifest.

## Failing closed

The replay counts, reports and continues when a member is unpriced, has no sector or no fundamentals, when a held name stops trading, when a cycle ends blocked or halted, and when the open is missing on a fill day (the order is rejected, never filled at the close). It aborts with a clear message on a digest mismatch, a non-empty output directory, a broker that is not the simulated one, a clock that does not move forward, any exception from the cycle, a failed cycle, a window outside the store's coverage or with fewer than three trading days, a day without a SPY close, or a precomputed shortlist that disagrees with the cycle's own. Its wiring builds only the simulated broker, so it has no route to a real one.

## Residual biases

These remain after the rules above, and every report prints them:

- Sectors are current, not historical.
- yfinance coverage of delisted names is incomplete (see the unpriced count).
- Membership accuracy depends on its source dataset.
- Late removals: dataset dates can trail the effective date, so a removed stock may be held a few extra days (late additions never look ahead).
- Ticker reuse not listed in the flagged-spans file may remain: a reused symbol can return another company's prices, CIK and facts.
- With monthly rebalances the breakers act only on the one-day move at cycle time; the would-trip breaker-day counts are the main evidence of breaker behaviour.

## Honesty counts

Next to the performance metrics (return against SPY, volatility, Sharpe ratio, maximum drawdown, turnover, trading costs and beta), the report gives counts of the unpriced members by reason (one reason is having no sector), members with no fundamentals, distinct flagged tickers excluded from a rebalance-day universe, held names that were closed out because they stopped trading, rejected orders, would-trip breaker days, and cycle outcomes such as blocked and halted. They show how much of the universe and how many cycles the result actually rests on.

In the dashboard charts, the SPY benchmark may cover only part of the NAV window. It is then drawn over the dates it covers. The backtest report's benchmark and excess returns are full-window figures. The benchmark is left out when it does not overlap the NAV dates, or when any covered close is not positive and finite.
