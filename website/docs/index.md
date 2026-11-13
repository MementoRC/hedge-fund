# hedge-fund

hedge-fund is a hybrid long/short equity system with two halves: a deterministic quantitative core and an LLM research layer. It trades paper money only. The code has no live-money execution, and v1 does not plan one.

## The two halves

**The quant core** is ordinary deterministic code: the same inputs give the same outputs. It scores the stock universe on sector-neutral factors, picks a shortlist of long and short candidates, builds a portfolio under exposure limits, checks that portfolio against a risk engine that can veto it, and turns the result into orders.

**The LLM research layer** is a fork of TradingAgents, a multi-agent system that researches one ticker at a time. It is not deterministic and every run costs money, so it only sees the shortlist. Each shortlisted name is researched several times. The agreement and the spread between the runs become a conviction signal. A per-cycle budget limits how many names, and how many runs per name, are researched.

## Why two halves

TradingAgents answers "what do I think about this stock today?" On its own it does not choose a universe, build a portfolio, enforce hard risk limits, evaluate itself without look-ahead, or place orders. The quant core supplies those. The LLM layer adds qualitative research on the names the core selects.

The risk engine has the last word. It is deterministic code, not a model, and its veto applies to every plan, including one shaped by the LLM layer. In a live cycle a human also approves the plan before any order is sent.

## Out of scope in v1

- Live-money trading.
- Intraday strategies.
- Non-US markets.
