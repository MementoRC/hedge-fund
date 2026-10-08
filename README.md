# hedge-fund

Hybrid hedge fund: a deterministic quant core plus an LLM research layer built on a fork of
[TradingAgents](https://github.com/MementoRC/TradingAgents). Paper trading only.
See `docs/DIRECTION.md`.

## Setup

The `dev` environment installs TradingAgents editable from a sibling clone, which must exist at
`../TradingAgents` (branch `hedge-fund` of `MementoRC/TradingAgents`):

```bash
git clone https://github.com/MementoRC/TradingAgents ../TradingAgents
git -C ../TradingAgents checkout hedge-fund
pixi install -e dev
pixi run -e dev setup-dev
```

## Commands

```bash
pixi run -e dev test
pixi run -e dev quality
pixi run -e dev app
```

All dependencies come from conda-forge (no pypi-dependencies). `setup-dev` installs this package and `../TradingAgents` editable with `python -m pip --no-deps`, plus `stockstats` (not yet on conda-forge). `langgraph-checkpoint-sqlite` is likewise not on conda-forge; the fork imports it lazily, so checkpointing is unavailable until it is packaged.
