# hedge-fund — Project Conventions

- **TDD**: write the failing test first, then the minimal implementation.
- **`engine/` is pure**: no I/O, no network, no LLM calls. Inputs and outputs are plain data.
- **Risk engine is deterministic and has veto power** over any decision, including LLM-derived ones.
- **Paper trading only.** No live-money execution.
- **Layout**: `src/hedge_fund/` with ports (protocols) in `ports/` and implementations in `adapters/`.
- **Branch flow**: never commit directly to `development` or `main`. Do work on a feature branch off `origin/development` (e.g. `feat/<topic>`) and PR it into `development`. Promotion from `development` to `main` is a separate PR.
- **Specs and plans**: `docs/superpowers/specs/YYYY-MM-DD-*.md` and `docs/superpowers/plans/YYYY-MM-DD-*.md`.
- **Tooling**: pixi only. `pixi run -e dev quality` and `pixi run -e dev test` must pass before commit.
- **TradingAgents**: sibling clone at `../TradingAgents` (fork branch `hedge-fund`); do not edit it from this repo.
- See `docs/DIRECTION.md` for architecture and scope.
