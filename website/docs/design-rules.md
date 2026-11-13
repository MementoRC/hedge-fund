# Design rules

These are the constraints the code is built to keep. Several are enforced by tests rather than by convention.

## The engine is pure

The domain types and the engine do no I/O, no network calls and no LLM calls. Inputs and outputs are plain data. That is what makes the quant core deterministic and testable without mocks. Everything that touches the outside world sits behind an interface (a port) and is implemented in an adapter. Services run the engine over injected ports. The command-line module is the composition root: it builds the brokers and wires the adapters, and it is the only place that builds a broker. The one other module allowed to import adapters is the read-only loader that the dashboard and site use to read saved data. An architecture test reads the source tree and fails on a forbidden import.

## The risk engine has a veto

The risk engine is deterministic code and it is not advisory. It runs on every target portfolio, including one derived from LLM research, and returns one verdict.

It rejects the book if any of these breach their limits (defaults shown, as fractions):

- gross exposure (sum of absolute weights) above 2.0, and net exposure beyond 10%;
- any single name above 10%;
- the book's beta beyond a 10% band around zero;
- any sector's net weight beyond 10% or gross weight above 50%;
- ex-ante annual volatility above 10%;
- any one name contributing more than 15% of total variance, or factor risk above 50% of total variance;
- a name the risk model cannot assess.

Two circuit breakers also act: a one-day loss of 2.5% or more, or a drawdown of 8% or more, trips a halt, and a halt always rejects. All exposure violations, and either the risk-limit violations or the unassessable names, are listed in the verdict. The engine never repairs a book; it only approves or rejects it.

A rejected plan is blocked before it reaches the approval gate. The gate is never asked, so a human answer cannot override the veto.

## Paper trading only

The system executes against a paper broker. There is no live-money path, and none is planned for v1. The backtest wiring, built in the same command-line module, builds only a simulated broker.

## Point-in-time honesty

An evaluation is only worth something if the model cannot see the future. So:

- prices are taken as they were seen on the modelled date: later splits are undone and dividends are adjusted only up to that date;
- fundamentals count only when they were filed before the modelled date, and a restatement replaces the original only after its own filing date;
- universe membership comes from a dated file, per date;
- backtests run on a frozen, content-hashed store, never on live vendor data;
- when something cannot be made point-in-time, the report lists it as a residual bias rather than hiding it. See [Backtest methodology](backtest-methodology.md).

The LLM layer cannot meet these rules: its models have seen the past, and its tools fetch live data. For that reason it is left out of the backtest.

## Test discipline

- Tests come first: a failing test, then the smallest change that passes it.
- The default suite runs offline; the few tests that call real vendor APIs are marked `network` and excluded unless asked for.
- Coverage is enforced at 100% of the package; the test command fails below it.
- Lint, formatting and type checks must pass before a commit.
- The backtest has a golden test: a small frozen store replayed end to end, with pinned results, so any change in behaviour shows up as a diff.
- Tooling is pixi only, and third-party dependencies come from conda-forge.
