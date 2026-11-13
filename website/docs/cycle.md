# A cycle, step by step

A cycle is one paper-trading run, from reading the account to sending orders. It runs over injected interfaces: a data provider, a fundamentals provider, a researcher, a broker, an approval gate and stores. Whatever happens, the cycle tries to save exactly one report at the end. The stage names below are the ones the report records, so a failed cycle says where it stopped. The recorded stage can also be `inputs` (the checks before the first stage) or `budget` (the cost estimate in the research stage).

## Before the stages

The cycle first checks its inputs. The date must be today or the previous weekday; it is refused if it is the previous weekday but today's session has already opened, because the broker's equity is no longer that day's close. The number of research runs per name must be at least the minimum that makes a verdict possible. A budget, if given, must be valid.

## The stages, in order

| # | Stage | What happens |
|---|---|---|
| 1 | `account` | Read the account from the broker: equity and holdings. Unless this is a dry run, save the snapshot. |
| 2 | `nav` | Rebuild the net asset value history from the saved snapshots plus the one just read. |
| 3 | `preflight` | Check the circuit breakers on that history. If the day's loss is 2.5% or more, or the drawdown is 8% or more, the cycle ends here as `halted-preflight`. |
| 4 | `data` | Fetch prices for the universe, then fundamentals for the names that have prices. Names that cannot be priced or lack fundamentals are recorded as unresolved. |
| 5 | `shortlist` | Score the names on the factors and pick the long and short candidates. |
| 6 | `research` | Research the shortlist (below). |
| 7 | `portfolio` | Build target weights from the research, the prices and a benchmark. |
| 8 | `risk` | The risk engine assesses the target book and returns a verdict. |
| 9 | `prices` | Find a price for every target and held name. |
| 10 | `orders` | Turn the target, the verdict and the current holdings into an order plan. A rejected or halted verdict gives a blocked plan with no orders. |
| 11 | `submission_window` | Outside a dry run, refuse to continue inside the weekday session window, 09:25 to 16:15 New York time, unless explicitly overridden. Orders are meant to be sent before 09:25 or after 16:15. |
| 12 | `pending_orders` | For a plan that would send orders, refuse if orders from an earlier cycle are still open at the broker, so the book is not sized against holdings that are about to change. |
| 13 | `approval` | Offer the plan to the approval gate (below). |
| 14 | `execution` | If approved, check the window again and send the orders, reducing orders first. If any reducing order fails, the opening orders are skipped, and the opening leg of a flip is deferred. |

After the stages the report is saved. A dry run stops after `orders` and never reaches the approval; it also does not save the account snapshot.

## The research stage

If research for the same date is already stored, has no failed runs and fits the shortlist, it is reused unless fresh research is asked for. Otherwise, when a budget is set, a `budget` step first estimates the cost of a run from earlier cycles (or a fallback price) and cuts the runs per name first, down to the minimum that makes a verdict possible, then drops names from the end of the shortlist. A budget that fits no name ends the cycle as `nothing-to-do` with no portfolio, no orders and no broker write. Without a budget, the whole shortlist is researched. The result is stored, and the spend is recorded in the report.

## The approval stage

A blocked plan (the risk engine vetoed or halted) is recorded as `blocked` and an empty plan as `nothing-to-do`. Neither reaches the gate, so no answer from a person can override the veto. Otherwise the gate is asked once. Only an explicit yes approves; anything else is `rejected`.

## Outcomes

| Outcome | Meaning |
|---|---|
| `submitted` | The plan was approved and the orders were sent. |
| `nothing-to-do` | The budget fitted no name, or the plan had no orders. |
| `rejected` | The gate did not approve. |
| `blocked` | The risk engine's veto or halt blocked the plan. |
| `halted-preflight` | A circuit breaker had already tripped before planning. |
| `dry-run` | The plan was built and the cycle stopped before approval. |
| `error` | A stage raised an exception. The report names the failed stage. |
| `interrupted` | The cycle was interrupted. The report names the stage. |

If a stage raises, a partial report with the failed stage and the error is saved and the original exception is raised again. If a completed cycle's report cannot be saved, that is raised as its own error carrying the report.
