"""Backtests page: the run table, the selected run's full-window figures, breaker days and
residual biases, and equity vs SPY.

Run figures (``BacktestRun``) are stored full-window metrics; the SPY line on the equity
chart covers only the NAV dates it has (``benchmark_window`` / ``benchmark_partial``).
"""

from __future__ import annotations

from hedge_fund.dashboard.format import fmt_date, fmt_number, fmt_percent
from hedge_fund.dashboard.overview import Headline
from hedge_fund.viz.views import BacktestRun


def run_metrics(run: BacktestRun) -> tuple[Headline, ...]:
    window = f"Run window: {fmt_date(run.start)} to {fmt_date(run.end)}"
    return (
        Headline("Return (full window)", fmt_percent(run.total_return), window),
        Headline("SPY (full window)", fmt_percent(run.benchmark_return), window),
        Headline("Excess (full window)", fmt_percent(run.excess_return), window),
        Headline("Max drawdown (full window)", fmt_percent(run.max_drawdown), window),
        Headline("Sharpe (full window)", fmt_number(run.sharpe), window),
        Headline("Beta (full window)", fmt_number(run.beta), window),
        Headline("Breaker days", str(run.breaker_days), "Daily-loss plus drawdown breaker days"),
        Headline("Trading days", str(run.trading_days), window),
    )


def bias_lines(run: BacktestRun) -> str:
    """The run's residual biases as a markdown list ("" when it lists none)."""
    return "\n".join(f"- {bias}" for bias in run.biases)
