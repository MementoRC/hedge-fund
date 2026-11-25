"""Backtest page: the root's one run, its full-window figures, biases, equity vs SPY."""

from __future__ import annotations

from datetime import date

import pytest

from hedge_fund.dashboard import backtests as dashboard_backtests
from hedge_fund.sitegen.pages import (
    EQUITY_NOTE,
    NO_BIAS,
    NO_EQUITY,
    SITE_NO_SPY,
    backtest_page,
    cell,
)
from hedge_fund.viz.views import backtest_view
from tests.dashboard.builders import NAV_VALUES, make_backtest_report
from tests.sitegen.helpers import PLAIN, chart_ids, figures
from tests.viz.builders import make_nav, make_snapshots

SNAPSHOTS = make_snapshots(*NAV_VALUES)


def _page(*biases: str, run_id: str = "run-1", spy: bool = False, snapshots: bool = True) -> str:
    report = make_backtest_report(run_id=run_id, biases=biases)
    benchmark = make_nav(50.0, 51.0, 52.0, 53.0) if spy else None
    view = backtest_view((report,), SNAPSHOTS if snapshots else (), benchmark)
    return backtest_page(view, PLAIN)


def test_the_page_names_the_one_run_and_its_full_window_figures() -> None:
    page = _page()

    assert page.startswith("# Backtest\n")
    assert r"This root holds one backtest run, run\-1." in page
    for label in (
        "Return (full window)",
        "SPY (full window)",
        "Excess (full window)",
        "Max drawdown (full window)",
        "Sharpe (full window)",
        "Beta (full window)",
        "Breaker days",
        "Trading days",
    ):
        assert f"| {cell(label)} |" in page


def test_biases_are_an_escaped_list() -> None:
    page = _page("<script>alert(1)</script>", "*x* & y", "two\n\nlines")

    assert r"- &lt;script&gt;alert\(1\)&lt;/script&gt;" in page
    assert r"- \*x\* &amp; y" in page
    assert "- two lines" in page
    assert NO_BIAS not in page


def test_no_bias_gives_the_note() -> None:
    assert NO_BIAS == "The run lists no residual bias."
    assert NO_BIAS in _page()


def test_a_run_id_with_whitespace_stays_inline() -> None:
    assert "run one" in _page(run_id="run\r\none")


def test_equity_vs_spy_without_spy_says_so() -> None:
    page = _page()

    assert chart_ids(page) == ["backtest-equity"]
    assert SITE_NO_SPY in page
    assert EQUITY_NOTE in page
    assert [trace["name"] for trace in figures(page)["backtest-equity-light"]["data"]] == [
        "Strategy"
    ]


def test_equity_vs_spy_with_spy_draws_both_lines() -> None:
    page = _page(spy=True)

    assert SITE_NO_SPY not in page
    assert "SPY covers the whole NAV history, 2024-01-02 to 2024-01-05." in page
    assert EQUITY_NOTE in page
    names = [trace["name"] for trace in figures(page)["backtest-equity-dark"]["data"]]
    assert names == ["Strategy", "SPY"]


def test_a_partial_benchmark_gets_the_partial_window_note() -> None:
    report = make_backtest_report()
    benchmark = make_nav(51.0, 52.0, 53.0, start=date(2024, 1, 3))
    view = backtest_view((report,), SNAPSHOTS, benchmark)
    assert view.benchmark_partial

    page = backtest_page(view, PLAIN)

    assert (
        "SPY covers only 2024-01-03 to 2024-01-05 (partial window): the chart rebases it to "
        "the strategy at 2024-01-03 while the strategy line stays full length."
    ) in page
    assert SITE_NO_SPY not in page
    assert EQUITY_NOTE in page


def test_the_copied_wording_matches_the_dashboard() -> None:
    assert NO_BIAS == dashboard_backtests.NO_BIAS
    assert EQUITY_NOTE == dashboard_backtests.EQUITY_NOTE


@pytest.mark.parametrize("count", [0, 2])
def test_a_root_without_exactly_one_run_is_rejected(count: int) -> None:
    reports = tuple(make_backtest_report(run_id=f"run-{i}") for i in range(count))
    view = backtest_view(reports, SNAPSHOTS, None)

    with pytest.raises(ValueError, match=f"exactly one run, got {count}"):
        backtest_page(view, PLAIN)


def test_no_snapshots_gives_only_the_note_with_no_chart_or_captions() -> None:
    page = _page(snapshots=False)

    assert NO_EQUITY in page
    assert chart_ids(page) == []
    assert SITE_NO_SPY not in page
    assert EQUITY_NOTE not in page
    assert page.endswith(f"## Equity vs SPY\n\n{NO_EQUITY}\n")
