"""Stats page: the test result and the coverage percentage, or that they were not produced."""

from __future__ import annotations

from hedge_fund.sitegen.pages import (
    DEMO_BANNER,
    NO_COVERAGE,
    NO_TESTS,
    JunitCounts,
    Stats,
    stats_page,
)
from tests.sitegen.helpers import DEMO, PLAIN


def test_no_failure_and_no_error_is_passed() -> None:
    page = stats_page(Stats(JunitCounts(10, 0, 0, 2), 100.0), PLAIN)

    assert "Test suite: **passed** (8 passed, 0 failed, 0 errors, 2 skipped)." in page
    assert "Coverage: 100.00%." in page
    assert "below the 100% gate" not in page


def test_a_failure_or_an_error_is_failed() -> None:
    assert "**failed** (9 passed, 1 failed" in stats_page(
        Stats(JunitCounts(10, 1, 0, 0), None), PLAIN
    )
    assert "**failed** (9 passed, 0 failed, 1 errors" in stats_page(
        Stats(JunitCounts(10, 0, 1, 0), None), PLAIN
    )


def test_coverage_under_100_is_flagged_even_when_it_rounds_to_100() -> None:
    assert "Coverage: 100.00%, below the 100% gate." in stats_page(Stats(None, 99.996), PLAIN)
    assert "Coverage: 87.50%, below the 100% gate." in stats_page(Stats(None, 87.5), PLAIN)


def test_missing_stats_say_they_were_not_produced() -> None:
    page = stats_page(Stats(None, None), PLAIN)

    assert NO_TESTS == "Test results were not produced in this build."
    assert NO_COVERAGE == "Coverage was not produced in this build."
    assert NO_TESTS in page
    assert NO_COVERAGE in page


def test_exactly_one_missing_input_names_only_that_input() -> None:
    no_coverage = stats_page(Stats(JunitCounts(10, 0, 0, 0), None), PLAIN)
    no_tests = stats_page(Stats(None, 100.0), PLAIN)

    assert NO_COVERAGE in no_coverage
    assert NO_TESTS not in no_coverage
    assert NO_TESTS in no_tests
    assert NO_COVERAGE not in no_tests


def test_the_stats_page_has_the_demo_banner_on_a_demo_root() -> None:
    assert DEMO_BANNER in stats_page(Stats(None, None), DEMO)
    assert stats_page(Stats(None, None), PLAIN).startswith("# Stats\n")
