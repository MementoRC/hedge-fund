"""sitegen.build: stats files, the --out safety rules, and the full build of a root."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hedge_fund.sitegen.build import read_stats
from hedge_fund.sitegen.pages import JunitCounts, Stats

JUNIT = (
    '<?xml version="1.0" encoding="utf-8"?><testsuites><testsuite name="pytest" errors="0" '
    'failures="1" skipped="2" tests="12" time="1.0"/></testsuites>'
)
BARE_JUNIT = '<testsuite name="pytest" errors="1" failures="0" skipped="0" tests="3"/>'
ENTITY_JUNIT = (
    '<?xml version="1.0"?>\n<!DOCTYPE testsuites [<!ENTITY boom "x">]>\n'
    '<testsuites><testsuite errors="0" failures="0" skipped="0" tests="1"/></testsuites>'
)
DTD_JUNIT = (
    '<?xml version="1.0"?>\n<!DOCTYPE testsuites>\n'
    '<testsuites><testsuite errors="0" failures="0" skipped="0" tests="1"/></testsuites>'
)


def _stats_dir(tmp_path: Path, junit: str | None = None, coverage: object = None) -> Path:
    stats = tmp_path / "stats"
    stats.mkdir(parents=True, exist_ok=True)
    if junit is not None:
        (stats / "junit.xml").write_text(junit, encoding="utf-8")
    if coverage is not None:
        text = coverage if isinstance(coverage, str) else json.dumps(coverage)
        (stats / "coverage.json").write_text(text, encoding="utf-8")
    return stats


def _suite(tests: int, failures: int, errors: int, skipped: int) -> str:
    return (
        f'<testsuites><testsuite errors="{errors}" failures="{failures}" '
        f'skipped="{skipped}" tests="{tests}"/></testsuites>'
    )


def test_read_stats_reads_the_testsuite_under_testsuites_and_the_coverage(tmp_path: Path) -> None:
    stats = _stats_dir(tmp_path, JUNIT, {"totals": {"percent_covered": 99.996}})

    assert read_stats(stats) == Stats(JunitCounts(12, 1, 0, 2), 99.996)


def test_read_stats_accepts_a_testsuite_root(tmp_path: Path) -> None:
    assert read_stats(_stats_dir(tmp_path, BARE_JUNIT)).junit == JunitCounts(3, 0, 1, 0)


def test_missing_files_are_not_produced(tmp_path: Path) -> None:
    assert read_stats(_stats_dir(tmp_path)) == Stats(None, None)


@pytest.mark.parametrize(
    "junit",
    [
        "<testsuites",
        "<testsuites><other/></testsuites>",
        '<testsuites><testsuite tests="1"/></testsuites>',
        '<testsuites><testsuite errors="x" failures="0" skipped="0" tests="1"/></testsuites>',
    ],
)
def test_a_malformed_junit_is_not_produced(tmp_path: Path, junit: str) -> None:
    assert read_stats(_stats_dir(tmp_path, junit)).junit is None


@pytest.mark.parametrize("junit", [ENTITY_JUNIT, DTD_JUNIT])
def test_a_junit_with_a_dtd_or_an_entity_is_rejected(tmp_path: Path, junit: str) -> None:
    assert read_stats(_stats_dir(tmp_path, junit)).junit is None


@pytest.mark.parametrize(
    "junit",
    [
        _suite(-1, 0, 0, 0),
        _suite(5, -1, 0, 0),
        _suite(5, 0, -1, 0),
        _suite(5, 0, 0, -1),
        _suite(5, 3, 2, 1),
        _suite(0, 1, 0, 0),
    ],
)
def test_negative_or_inconsistent_junit_counts_are_not_produced(tmp_path: Path, junit: str) -> None:
    assert read_stats(_stats_dir(tmp_path, junit)).junit is None


def test_counts_that_add_up_exactly_are_produced(tmp_path: Path) -> None:
    junit = read_stats(_stats_dir(tmp_path, _suite(6, 3, 2, 1))).junit

    assert junit == JunitCounts(6, 3, 2, 1)


@pytest.mark.parametrize(
    "coverage",
    [
        "{",
        {"totals": {}},
        {"other": 1},
        ["x"],
        {"totals": {"percent_covered": "x"}},
        {"totals": {"percent_covered": None}},
        {"totals": {"percent_covered": True}},
        '{"totals": {"percent_covered": NaN}}',
        '{"totals": {"percent_covered": Infinity}}',
        '{"totals": {"percent_covered": -Infinity}}',
        '{"totals": {"percent_covered": 1e999}}',
    ],
)
def test_a_malformed_coverage_is_not_produced(tmp_path: Path, coverage: object) -> None:
    assert read_stats(_stats_dir(tmp_path, coverage=coverage)).coverage is None


def test_an_integer_coverage_is_a_float(tmp_path: Path) -> None:
    stats = _stats_dir(tmp_path, coverage={"totals": {"percent_covered": 100}})

    assert read_stats(stats).coverage == 100.0
