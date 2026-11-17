"""Write the site's generated result pages (the only sitegen module that does I/O).

Reads a root only through ``viz.load``; refuses every root that ``classify_root`` refuses
(live and paper roots included); reads ``junit.xml`` with ``defusedxml`` (untrusted XML, no
DTD or entities) and ``coverage.json`` from ``--stats-dir``. A missing or malformed stats
file makes the Stats page say so; it never fails the build.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from defusedxml.ElementTree import ParseError, parse

from hedge_fund.sitegen.pages import JunitCounts, Stats

JUNIT_FILE = "junit.xml"
COVERAGE_FILE = "coverage.json"
JUNIT_FIELDS = ("tests", "failures", "errors", "skipped")


def read_stats(stats_dir: Path) -> Stats:
    return Stats(
        junit=_junit(stats_dir / JUNIT_FILE), coverage=_coverage(stats_dir / COVERAGE_FILE)
    )


def _junit(path: Path) -> JunitCounts | None:
    """The ``testsuite`` under pytest's ``<testsuites>`` wrapper (or the root itself).

    Negative counts, or more failures, errors and skips than tests, are malformed.
    """
    try:
        top = parse(path, forbid_dtd=True).getroot()
        suite = top if top.tag == "testsuite" else top.find("testsuite")
        if suite is None:
            return None
        counts = JunitCounts(*(int(suite.attrib[name]) for name in JUNIT_FIELDS))
    except (OSError, ParseError, ValueError, KeyError):  # DefusedXmlException is a ValueError
        return None
    consistent = (
        min(counts.tests, counts.failures, counts.errors, counts.skipped) >= 0
        and counts.passed >= 0
    )
    return counts if consistent else None


def _coverage(path: Path) -> float | None:
    """``totals.percent_covered``; None unless it is a finite number."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))["totals"]["percent_covered"]
    except (OSError, ValueError, KeyError, TypeError):
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value) if math.isfinite(value) else None
