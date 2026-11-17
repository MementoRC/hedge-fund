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
MARKER = ".hf-site-generated"


class SiteError(ValueError):
    """A root or an --out the builder refuses; the message is one line."""


def check_out(out: Path, root: Path) -> None:
    """Raise SiteError unless ``out`` is safe to delete and rewrite.

    Refused: the filesystem root, the home dir, any ``out`` that is or contains ``root`` or
    the current directory (``.`` included), a non-directory, and a non-empty directory
    without ``MARKER``. An absent or empty ``out`` is fine. Paths are resolved first, so
    symlinks and ``..`` cannot bypass the checks.
    """
    target = out.resolve()
    if target == Path(target.anchor):
        raise SiteError(f"--out {out} is the filesystem root: refusing to write there")
    if target == Path.home().resolve():
        raise SiteError(f"--out {out} is the home directory: refusing to write there")
    guarded = ((root.resolve(), "--root"), (Path.cwd().resolve(), "the current directory"))
    for inner, name in guarded:
        if target == inner or target in inner.parents:
            raise SiteError(f"--out {out} is or contains {name}: refusing to delete it")
    if target.exists() and not target.is_dir():
        raise SiteError(f"--out {out} is not a directory")
    if target.is_dir() and any(target.iterdir()) and not (target / MARKER).is_file():
        raise SiteError(f"--out {out} is not empty and has no {MARKER}: refusing to delete it")


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
