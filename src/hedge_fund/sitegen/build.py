"""Write the site's generated result pages (the only sitegen module that does I/O).

Reads a root only through ``viz.load``; refuses every root that ``classify_root`` refuses
(live and paper roots included); reads ``junit.xml`` with ``defusedxml`` (untrusted XML, no
DTD or entities) and ``coverage.json`` from ``--stats-dir``. A missing or malformed stats
file makes the Stats page say so; it never fails the build.
"""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

from defusedxml.ElementTree import ParseError, parse
from plotly.offline import get_plotlyjs

from hedge_fund.sitegen.assets import (
    ASSETS_DIR,
    CHARTS_JS,
    PALETTE_CSS,
    PLOTLY_JS,
    charts_js,
    palette_css,
)
from hedge_fund.sitegen.pages import (
    BACKTEST_PAGE,
    CYCLES_PAGE,
    PORTFOLIO_PAGE,
    STATS_PAGE,
    JunitCounts,
    PageContext,
    Stats,
    backtest_page,
    cycles_page,
    portfolio_page,
    stats_page,
)
from hedge_fund.viz.load import RootError, RootKind, Workspace, classify_root, load_workspace
from hedge_fund.viz.views import (
    backtest_view,
    cycle_view,
    overview_view,
    portfolio_view,
    research_view,
)

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
    symlinks and ``..`` cannot bypass the checks. Callers must delete the resolved path that
    this function validated, not the raw ``out`` argument.
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
    if not target.is_dir():
        return
    try:
        non_empty = any(target.iterdir())
    except OSError as exc:
        raise SiteError(f"cannot read --out {out}: {exc.strerror or exc}") from exc
    marker = target / MARKER
    if non_empty and not (marker.is_file() and not marker.is_symlink()):
        raise SiteError(f"--out {out} is not empty and has no {MARKER}: refusing to delete it")


def build_site(root: Path, out: Path, stats_dir: Path) -> None:
    """Build the site for ``root`` into ``out``.

    Classify ``root`` (demo or backtest only), check ``out``, then rewrite it: the marker,
    the four pages, ``palette.css``, ``charts.js`` and ``plotly.min.js``.

    Every refusal is a SiteError raised before anything is deleted.
    """
    try:
        kind = classify_root(root)
    except RootError as exc:
        raise SiteError(str(exc)) from exc
    check_out(out, root)
    target = out.resolve()  # the path check_out validated; never delete the raw ``out``
    pages = render_pages(load_workspace(root), kind, read_stats(stats_dir))
    plotly_js = get_plotlyjs()
    if target.exists():
        shutil.rmtree(target)
    assets = target / ASSETS_DIR
    assets.mkdir(parents=True)
    _write(target / MARKER, "")
    for name, text in pages.items():
        _write(target / name, text)
    _write(assets / PALETTE_CSS, palette_css())
    _write(assets / CHARTS_JS, charts_js())
    _write(assets / PLOTLY_JS, plotly_js)


def render_pages(workspace: Workspace, kind: RootKind, stats: Stats) -> dict[str, str]:
    """Page file name to Markdown, from the existing view builders."""
    ws = workspace
    context = PageContext(demo=kind is RootKind.DEMO, unreadable=ws.unreadable)
    costs = research_view(ws.research, ws.cycles, ws.verifications, ws.attributions).costs
    return {
        PORTFOLIO_PAGE: portfolio_page(
            overview_view(ws.snapshots, ws.benchmark, ws.cycles),
            portfolio_view(ws.snapshots, ws.benchmark, ws.cycles),
            context,
        ),
        CYCLES_PAGE: cycles_page(cycle_view(ws.cycles), costs, context),
        BACKTEST_PAGE: backtest_page(
            backtest_view(ws.backtests, ws.snapshots, ws.benchmark), context
        ),
        STATS_PAGE: stats_page(stats, context),
    }


def _write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8", newline="\n")


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
