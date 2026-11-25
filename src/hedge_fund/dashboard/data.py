"""What a page knows about the chosen root: its workspace, its kind, and a change-aware cache.

Stores are read only through ``viz.load``. ``kind`` comes from ``classify_root`` alone; the
workspace's own ``demo`` field is never used to decide demo vs real.

Cache limitation: the fingerprint covers only the files under the root itself. A demo root's
benchmark lives in the sibling ``<root>.store`` directory, which is not fingerprinted, so a
change there alone does not invalidate the cached state.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from stat import S_ISREG

import streamlit as st

from hedge_fund.viz.load import RootError, RootKind, Workspace, classify_root, load_workspace

DEFAULT_ROOT = "data/demo"
Fingerprint = tuple[tuple[str, int, int], ...]


@dataclass(frozen=True)
class RootState:
    """``workspace`` is None when ``root`` is not a directory. ``kind`` is None for a root that
    ``classify_root`` refuses (a live root, an empty directory, a bad marker), with the reason
    in ``kind_error``. ``read_error`` is set when ``classify_root`` could not read the root at all
    (an OSError, such as a permission failure); the kind is then None and ``kind_error`` is None.
    """

    root: Path
    workspace: Workspace | None
    kind: RootKind | None
    kind_error: str | None
    read_error: str | None = None
    is_live: bool = False


def store_fingerprint(root: Path) -> Fingerprint:
    """``(relative name, mtime_ns, size)`` of every regular file under ``root``, sorted.

    The sibling ``<root>.store`` is not included. This walks the whole root on every rerun, and
    ``rglob`` silently skips subdirectories it cannot read.
    """
    if not root.is_dir():
        return ()
    entries: list[tuple[str, int, int]] = []
    for path in sorted(root.rglob("*")):
        try:
            info = path.stat()
        except OSError:  # a dangling link, or a file that vanished mid-walk
            continue
        if S_ISREG(info.st_mode):
            entries.append((path.relative_to(root).as_posix(), info.st_mtime_ns, info.st_size))
    return tuple(entries)


def _is_live(workspace: Workspace) -> bool:
    """A plain live/paper root: it holds snapshots, no valid marker, no backtest summary, and
    nothing unreadable (an unreadable marker or summary would also be an unreadable entry), so
    the refusal is just "this is not a backtest root". Any other refusal stays an oddity worth
    showing, an empty directory included.
    """
    # workspace.demo only labels the caption here; it never drives the DEMO badge (kind does).
    return (
        bool(workspace.snapshots)
        and workspace.demo is None
        and not workspace.backtests
        and not workspace.unreadable
    )


def inspect_root(root: Path) -> RootState:
    if not root.is_dir():
        return RootState(root, None, None, None)
    workspace = load_workspace(root)
    try:
        kind = classify_root(root)
    except RootError as exc:
        return RootState(root, workspace, None, str(exc), is_live=_is_live(workspace))
    except OSError as exc:  # classify_root maps only ValueError; I/O errors escape it
        return RootState(root, workspace, None, None, read_error=str(exc))
    return RootState(root, workspace, kind, None)


@st.cache_resource(show_spinner=False, max_entries=8)
def _cached(root: str, fingerprint: Fingerprint) -> RootState:
    del fingerprint  # part of the cache key only: a changed file means a new key
    return inspect_root(Path(root))


def load_state(text: str) -> RootState:
    """The state of the root typed in the sidebar (blank means the default), cached per root
    and invalidated when any file under it changes (not the sibling ``.store``).

    The value is shared across reruns and sessions: callers must not mutate it.
    """
    root = Path(text.strip() or DEFAULT_ROOT)
    try:
        root = root.expanduser()
    except RuntimeError:  # "~user" with no resolvable home directory
        return RootState(root, None, None, None)
    return _cached(str(root), store_fingerprint(root))
