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
    in ``kind_error``.
    """

    root: Path
    workspace: Workspace | None
    kind: RootKind | None
    kind_error: str | None


def store_fingerprint(root: Path) -> Fingerprint:
    """``(relative name, mtime_ns, size)`` of every regular file under ``root``, sorted.

    The sibling ``<root>.store`` is not included.
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


def inspect_root(root: Path) -> RootState:
    if not root.is_dir():
        return RootState(root, None, None, None)
    workspace = load_workspace(root)
    try:
        kind = classify_root(root)
    except RootError as exc:
        return RootState(root, workspace, None, str(exc))
    return RootState(root, workspace, kind, None)


@st.cache_resource(show_spinner=False, max_entries=8)
def _cached(root: str, fingerprint: Fingerprint) -> RootState:
    del fingerprint  # part of the cache key only: a changed file means a new key
    return inspect_root(Path(root))


def load_state(text: str) -> RootState:
    """The state of the root typed in the sidebar (blank means the default), cached per root
    and invalidated when any file under it changes (not the sibling ``.store``).
    """
    root = Path(text.strip() or DEFAULT_ROOT)
    return _cached(str(root), store_fingerprint(root))
