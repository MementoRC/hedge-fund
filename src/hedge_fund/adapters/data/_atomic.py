"""Atomic, durable file writes: write and fsync a temp file in the target directory, then publish.

``atomic_write`` publishes with ``os.replace`` (overwrites); ``atomic_create`` publishes with
``os.link`` (refuses an existing name, atomically). Both fsync the temp file before publishing
and the directory after, so a power loss cannot leave a durable rename of an empty file.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any


def atomic_write(path: Path, write: Callable[[Path], Any]) -> None:
    """Atomically and durably create or replace `path` with what write(tmp) produces.

    Readers never see a partial file; the data and the rename are fsynced.
    """
    tmp = _write_tmp(path, write)
    try:
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    _fsync(path.parent)


def atomic_create(path: Path, write: Callable[[Path], Any]) -> None:
    """Atomically and durably create `path`; FileExistsError, leaving it intact, if it exists.

    Unlike a ``exists()`` check followed by ``atomic_write``, two concurrent creators cannot
    both succeed: the loser's ``os.link`` raises FileExistsError.
    """
    tmp = _write_tmp(path, write)
    try:
        os.link(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    _fsync(path.parent)


def _write_tmp(path: Path, write: Callable[[Path], Any]) -> Path:
    """Run write(tmp) on a fresh temp file beside `path` and fsync it; no temp left on failure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    os.close(fd)
    tmp = Path(name)
    try:
        write(tmp)
        _fsync(tmp)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return tmp


def _fsync(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
