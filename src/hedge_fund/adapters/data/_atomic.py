"""Atomic file writes: write a temp file in the target directory, then os.replace."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any


def atomic_write(path: Path, write: Callable[[Path], Any]) -> None:
    """Create `path` by calling write(tmp) and renaming, so readers never see a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    os.close(fd)
    tmp = Path(name)
    try:
        write(tmp)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
