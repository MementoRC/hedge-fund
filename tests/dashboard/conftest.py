"""Dashboard test setup: no real ``data/`` is ever read and every test starts uncached."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
import streamlit as st


@pytest.fixture(autouse=True)
def _isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.chdir(tmp_path)  # the default root ``data/demo`` is relative to the cwd
    st.cache_resource.clear()
    yield
    st.cache_resource.clear()
