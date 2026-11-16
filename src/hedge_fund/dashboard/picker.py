"""The row selector shared by the Cycles and Backtests pages."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import streamlit as st


def pick_index(
    label: str,
    order: Sequence[int],
    label_of: Callable[[int], str],
    *,
    key: str,
) -> int:
    """The row index picked in a selectbox that lists ``order`` (the row indices, in display
    order) under ``label_of``'s labels.

    ``key`` must be scoped to the root, so an index saved for another root is never read. An
    index out of range (the root's own files changed under the same key) or no index at all
    falls back to the first option listed, ``order[0]``.
    """
    index = st.selectbox(label, list(order), format_func=label_of, key=key)
    in_range = index is not None and 0 <= index < len(order)
    return index if in_range else order[0]
