"""The dashboard's own wording. Pure: no Streamlit, no I/O.

The shared formatting lives in ``viz/format.py``. This module keeps only the no-SPY text,
which points at the dashboard's sidebar (the site passes its own wording).
"""

from __future__ import annotations

NO_SPY = (
    "No SPY series for this root. A demo root takes SPY from its sibling `<root>.store`; "
    "a problem loading it is listed in the sidebar."
)
