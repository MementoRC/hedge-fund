"""The FRED series the macro factors read: never revised and market priced, so point in time."""

from __future__ import annotations

RATE_SERIES = "DGS10"  # 10-year Treasury constant-maturity yield, percent
RISKOFF_SERIES = "VIXCLS"  # CBOE VIX close, index points
MACRO_SERIES: tuple[str, ...] = (RATE_SERIES, RISKOFF_SERIES)
