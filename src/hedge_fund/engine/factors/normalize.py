"""Cross-sectional normalization: winsorization and sector-neutral z-scores."""

from __future__ import annotations

import pandas as pd

MIN_SECTOR_SIZE = 3


def winsorize(values: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    """Clip ``values`` to its own [lower, upper] cross-sectional quantiles."""
    return values.clip(lower=values.quantile(lower), upper=values.quantile(upper))


def sector_zscore(values: pd.Series, sectors: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Z-score ``values`` within each sector (population std).

    Sectors with fewer than MIN_SECTOR_SIZE names, or zero std, use universe mean/std
    instead. Returns ``(z, fallback)`` indexed like ``values``; ``fallback`` marks tickers
    scored against the universe. Where the applicable std is 0, z is 0.
    """
    grouped = values.groupby(sectors)
    mean = grouped.transform("mean")
    std = grouped.transform(lambda group: group.std(ddof=0))
    count = grouped.transform("count")
    fallback = (count < MIN_SECTOR_SIZE) | (std == 0)
    mean = mean.where(~fallback, values.mean())
    std = std.where(~fallback, values.std(ddof=0))
    z = ((values - mean) / std).where(std != 0, 0.0)
    return z, fallback
