"""A coarse SEC SIC code to GICS sector map, for members missing from the sector file (pure).

The first matching inclusive range wins, so narrow ranges come before the wider ranges they
sit in. SIC divisions do not line up with GICS: this is a fallback for delisted members
only, and the frozen store lists every name sectored this way (``sic_sectors``).
"""

from __future__ import annotations

SIC_RANGES: tuple[tuple[int, int, str], ...] = (
    (100, 999, "Consumer Staples"),  # agriculture, forestry, fishing
    (1000, 1099, "Materials"),  # metal mining
    (1200, 1399, "Energy"),  # coal, oil and gas extraction and services
    (1400, 1499, "Materials"),  # nonmetallic minerals
    (1500, 1799, "Industrials"),  # construction
    (2000, 2199, "Consumer Staples"),  # food, beverages, tobacco
    (2200, 2399, "Consumer Discretionary"),  # textiles, apparel
    (2400, 2499, "Materials"),  # lumber and wood
    (2500, 2599, "Consumer Discretionary"),  # furniture
    (2600, 2699, "Materials"),  # paper
    (2700, 2799, "Communication Services"),  # printing and publishing
    (2830, 2836, "Health Care"),  # drugs, biologics
    (2840, 2844, "Consumer Staples"),  # soap, detergents, cosmetics
    (2800, 2899, "Materials"),  # other chemicals
    (2900, 2999, "Energy"),  # petroleum refining
    (3000, 3399, "Materials"),  # rubber, plastics, glass, stone, primary metals
    (3570, 3579, "Information Technology"),  # computers, office equipment
    (3600, 3699, "Information Technology"),  # electronic equipment, semiconductors
    (3710, 3716, "Consumer Discretionary"),  # motor vehicles
    (3840, 3851, "Health Care"),  # medical instruments and supplies
    (3400, 3899, "Industrials"),  # fabricated metal, machinery, transport equipment
    (3900, 3999, "Consumer Discretionary"),  # miscellaneous manufacturing
    (4000, 4799, "Industrials"),  # transportation
    (4800, 4899, "Communication Services"),  # telecommunications, broadcasting
    (4900, 4999, "Utilities"),  # electric, gas, water, sanitary
    (5000, 5199, "Industrials"),  # wholesale trade
    (5400, 5499, "Consumer Staples"),  # food stores
    (5912, 5912, "Consumer Staples"),  # drug stores
    (5200, 5999, "Consumer Discretionary"),  # other retail
    (6798, 6798, "Real Estate"),  # real estate investment trusts
    (6500, 6599, "Real Estate"),  # real estate
    (6000, 6799, "Financials"),  # banks, brokers, insurance, holding companies
    (7370, 7379, "Information Technology"),  # software, data processing
    (7800, 7999, "Communication Services"),  # motion pictures, amusement
    (8000, 8099, "Health Care"),  # health services
    (7000, 8999, "Industrials"),  # other services
)


def sic_sector(sic: str | None) -> str | None:
    """The GICS sector for an SEC SIC code string; None if missing, malformed or unmapped."""
    if sic is None or not sic.strip().isdigit():
        return None
    code = int(sic)
    for low, high, sector in SIC_RANGES:
        if low <= code <= high:
            return sector
    return None
