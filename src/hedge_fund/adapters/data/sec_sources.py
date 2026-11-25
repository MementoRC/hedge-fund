"""SEC lookups the backtest snapshot needs beyond companyfacts: ticker CIKs and SIC codes."""

from __future__ import annotations

from hedge_fund.adapters.data.edgar_provider import EdgarProvider
from hedge_fund.ports.fundamentals import FundamentalsProviderError

COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"


class SecLookups:
    """Current ticker -> CIK map and per-CIK SIC codes through one EdgarProvider's client."""

    def __init__(self, client: EdgarProvider) -> None:
        self._client = client

    def company_ciks(self) -> dict[str, str]:
        """Uppercase ticker -> 10-digit CIK from company_tickers.json (first entry wins).

        Current tickers only: a delisted name is absent. Raises FundamentalsProviderError
        when the file is missing or malformed.
        """
        payload = self._client.get_json(COMPANY_TICKERS_URL)
        if payload is None:
            raise FundamentalsProviderError(f"{COMPANY_TICKERS_URL}: HTTP 404")
        ciks: dict[str, str] = {}
        try:
            for entry in payload.values():
                ciks.setdefault(str(entry["ticker"]).upper(), f"{int(entry['cik_str']):010d}")
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            raise FundamentalsProviderError(f"malformed company_tickers.json: {exc!r}") from exc
        return ciks

    def sic(self, cik: str) -> str | None:
        """The SIC code string in the CIK's submissions file; None if absent or empty."""
        payload = self._client.get_json(SUBMISSIONS_URL.format(cik=cik))
        value = None if payload is None else payload.get("sic")
        return value if isinstance(value, str) and value.strip() else None
