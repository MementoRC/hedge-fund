"""Tests for the SEC EDGAR companyfacts adapter (offline: get, clock and sleep injected)."""

import time
from typing import Any

import pandas as pd
import pytest
import requests

from hedge_fund.adapters.data.edgar_provider import EdgarProvider
from hedge_fund.ports.fundamentals import FACT_COLUMNS, FundamentalsProviderError

CIK = "0000320193"
AGENT = "hedge-fund tests tests@example.com"
URL = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{CIK}.json"
Q1 = "0000320193-24-000010"
KA = "0000320193-24-000002"


def _entry(end: str, val: float, accn: str, form: str, filed: str, **extra: Any) -> dict[str, Any]:
    return {
        "end": end,
        "val": val,
        "accn": accn,
        "fy": 2024,
        "fp": "Q1",
        "form": form,
        "filed": filed,
        **extra,
    }


COMPANYFACTS: dict[str, Any] = {
    "cik": 320193,
    "entityName": "Example Inc.",
    "facts": {
        "dei": {
            "EntityCommonStockSharesOutstanding": {
                "units": {"shares": [_entry("2024-04-25", 49000, Q1, "10-Q", "2024-05-01")]}
            },
            "EntityPublicFloat": {
                "units": {"USD": [_entry("2023-09-30", 1e9, KA, "10-K", "2024-03-01")]}
            },
        },
        "us-gaap": {
            "Revenues": {
                "units": {
                    "USD": [
                        _entry("2024-03-31", 110, Q1, "10-Q", "2024-05-01", start="2024-01-01"),
                        _entry(
                            "2023-12-31",
                            400,
                            KA,
                            "10-K/A",
                            "2024-03-01",
                            start="2023-01-01",
                            frame="CY2023",
                        ),
                        _entry("2024-03-31", 111, "x", "8-K", "2024-05-02", start="2024-01-01"),
                    ],
                    "EUR": [_entry("2024-03-31", 99, Q1, "10-Q", "2024-05-01", start="2024-01-01")],
                }
            },
            "Assets": {
                "units": {
                    "USD": [
                        _entry("2024-03-31", 1050, Q1, "10-Q", "2024-05-01"),
                        _entry("2024-03-31", 1049, "y", "S-1", "2024-05-03"),
                    ]
                }
            },
            "Goodwill": {"units": {"USD": [_entry("2024-03-31", 5, Q1, "10-Q", "2024-05-01")]}},
        },
    },
}


class FakeTime:
    """Monotonic clock that advances only when slept on or when a fake request takes time."""

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


class FakeResponse:
    def __init__(self, status_code: int = 200, payload: Any = COMPANYFACTS) -> None:
        self.status_code = status_code
        self._payload = payload

    def json(self) -> Any:
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


class FakeGet:
    """Stand-in for requests.get: replays outcomes in order and records each call."""

    def __init__(
        self,
        *outcomes: FakeResponse | Exception,
        clock: FakeTime | None = None,
        latency: float = 0.0,
    ) -> None:
        self.outcomes = list(outcomes)
        self.clock = clock
        self.latency = latency
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def __call__(self, url: str, **kwargs: Any) -> FakeResponse:
        self.calls.append((url, kwargs))
        if self.clock is not None:
            self.clock.now += self.latency
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _provider(
    get: FakeGet, clock: FakeTime | None = None, ciks: dict[str, str] | None = None
) -> EdgarProvider:
    fake_time = clock if clock is not None else FakeTime()
    return EdgarProvider(
        ciks if ciks is not None else {"AAA": CIK},
        get=get,
        user_agent=AGENT,
        clock=fake_time.clock,
        sleep=fake_time.sleep,
    )


def test_keeps_curated_tags_in_usd_or_shares_from_report_forms() -> None:
    facts = _provider(FakeGet(FakeResponse())).facts(["AAA"])
    assert tuple(facts.columns) == FACT_COLUMNS
    assert facts["tag"].tolist() == [
        "us-gaap:Revenues",
        "us-gaap:Revenues",
        "us-gaap:Assets",
        "dei:EntityCommonStockSharesOutstanding",
    ]
    assert facts["value"].tolist() == [110.0, 400.0, 1050.0, 49000.0]
    assert facts["form"].tolist() == ["10-Q", "10-K/A", "10-Q", "10-Q"]
    assert facts["accn"].tolist() == [Q1, KA, Q1, Q1]
    assert set(facts["ticker"]) == {"AAA"}


def test_dates_are_parsed_and_instants_have_nat_start() -> None:
    facts = _provider(FakeGet(FakeResponse())).facts(["AAA"])
    assert facts.loc[0, "start"] == pd.Timestamp("2024-01-01")
    assert facts.loc[0, "end"] == pd.Timestamp("2024-03-31")
    assert facts.loc[0, "filed"] == pd.Timestamp("2024-05-01")
    assert pd.isna(facts.loc[2, "start"])


def test_request_url_user_agent_and_timeout() -> None:
    get = FakeGet(FakeResponse())
    _provider(get).facts(["AAA"])
    assert get.calls == [(URL, {"headers": {"User-Agent": AGENT}, "timeout": 30})]


def test_404_gives_no_rows() -> None:
    facts = _provider(FakeGet(FakeResponse(404))).facts(["AAA"])
    assert facts.empty
    assert tuple(facts.columns) == FACT_COLUMNS


def test_ticker_without_cik_gives_no_rows_and_no_request() -> None:
    get = FakeGet()
    assert _provider(get).facts(["ZZZ"]).empty
    assert get.calls == []


def test_tickers_are_deduplicated_and_concatenated() -> None:
    get = FakeGet(FakeResponse(), FakeResponse())
    facts = _provider(get, ciks={"AAA": CIK, "BBB": "0000000002"}).facts(["AAA", "BBB", "AAA"])
    assert len(get.calls) == 2
    assert facts["ticker"].tolist() == ["AAA"] * 4 + ["BBB"] * 4


def test_tickers_sharing_a_cik_each_get_rows_under_their_own_ticker() -> None:
    get = FakeGet(FakeResponse(), FakeResponse())
    facts = _provider(get, ciks={"GOOG": CIK, "GOOGL": CIK}).facts(["GOOG", "GOOGL"])
    assert facts["ticker"].tolist() == ["GOOG"] * 4 + ["GOOGL"] * 4
    assert [url for url, _ in get.calls] == [URL, URL]


def test_retries_429_5xx_timeouts_and_connection_errors_with_backoff() -> None:
    fake_time = FakeTime()
    get = FakeGet(FakeResponse(429), FakeResponse(503), requests.Timeout("slow"), FakeResponse())
    facts = _provider(get, fake_time).facts(["AAA"])
    assert len(get.calls) == 4
    assert fake_time.sleeps == [1.0, 2.0, 4.0]
    assert len(facts) == 4


@pytest.mark.parametrize("outcome", [FakeResponse(500), requests.ConnectionError("down")])
def test_gives_up_after_three_retries(outcome: FakeResponse | Exception) -> None:
    get = FakeGet(*[outcome] * 4)
    with pytest.raises(FundamentalsProviderError, match="after 3 retries"):
        _provider(get).facts(["AAA"])
    assert len(get.calls) == 4


def test_other_http_status_fails_without_retry() -> None:
    get = FakeGet(FakeResponse(403))
    with pytest.raises(FundamentalsProviderError, match="HTTP 403"):
        _provider(get).facts(["AAA"])
    assert len(get.calls) == 1


def test_other_request_errors_fail_without_retry() -> None:
    boom = requests.exceptions.InvalidURL("bad url")
    get = FakeGet(boom)
    with pytest.raises(FundamentalsProviderError) as info:
        _provider(get).facts(["AAA"])
    assert info.value.__cause__ is boom
    assert len(get.calls) == 1


@pytest.mark.parametrize("payload", [ValueError("not json"), ["not", "a", "mapping"]])
def test_invalid_json_payload_fails(payload: Any) -> None:
    with pytest.raises(FundamentalsProviderError, match="JSON"):
        _provider(FakeGet(FakeResponse(200, payload))).facts(["AAA"])


def test_malformed_entry_fails() -> None:
    payload = {"facts": {"us-gaap": {"Assets": {"units": {"USD": [{"form": "10-Q"}]}}}}}
    with pytest.raises(FundamentalsProviderError, match="malformed"):
        _provider(FakeGet(FakeResponse(200, payload))).facts(["AAA"])


def _assets_payload(entries: Any) -> dict[str, Any]:
    return {"facts": {"us-gaap": {"Assets": {"units": {"USD": entries}}}}}


@pytest.mark.parametrize(
    "entry",
    [
        _entry("2024-03-31", "abc", Q1, "10-Q", "2024-05-01"),  # type: ignore[arg-type]
        _entry("not-a-date", 1, Q1, "10-Q", "2024-05-01"),
        _entry("2024-03-31", 1, Q1, "10-Q", "also-bad"),
        {**_entry("2024-03-31", 1, Q1, "10-Q", "2024-05-01"), "accn": None},
    ],
)
def test_uncoercible_values_raise_provider_error_chained(entry: dict[str, Any]) -> None:
    get = FakeGet(FakeResponse(200, _assets_payload([entry])))
    with pytest.raises(FundamentalsProviderError, match="malformed") as info:
        _provider(get).facts(["AAA"])
    assert info.value.__cause__ is not None


def test_missing_accn_raises_provider_error_chained() -> None:
    entry = _entry("2024-03-31", 1, Q1, "10-Q", "2024-05-01")
    del entry["accn"]
    with pytest.raises(FundamentalsProviderError, match="malformed") as info:
        _provider(FakeGet(FakeResponse(200, _assets_payload([entry])))).facts(["AAA"])
    assert isinstance(info.value.__cause__, KeyError)


def test_non_list_unit_entries_raise_provider_error() -> None:
    payload = _assets_payload({"end": "2024-03-31"})
    with pytest.raises(FundamentalsProviderError, match="not a list"):
        _provider(FakeGet(FakeResponse(200, payload))).facts(["AAA"])


@pytest.mark.parametrize("cik", [None, 320193, b"0000320193"])
def test_non_string_cik_rejected_at_construction(cik: Any) -> None:
    with pytest.raises(FundamentalsProviderError, match="AAA"):
        _provider(FakeGet(), ciks={"AAA": cik})


@pytest.mark.parametrize("cik", ["../x", "1?x", "123", "００００００００００", "0000320193\n"])
def test_invalid_cik_rejected_at_construction_without_request(cik: str) -> None:
    get = FakeGet()
    with pytest.raises(FundamentalsProviderError, match="AAA"):
        _provider(get, ciks={"AAA": cik})
    assert get.calls == []


def test_requests_are_spaced_at_least_0_12_seconds() -> None:
    fake_time = FakeTime()
    get = FakeGet(FakeResponse(), FakeResponse())
    _provider(get, fake_time, ciks={"AAA": CIK, "BBB": "0000000002"}).facts(["AAA", "BBB"])
    assert fake_time.sleeps == [pytest.approx(0.12)]


def test_no_wait_when_enough_time_has_passed() -> None:
    fake_time = FakeTime()
    get = FakeGet(FakeResponse(), FakeResponse(), clock=fake_time, latency=0.5)
    _provider(get, fake_time, ciks={"AAA": CIK, "BBB": "0000000002"}).facts(["AAA", "BBB"])
    assert fake_time.sleeps == []


def test_user_agent_falls_back_to_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SEC_USER_AGENT", "env agent env@example.com")
    get = FakeGet(FakeResponse())
    EdgarProvider({"AAA": CIK}, get=get).facts(["AAA"])
    assert get.calls[0][1]["headers"] == {"User-Agent": "env agent env@example.com"}


def test_missing_user_agent_raises_at_construction(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SEC_USER_AGENT", raising=False)
    with pytest.raises(FundamentalsProviderError, match="SEC_USER_AGENT"):
        EdgarProvider({"AAA": CIK})


def test_defaults_are_requests_get_and_monotonic_time() -> None:
    provider = EdgarProvider({}, user_agent=AGENT)
    assert provider._get is requests.get
    assert provider._clock is time.monotonic
    assert provider._sleep is time.sleep
