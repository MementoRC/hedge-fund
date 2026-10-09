"""Tests for the DataProvider port."""

from datetime import date

from hedge_fund.ports.data import DataProvider, DataProviderError
from tests.factories import make_closes
from tests.fakes import FakeProvider


def test_data_provider_error_is_an_exception() -> None:
    assert issubclass(DataProviderError, Exception)


def test_fake_provider_satisfies_the_port() -> None:
    provider: DataProvider = FakeProvider(make_closes(["AAA"]), {"AAA": "Energy"})
    assert provider.sectors(["AAA"]) == {"AAA": "Energy"}
    closes = provider.closes(["AAA", "ZZZ"], date(2026, 9, 1), date(2026, 9, 30))
    assert list(closes.columns) == ["AAA", "ZZZ"]
    assert closes["ZZZ"].isna().all()
