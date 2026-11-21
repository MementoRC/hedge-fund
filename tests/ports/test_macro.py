"""Tests for the MacroProvider port."""

from datetime import date

import pandas as pd

from hedge_fund.ports.macro import MacroProvider, MacroProviderError
from tests.fakes import FakeMacroProvider


def test_macro_provider_error_is_an_exception() -> None:
    assert issubclass(MacroProviderError, Exception)


def test_the_fake_satisfies_the_port_and_records_its_calls() -> None:
    days = pd.bdate_range("2024-01-01", periods=5)
    frame = pd.DataFrame(
        {"DGS10": [4.0, 4.1, 4.2, 4.3, 4.4], "VIXCLS": [13.0, 14.0, 15.0, 16.0, 17.0]},
        index=days,
    )
    fake = FakeMacroProvider(frame)
    provider: MacroProvider = fake

    out = provider.series(["VIXCLS", "DGS10"], date(2024, 1, 2), date(2024, 1, 4))

    assert list(out.columns) == ["VIXCLS", "DGS10"]
    assert list(out.index) == list(days[1:4])
    assert fake.calls == [(("VIXCLS", "DGS10"), date(2024, 1, 2), date(2024, 1, 4))]
