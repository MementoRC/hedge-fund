"""Tests for the TradingAgents-backed researcher (offline: stub graph, fake modules)."""

import dataclasses
import sys
import threading
import types
import uuid
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import Any, cast

import pytest
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, Generation, LLMResult

from hedge_fund.adapters.research import tradingagents as ta
from hedge_fund.adapters.research.tradingagents import ResearchConfig, TradingAgentsResearcher
from hedge_fund.domain.cost import ModelUsage, ResearchRunError
from hedge_fund.domain.research import Rating

AS_OF = date(2026, 9, 30)
ANALYSTS = ("market", "social", "news", "fundamentals")
BASE = {"llm_provider": "openai", "results_dir": "/base/results", "unrelated": "kept"}
FULL_STATE = {
    "market_report": "M",
    "sentiment_report": "S",
    "news_report": "N",
    "fundamentals_report": "F",
    "final_trade_decision": "FINAL: buy",
}


def _llm_result(
    response_metadata: dict[str, Any],
    input_tokens: int,
    output_tokens: int,
    *,
    cache_read: int = 0,
    cache_creation: int = 0,
) -> LLMResult:
    """One chat call's result with LangChain ``usage_metadata`` (input includes cached tokens)."""
    message = AIMessage(
        content="ok",
        usage_metadata={
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "input_token_details": {"cache_read": cache_read, "cache_creation": cache_creation},
        },
        response_metadata=response_metadata,
    )
    return LLMResult(generations=[[ChatGeneration(message=message)]])


OPUS_CALL = _llm_result({"model": "claude-opus-5-5"}, 1000, 200, cache_read=300, cache_creation=100)
SONNET_CALL = _llm_result({"model_name": "claude-sonnet-5"}, 500, 50)
BROKEN_CALL = types.SimpleNamespace(  # usage the counter cannot read: int("lots") raises
    generations=[
        [
            types.SimpleNamespace(
                message=types.SimpleNamespace(
                    usage_metadata={"input_tokens": "lots"}, response_metadata={}
                )
            )
        ]
    ]
)


def _install(monkeypatch: pytest.MonkeyPatch, name: str, **attrs: Any) -> types.ModuleType:
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, name, module)
    return module


@pytest.fixture(autouse=True)
def fake_default_config(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, "tradingagents.default_config", build_default_config=lambda: dict(BASE))


class StubFactory:
    """Graph factory that is also the graph: records config and what it saw at propagate time.

    ``propagate`` reports each of ``calls`` to every callback (as the fork's LLM clients
    would), then raises ``error`` or returns.
    """

    def __init__(
        self,
        state: dict[str, Any] | None = None,
        signal: str = "Buy",
        error: Exception | None = None,
        calls: Sequence[Any] = (),
    ) -> None:
        self.state = FULL_STATE if state is None else state
        self.signal = signal
        self.error = error
        self.calls = calls
        self.config: dict[str, Any] = {}
        self.analysts: Sequence[str] = ()
        self.callbacks: list[BaseCallbackHandler] = []
        self.propagate_calls: list[tuple[str, str]] = []
        self.work_dir: Path | None = None
        self.work_existed = False
        self.memory_dir_existed = True

    def __call__(
        self,
        config: dict[str, Any],
        analysts: Sequence[str],
        callbacks: list[BaseCallbackHandler],
    ) -> "StubFactory":
        self.config = config
        self.analysts = analysts
        self.callbacks = callbacks
        return self

    def propagate(self, ticker: str, date_str: str) -> tuple[dict[str, Any], str]:
        self.propagate_calls.append((ticker, date_str))
        self.work_dir = Path(self.config["results_dir"]).parent
        self.work_existed = self.work_dir.is_dir()
        self.memory_dir_existed = (self.work_dir / "memory").exists()
        for response in self.calls:
            for callback in self.callbacks:
                callback.on_llm_end(response, run_id=uuid.uuid4())
        if self.error is not None:
            raise self.error
        return self.state, self.signal


def _researcher(root: Path, factory: StubFactory) -> TradingAgentsResearcher:
    return TradingAgentsResearcher(ResearchConfig(root=root), factory)


def test_research_config_defaults_and_frozen(tmp_path: Path) -> None:
    config = ResearchConfig(root=tmp_path)
    assert config.provider == "anthropic"
    assert config.deep_model == "claude-opus-5-5"
    assert config.quick_model == "claude-sonnet-5"
    assert config.debate_rounds == 1
    assert config.risk_rounds == 1
    assert config.analysts == ANALYSTS
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.provider = "openai"  # type: ignore[misc]


def test_run_returns_a_record(tmp_path: Path) -> None:
    factory = StubFactory()
    record = _researcher(tmp_path / "nested" / "research", factory).run("AAPL", AS_OF, 2)
    assert record.ticker == "AAPL"
    assert record.as_of == AS_OF
    assert record.run_index == 2
    assert record.rating is Rating.BUY
    assert record.decision == "FINAL: buy"
    assert record.reports == {
        "market_report": "M",
        "sentiment_report": "S",
        "news_report": "N",
        "fundamentals_report": "F",
    }
    assert record.seconds >= 0.0
    assert record.error is None
    assert record.valid
    assert factory.propagate_calls == [("AAPL", "2026-09-30")]


def test_graph_config_keys_and_temp_layout(tmp_path: Path) -> None:
    root = tmp_path / "nested" / "research"
    factory = StubFactory()
    _researcher(root, factory).run("AAPL", AS_OF, 0)
    config = factory.config
    work = factory.work_dir
    assert work is not None
    assert work.parent == root
    assert factory.analysts == ANALYSTS
    assert config["llm_provider"] == "anthropic"  # explicit values win over the base config
    assert config["deep_think_llm"] == "claude-opus-5-5"
    assert config["quick_think_llm"] == "claude-sonnet-5"
    assert config["max_debate_rounds"] == 1
    assert config["max_risk_discuss_rounds"] == 1
    assert config["results_dir"] == str(work / "results")
    assert config["data_cache_dir"] == str(work / "cache")
    assert config["memory_log_path"] == str(work / "memory" / "trading_memory.md")
    assert config["unrelated"] == "kept"  # base config is the starting point
    assert factory.work_existed
    assert not factory.memory_dir_existed  # path only, not pre-created


def test_custom_config_values_are_passed(tmp_path: Path) -> None:
    factory = StubFactory()
    config = ResearchConfig(
        root=tmp_path,
        provider="openai",
        deep_model="deep",
        quick_model="quick",
        debate_rounds=3,
        risk_rounds=2,
        analysts=("market",),
    )
    TradingAgentsResearcher(config, factory).run("AAPL", AS_OF, 0)
    assert factory.analysts == ("market",)
    assert factory.config["llm_provider"] == "openai"
    assert factory.config["deep_think_llm"] == "deep"
    assert factory.config["quick_think_llm"] == "quick"
    assert factory.config["max_debate_rounds"] == 3
    assert factory.config["max_risk_discuss_rounds"] == 2


def test_temp_dir_removed_after_success(tmp_path: Path) -> None:
    _researcher(tmp_path / "r", StubFactory()).run("AAPL", AS_OF, 0)
    assert list((tmp_path / "r").iterdir()) == []


def test_temp_dir_removed_even_on_exception(tmp_path: Path) -> None:
    factory = StubFactory(error=RuntimeError("graph blew up"))
    with pytest.raises(ResearchRunError, match="RuntimeError: graph blew up"):
        _researcher(tmp_path / "r", factory).run("AAPL", AS_OF, 0)
    assert list((tmp_path / "r").iterdir()) == []


@pytest.mark.parametrize(
    ("signal", "expected"),
    [
        ("Buy", Rating.BUY),
        ("Overweight", Rating.OVERWEIGHT),
        ("Hold", Rating.HOLD),
        ("Underweight", Rating.UNDERWEIGHT),
        ("Sell", Rating.SELL),
        ("REVIEW", Rating.REVIEW),
    ],
)
def test_signal_maps_to_rating(tmp_path: Path, signal: str, expected: Rating) -> None:
    record = _researcher(tmp_path, StubFactory(signal=signal)).run("AAPL", AS_OF, 0)
    assert record.rating is expected


def test_unknown_signal_is_a_research_run_error_and_cleans_up(tmp_path: Path) -> None:
    with pytest.raises(ResearchRunError, match="ValueError: 'Maybe'") as info:
        _researcher(tmp_path / "r", StubFactory(signal="Maybe")).run("AAPL", AS_OF, 0)
    assert isinstance(info.value.__cause__, ValueError)
    assert list((tmp_path / "r").iterdir()) == []


def test_missing_reports_default_to_empty_strings(tmp_path: Path) -> None:
    record = _researcher(tmp_path, StubFactory(state={})).run("AAPL", AS_OF, 0)
    assert record.decision == ""
    assert record.reports == {
        "market_report": "",
        "sentiment_report": "",
        "news_report": "",
        "fundamentals_report": "",
    }


def test_default_factory_builds_the_graph_under_the_lock(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    seen: dict[str, Any] = {}

    class FakeGraph:
        def __init__(
            self, *, selected_analysts: list[str], config: dict[str, Any], callbacks: list[Any]
        ) -> None:
            seen["locked_at_init"] = ta._GRAPH_LOCK.locked()
            seen["analysts"] = selected_analysts
            seen["config"] = config
            seen["callbacks"] = callbacks

        def propagate(self, ticker: str, date_str: str) -> tuple[dict[str, str], str]:
            seen["locked_at_propagate"] = ta._GRAPH_LOCK.locked()
            return {"final_trade_decision": "ok"}, "Hold"

    _install(monkeypatch, "tradingagents.graph.trading_graph", TradingAgentsGraph=FakeGraph)
    record = TradingAgentsResearcher(ResearchConfig(root=tmp_path / "r")).run("MSFT", AS_OF, 0)
    assert record.rating is Rating.HOLD
    assert record.decision == "ok"
    assert seen["locked_at_init"] is True
    assert seen["locked_at_propagate"] is False
    assert not ta._GRAPH_LOCK.locked()
    assert seen["analysts"] == list(ANALYSTS)
    assert seen["config"]["llm_provider"] == "anthropic"
    assert seen["config"]["unrelated"] == "kept"
    [counter] = seen["callbacks"]
    assert isinstance(counter, ta._UsageCounter)


# --- usage capture -----------------------------------------------------------------------


def test_each_run_gets_one_fresh_counter(tmp_path: Path) -> None:
    factory = StubFactory()
    researcher = _researcher(tmp_path, factory)
    researcher.run("AAPL", AS_OF, 0)
    [first] = factory.callbacks
    researcher.run("AAPL", AS_OF, 1)
    [second] = factory.callbacks
    assert isinstance(first, ta._UsageCounter)
    assert second is not first


def test_a_run_records_its_usage_per_model_sorted_by_model(tmp_path: Path) -> None:
    factory = StubFactory(calls=[SONNET_CALL, OPUS_CALL, OPUS_CALL])
    record = _researcher(tmp_path, factory).run("AAPL", AS_OF, 0)
    assert record.usage == (
        ModelUsage("claude-opus-5-5", 2, 2000, 400, 200, 600),
        ModelUsage("claude-sonnet-5", 1, 500, 50, 0, 0),
    )


def test_a_run_without_llm_calls_has_empty_usage(tmp_path: Path) -> None:
    assert _researcher(tmp_path, StubFactory()).run("AAPL", AS_OF, 0).usage == ()


def test_a_failed_run_raises_with_the_usage_counted_so_far(tmp_path: Path) -> None:
    boom = RuntimeError("boom")
    factory = StubFactory(error=boom, calls=[OPUS_CALL])
    with pytest.raises(ResearchRunError, match="RuntimeError: boom") as info:
        _researcher(tmp_path, factory).run("AAPL", AS_OF, 0)
    assert info.value.__cause__ is boom
    assert info.value.usage == (ModelUsage("claude-opus-5-5", 1, 1000, 200, 100, 300),)


def test_a_failing_factory_raises_with_empty_usage(tmp_path: Path) -> None:
    def factory(
        config: dict[str, Any], analysts: Sequence[str], callbacks: list[BaseCallbackHandler]
    ) -> StubFactory:
        raise OSError("no graph")

    with pytest.raises(ResearchRunError, match="OSError: no graph") as info:
        TradingAgentsResearcher(ResearchConfig(root=tmp_path), factory).run("AAPL", AS_OF, 0)
    assert info.value.usage == ()


def test_a_counter_failure_gives_the_record_unknown_usage(tmp_path: Path) -> None:
    record = _researcher(tmp_path, StubFactory(calls=[BROKEN_CALL, OPUS_CALL])).run(
        "AAPL", AS_OF, 0
    )
    assert record.usage is None
    assert record.rating is Rating.BUY


def _count(*responses: Any) -> tuple[ModelUsage, ...] | None:
    counter = ta._UsageCounter()
    for response in responses:
        counter.on_llm_end(response, run_id=uuid.uuid4())
    return counter.usage()


def test_the_model_comes_from_model_then_model_name_else_unknown() -> None:
    assert _count(
        _llm_result({"model": "a", "model_name": "b"}, 1, 1),
        _llm_result({"model_name": "b"}, 1, 1),
        _llm_result({}, 1, 1),
    ) == (
        ModelUsage("a", 1, 1, 1, 0, 0),
        ModelUsage("b", 1, 1, 1, 0, 0),
        ModelUsage("unknown", 1, 1, 1, 0, 0),
    )


def test_usage_is_sorted_by_model_whatever_the_arrival_order() -> None:
    usage = _count(
        _llm_result({"model": "zeta"}, 1, 1),
        _llm_result({"model": "alpha"}, 1, 1),
        _llm_result({"model": "mid"}, 1, 1),
    )
    assert usage is not None
    assert [row.model for row in usage] == ["alpha", "mid", "zeta"]


def test_generations_without_usage_are_not_counted() -> None:
    assert _count(LLMResult(generations=[[Generation(text="plain")]])) == ()


def test_every_generation_of_a_response_is_counted() -> None:
    generation = OPUS_CALL.generations[0][0]
    both = LLMResult(generations=[[generation], [generation]])
    assert _count(both) == (ModelUsage("claude-opus-5-5", 2, 2000, 400, 200, 600),)


def test_the_counter_is_safe_under_concurrent_calls() -> None:
    counter = ta._UsageCounter()

    def report() -> None:
        for _ in range(50):
            counter.on_llm_end(SONNET_CALL, run_id=uuid.uuid4())

    threads = [threading.Thread(target=report) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert counter.usage() == (ModelUsage("claude-sonnet-5", 400, 200_000, 20_000, 0, 0),)


def test_a_counter_failure_marks_usage_unknown_without_raising() -> None:
    counter = ta._UsageCounter()
    counter.on_llm_end(cast(LLMResult, BROKEN_CALL), run_id=uuid.uuid4())  # must not raise
    counter.on_llm_end(SONNET_CALL, run_id=uuid.uuid4())
    assert counter.usage() is None
