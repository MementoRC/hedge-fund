"""TradingAgents-backed Researcher: one isolated, freshly configured graph run per call.

Each run counts its LLM usage with a fresh callback handler passed through the fork's
``callbacks`` argument, which forwards it to both the deep and the quick client.
"""

from __future__ import annotations

import importlib
import shutil
import tempfile
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Protocol, cast

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from hedge_fund.domain.cost import ModelUsage, ResearchRunError
from hedge_fund.domain.research import Rating, RunRecord
from hedge_fund.engine.cost import merge_usage

# TradingAgentsGraph.__init__ merges into a process-global config dict without a lock.
_GRAPH_LOCK = threading.Lock()
_REPORT_KEYS = ("market_report", "sentiment_report", "news_report", "fundamentals_report")
_DECISION_KEY = "final_trade_decision"
_UNKNOWN_MODEL = "unknown"


class Graph(Protocol):
    """Anything with TradingAgentsGraph's ``propagate`` shape."""

    def propagate(self, ticker: str, date_str: str) -> tuple[Mapping[str, Any], str]: ...


GraphFactory = Callable[[dict[str, Any], Sequence[str], list[BaseCallbackHandler]], Graph]


@dataclass(frozen=True)
class ResearchConfig:
    """Where per-run scratch dirs live, and which models and analysts TradingAgents uses."""

    root: Path
    provider: str = "anthropic"
    deep_model: str = "claude-opus-5-5"
    quick_model: str = "claude-sonnet-5"
    debate_rounds: int = 1
    risk_rounds: int = 1
    analysts: tuple[str, ...] = ("market", "social", "news", "fundamentals")


class _UsageCounter(BaseCallbackHandler):
    """One run's per-call LLM usage from ``on_llm_end``; never raises into the graph.

    The graph may call LLMs from several threads, so a lock guards the calls. Any error
    reading a response marks the run's usage unknown: ``usage()`` is then None.
    """

    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.Lock()
        self._calls: list[ModelUsage] = []
        self._failed = False

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        try:
            calls = [
                _call_usage(generation) for batch in response.generations for generation in batch
            ]
        except Exception:
            with self._lock:
                self._failed = True
            return
        with self._lock:
            self._calls.extend(call for call in calls if call is not None)

    def usage(self) -> tuple[ModelUsage, ...] | None:
        """The calls so far summed per model, sorted by model; None after a counter error."""
        with self._lock:
            return None if self._failed else merge_usage(self._calls)


def _call_usage(generation: Any) -> ModelUsage | None:
    """One call's usage from a generation's message; None when it has no ``usage_metadata``.

    The model is ``response_metadata["model"]``, else ``["model_name"]``, else ``"unknown"``;
    cache tokens come from ``input_token_details`` (``cache_creation``, ``cache_read``).
    """
    message = getattr(generation, "message", None)
    metadata = getattr(message, "usage_metadata", None)
    if not metadata:
        return None
    details = metadata.get("input_token_details") or {}
    response = getattr(message, "response_metadata", None) or {}
    return ModelUsage(
        model=str(response.get("model") or response.get("model_name") or _UNKNOWN_MODEL),
        calls=1,
        input_tokens=int(metadata.get("input_tokens") or 0),
        output_tokens=int(metadata.get("output_tokens") or 0),
        cache_write_tokens=int(details.get("cache_creation") or 0),
        cache_read_tokens=int(details.get("cache_read") or 0),
    )


class TradingAgentsResearcher:
    """Researcher running the forked TradingAgents, isolated per run.

    ``graph_factory`` builds the graph from (config, analysts, callbacks); None means the
    real TradingAgentsGraph, lazily imported so this module loads without tradingagents.
    """

    def __init__(self, config: ResearchConfig, graph_factory: GraphFactory | None = None) -> None:
        self._config = config
        self._factory: GraphFactory = graph_factory or _tradingagents_graph

    def run(self, ticker: str, as_of: date, run_index: int) -> RunRecord:
        """One full TradingAgents pass; raises on any failure (the service records it).

        Each run gets a fresh scratch dir under ``root`` (results, cache and an empty
        memory log, so no reflection calls and no cross-run state), removed on exit, and a
        fresh usage counter. Any failure in the graph, an unknown signal string included,
        raises ResearchRunError from the original exception, carrying the usage so far.
        """
        self._config.root.mkdir(parents=True, exist_ok=True)
        work = Path(tempfile.mkdtemp(dir=self._config.root))
        counter = _UsageCounter()
        try:
            return self._run(work, counter, ticker, as_of, run_index)
        except Exception as exc:
            raise ResearchRunError(f"{type(exc).__name__}: {exc}", counter.usage()) from exc
        finally:
            shutil.rmtree(work, ignore_errors=True)

    def _run(
        self, work: Path, counter: _UsageCounter, ticker: str, as_of: date, run_index: int
    ) -> RunRecord:
        graph = self._factory(self._graph_config(work), self._config.analysts, [counter])
        started = time.perf_counter()
        state, signal = graph.propagate(ticker, as_of.isoformat())
        seconds = time.perf_counter() - started
        return RunRecord(
            ticker=ticker,
            as_of=as_of,
            run_index=run_index,
            rating=Rating(signal),
            decision=str(state.get(_DECISION_KEY) or ""),
            reports={key: str(state.get(key) or "") for key in _REPORT_KEYS},
            seconds=seconds,
            error=None,
            usage=counter.usage(),
        )

    def _graph_config(self, work: Path) -> dict[str, Any]:
        """Fresh TradingAgents config; our keys are set after the env-folding base call."""
        config = _base_config()
        config.update(
            {
                "llm_provider": self._config.provider,
                "deep_think_llm": self._config.deep_model,
                "quick_think_llm": self._config.quick_model,
                "max_debate_rounds": self._config.debate_rounds,
                "max_risk_discuss_rounds": self._config.risk_rounds,
                "results_dir": str(work / "results"),
                "data_cache_dir": str(work / "cache"),
                "memory_log_path": str(work / "memory" / "trading_memory.md"),
            }
        )
        return config


def _base_config() -> dict[str, Any]:
    module = importlib.import_module("tradingagents.default_config")
    return dict(module.build_default_config())


def _tradingagents_graph(
    config: dict[str, Any], analysts: Sequence[str], callbacks: list[BaseCallbackHandler]
) -> Graph:
    """Build the real graph; construction is serialized, ``propagate`` runs outside the lock.

    The fork forwards ``callbacks`` to its deep and quick LLM clients.
    """
    module = importlib.import_module("tradingagents.graph.trading_graph")
    with _GRAPH_LOCK:
        return cast(
            Graph,
            module.TradingAgentsGraph(
                selected_analysts=list(analysts), config=config, callbacks=callbacks
            ),
        )
