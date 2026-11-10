from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from mematlas.benchmarks.base import BenchmarkAdapter
from mematlas.benchmarks.datasets import JsonDatasetBenchmark
from mematlas.benchmarks.locomo import LoCoMoBenchmark
from mematlas.benchmarks.longmemeval import LongMemEvalBenchmark
from mematlas.benchmarks.reasoningbank_trajectory import ReasoningBankTrajectoryBenchmark
from mematlas.benchmarks.ael_portfolio_slice import AELPortfolioSliceBenchmark
from mematlas.benchmarks.memcon_action_observation import MemConActionObservationBenchmark


def load_benchmark(
    kind: str,
    path: str | Path | None = None,
    *,
    limit: int | None = None,
    mode: str = "text",
    top_k: int | None = None,
    judge: Callable[..., Any] | None = None,
    max_questions_per_conversation: int | None = None,
) -> BenchmarkAdapter:
    normalized = kind.strip().lower()
    if normalized in {"reasoningbank_trajectory", "reasoningbank-trajectories"}:
        return ReasoningBankTrajectoryBenchmark()
    if normalized in {"ael_portfolio_slice", "ael-portfolio-slice"}:
        return AELPortfolioSliceBenchmark()
    if normalized in {"memcon_action_observation", "memcon-action-observation"}:
        return MemConActionObservationBenchmark()
    if path is None:
        raise ValueError(f"Benchmark kind {kind!r} requires a dataset path")
    if normalized == 'mind2web':
        from mematlas.benchmarks.mind2web import Mind2WebBenchmark
        return Mind2WebBenchmark(path)
    if normalized == 'hotpotqa':
        from mematlas.benchmarks.hotpotqa import HotpotQABenchmark
        return HotpotQABenchmark(path)
    if normalized == "locomo":
        return LoCoMoBenchmark(
            path,
            mode=mode,
            max_conversations=limit,
            max_questions_per_conversation=max_questions_per_conversation,
            top_k=top_k,
        )
    if normalized == "longmemeval":
        return LongMemEvalBenchmark(path, limit=limit, top_k=top_k, judge=judge)
    if normalized in {"custom", "json", "jsonl"}:
        return JsonDatasetBenchmark(path, benchmark_id="custom")
    raise ValueError(f"Unsupported benchmark kind: {kind}")


def _configure_execution(benchmark, services):
    """Give native QA its evaluator without exposing gold in actor prompts."""
    if getattr(benchmark, 'benchmark_id', None) != 'mind2web':
        return  # Ordinary benchmarks do not import the native web runtime.
    from mematlas.benchmarks.mind2web import Mind2WebBenchmark
    if isinstance(benchmark, Mind2WebBenchmark):
        services.objects['mind2web_benchmark'] = benchmark


def _scoring_needs_model(spec, benchmark, services):
    from collections.abc import Mapping
    enabled = not isinstance(spec, Mapping) or spec.get("official_judge", True)
    return (enabled and isinstance(benchmark, LongMemEvalBenchmark) and benchmark.judge is None
            and services.object("native_judge") is None)


def _configure_scoring(spec, benchmark, services, llm, model):
    """Attach the benchmark-owned judge without changing the caller's dataset."""
    from collections.abc import Mapping
    if not isinstance(benchmark, LongMemEvalBenchmark):
        return
    if isinstance(spec, Mapping) and not spec.get("official_judge", True):
        benchmark.judge = None
    elif benchmark.judge is None:
        from mematlas.benchmarks.longmemeval import LongMemEvalLLMJudge
        benchmark.judge = services.object("native_judge") or (LongMemEvalLLMJudge(llm, model=model) if llm else None)
