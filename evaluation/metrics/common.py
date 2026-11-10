from __future__ import annotations

import re
import math
import string
from collections import Counter
from abc import ABC, abstractmethod
from pathlib import Path
from statistics import mean, median
from time import perf_counter
from typing import Any, Iterable, Mapping, Sequence

from mematlas.core.evaluation import Benchmark as BenchmarkAdapter
from mematlas.core.evaluation import Prediction
from mematlas.core.models import OperationTrace, utc_now
from mematlas.core.telemetry import TraceCollector
from mematlas.core.telemetry import UsageLedger
from mematlas.core.files import storage_bytes


class Metric(ABC):
    name: str

    @abstractmethod
    def compute(self, predictions: Sequence[Prediction]) -> Any:
        raise NotImplementedError


class MetricRegistry:
    """Named factories for framework and user-defined evaluation metrics."""

    def __init__(self) -> None:
        self._factories: dict[str, Any] = {}

    def register(self, metric: type[Metric] | Metric | Any, *, name: str | None = None, replace: bool = False) -> None:
        metric_name = name or getattr(metric, "name", None)
        if not metric_name:
            raise ValueError("Metric registration requires a name")
        if metric_name in self._factories and not replace:
            raise ValueError(f"Metric already registered: {metric_name}")
        if isinstance(metric, Metric):
            self._factories[metric_name] = lambda metric=metric: metric
        elif isinstance(metric, type):
            self._factories[metric_name] = metric
        else:
            self._factories[metric_name] = metric

    def create(self, names: Iterable[str]) -> tuple[Metric, ...]:
        metrics = []
        for name in names:
            try:
                factory = self._factories[name]
            except KeyError as exc:
                raise ValueError(f"Metric {name!r} is not registered") from exc
            metric = factory()
            if not isinstance(metric, Metric):
                raise TypeError(f"Metric factory {name!r} did not return Metric")
            metrics.append(metric)
        return tuple(metrics)

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._factories))


class ExactMatchMetric(Metric):
    name = "common.exact_match"

    @staticmethod
    def _normalize(value: Any) -> str:
        text = str(value).strip().lower()
        text = "".join(character for character in text if character not in set(string.punctuation))
        text = re.sub(r"\b(a|an|the)\b", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def compute(self, predictions: Sequence[Prediction]) -> float | None:
        labeled = [item for item in predictions if item.reference is not None]
        if not labeled:
            return None
        return sum(
            item.output is not None and not item.metadata.get("evaluation_error")
            and self._normalize(item.output) == self._normalize(item.reference)
            for item in labeled
        ) / len(labeled)


class TokenF1Metric(Metric):
    name = "common.token_f1"

    def compute(self, predictions: Sequence[Prediction]) -> float | None:
        values = []
        for item in predictions:
            if item.reference is None:
                continue
            if item.output is None or item.metadata.get("evaluation_error"):
                values.append(0.0)
                continue
            predicted = ExactMatchMetric._normalize(item.output).split()
            expected = ExactMatchMetric._normalize(item.reference).split()
            if not predicted or not expected:
                values.append(float(predicted == expected))
                continue
            common = sum((Counter(predicted) & Counter(expected)).values())
            if common == 0:
                values.append(0.0)
                continue
            precision = common / len(predicted)
            recall = common / len(expected)
            values.append(2 * precision * recall / (precision + recall))
        return sum(values) / len(values) if values else None


def default_metric_registry() -> MetricRegistry:
    registry = MetricRegistry()
    registry.register(ExactMatchMetric)
    registry.register(TokenF1Metric)
    return registry


class Evaluator:
    def __init__(self, metrics: Iterable[Metric] = (ExactMatchMetric(), TokenF1Metric())):
        self.metrics = tuple(metrics)

    directory_size = staticmethod(storage_bytes)
    @staticmethod
    def _operation_resources(traces: TraceCollector) -> dict[str, Any]:
        all_traces = traces.all()
        retrieve_traces = [trace for trace in all_traces if trace.operation == "search"]
        retrieval_times = [trace.duration_ms for trace in retrieve_traces if trace.success]
        def count(operation: str) -> int:
            return sum(trace.operation == operation for trace in all_traces)

        def failures(operation: str) -> int:
            return sum(trace.operation == operation and not trace.success for trace in all_traces)

        updates = {"add", "episode_end", "update", "delete", "flush"}
        construction = [trace for trace in all_traces
                        if trace.operation in updates | {"setup"}
                        and trace.metadata.get("stage") in {"preparation", "construction"}]
        learning = [trace for trace in all_traces if trace.operation in updates
                    and trace.metadata.get("stage") == "training"]

        return {
            "common.retrieval_count": len(retrieval_times),
            "common.retrieval_failures": sum(not trace.success for trace in retrieve_traces),
            "common.retrieval_time_ms": sum(retrieval_times),
            "common.retrieval_mean_ms": mean(retrieval_times) if retrieval_times else 0.0,
            "common.retrieval_median_ms": median(retrieval_times) if retrieval_times else 0.0,
            "common.retrieval_max_ms": max(retrieval_times) if retrieval_times else 0.0,
            "common.memory_construction_time_s": sum(trace.duration_ms for trace in construction) / 1000,
            "common.memory_learning_time_s": sum(trace.duration_ms for trace in learning) / 1000,
            "common.judging_time_s": sum(trace.duration_ms for trace in all_traces
                                         if trace.operation == "judge_scoring") / 1000,
            "common.add_count": count("add"),
            "common.add_failures": failures("add"),
            "common.episode_end_count": count("episode_end"),
            "common.episode_end_failures": failures("episode_end"),
            "common.update_count": count("update"),
            "common.update_failures": failures("update"),
            "common.delete_count": count("delete"),
            "common.delete_failures": failures("delete"),
            "common.flush_count": count("flush"),
            "common.flush_failures": failures("flush"),
            "common.reset_count": count("reset"),
            "common.reset_failures": failures("reset"),
        }

    @staticmethod
    def _response_resources(predictions: Sequence[Prediction]) -> dict[str, Any]:
        durations = [prediction.metadata.get("response_time_s") for prediction in predictions
                     if not prediction.metadata.get("evaluation_error")]
        measured = [value for value in durations if isinstance(value, (int, float))
                    and not isinstance(value, bool) and math.isfinite(value) and value >= 0]
        return {
            "common.response_count": len(measured),
            "common.response_time_s": sum(measured),
            "common.response_mean_s": mean(measured) if measured else None,
            "common.response_timing_coverage": len(measured) / len(durations) if durations else None,
        }

    @staticmethod
    def _storage_resources(size: int | None) -> dict[str, Any]:
        return {"common.storage_bytes": size,
                "common.memory_storage_mib": size / (1024 ** 2) if size is not None else None,
                "common.storage_scope": "live_method_data_excludes_logs_and_recovery_snapshots"}

    async def evaluate(
        self,
        predictions: Sequence[Prediction],
        *,
        benchmark: BenchmarkAdapter,
        traces: TraceCollector,
        artifact_dir: str | Path | Iterable[str | Path],
        usage: UsageLedger | None = None,
        expected_case_count: int | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        common = {metric.name: metric.compute(predictions) for metric in self.metrics}
        from mematlas.core.telemetry import usage_scope, usage_context
        judging_at, judging_started = utc_now(), perf_counter()
        scoring_error, scored = None, False
        with usage_scope(usage or UsageLedger(), "judge"), usage_context(stage="evaluation", operation="native_scoring"):
            try:
                native_values = await benchmark.native_score_async(list(predictions))
                scored = True
            except Exception as exc:
                scoring_error = exc
                native_values = {"scoring_error": f"{type(exc).__name__}: {exc}"}
            finally:
                existing = traces.all()
                traces.add(OperationTrace(existing[0].method_id if existing else "evaluation",
                    "judge_scoring", judging_at, (perf_counter() - judging_started) * 1000,
                    scored,
                    error_type=type(scoring_error).__name__ if scoring_error else None,
                    error_message=str(scoring_error) if scoring_error else None,
                    metadata={"stage": "evaluation", "phase": "judge"}))
        native = {f"native.{key}": value for key, value in native_values.items()}
        trace_summary = traces.summary()
        resources = {
            "common.operation_count": trace_summary["operations"],
            "common.operation_failures": trace_summary["failures"],
            "common.wall_time_ms": trace_summary["duration_ms"],
            **self._storage_resources(self.directory_size(artifact_dir)),
            **self._operation_resources(traces),
            **self._response_resources(predictions),
        }
        self._add_coverage_resources(resources, predictions, expected_case_count)
        if usage is not None:
            resources.update(usage.resource_metrics())
        return {**common, **native}, resources

    @staticmethod
    def _add_coverage_resources(
        resources: dict[str, Any],
        predictions: Sequence[Prediction],
        expected_case_count: int | None,
    ) -> None:
        failed_count = sum(
            1 for prediction in predictions
            if prediction.metadata.get("evaluation_error")
        )
        missing_output_count = sum(
            prediction.output is None and not prediction.metadata.get("evaluation_error")
            for prediction in predictions
        )
        completed_count = len(predictions) - failed_count - missing_output_count
        expected = expected_case_count if expected_case_count is not None else len(predictions)
        resources.update({
            "common.expected_case_count": expected,
            "common.completed_case_count": completed_count,
            "common.failed_case_count": failed_count,
            "common.missing_output_case_count": missing_output_count,
            "common.case_coverage": completed_count / expected if expected else 1.0,
        })
