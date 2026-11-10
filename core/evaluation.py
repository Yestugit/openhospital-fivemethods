"""Evaluation data and the dataset/scoring boundary, independent of integrations."""
from dataclasses import dataclass, field
import json
from typing import Any, Iterable, Mapping, Protocol

from mematlas.core.models import Content, MemoryContribution, MemoryEpisode, MemoryEvent, Scope


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    query: Content
    scope: Scope = field(default_factory=dict)
    history: tuple[MemoryEvent, ...] = ()
    reference: Any = None
    episode: MemoryEpisode | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Prediction:
    case_id: str
    output: Any
    reference: Any
    applied_memory: tuple[MemoryContribution, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self):
        return {"case_id": self.case_id, "output": self.output, "reference": self.reference,
                "applied_memory": [row.to_dict() for row in self.applied_memory], "metadata": dict(self.metadata)}


    @classmethod
    def from_dict(cls, data):
        values = dict(data)
        values['applied_memory'] = tuple(MemoryContribution.from_dict(row) for row in values.get('applied_memory', ()))
        return cls(**values)


@dataclass(frozen=True)
class EvaluationResult:
    method_id: str
    benchmark_id: str
    predictions: tuple[Prediction, ...]
    metrics: Mapping[str, Any]
    resource_metrics: Mapping[str, Any]
    failures: tuple[Mapping[str, Any], ...] = ()
    run_metadata: Mapping[str, Any] = field(default_factory=dict)
    traces: tuple[Mapping[str, Any], ...] = ()
    learning_curve: tuple[Mapping[str, Any], ...] = ()

    def __post_init__(self):
        # Monetary accounting stays in raw usage/recovery records. Public
        # measurements retain tokens, timing, resources and source attribution.
        object.__setattr__(self, "resource_metrics", _resource_measurements(self.resource_metrics))
        object.__setattr__(self, "run_metadata", {key: value for key, value in self.run_metadata.items()
                                               if key != "cost_summary"})
        points = []
        for point in self.learning_curve:
            row = {key: value for key, value in point.items() if not _fee_field(key)}
            for key in ("resource_metrics", "window_usage"):
                if key in row:
                    row[key] = _resource_measurements(row[key])
            points.append(row)
        object.__setattr__(self, "learning_curve", tuple(points))

    def to_dict(self):
        value = {"method_id": self.method_id, "status": "failed" if self.failures else "completed",
                "benchmark_id": self.benchmark_id, "predictions": [row.to_dict() for row in self.predictions],
                "metrics": dict(self.metrics), "resource_metrics": dict(self.resource_metrics),
                "failures": list(self.failures), "run_metadata": dict(self.run_metadata), "traces": list(self.traces), "learning_curve": list(self.learning_curve)}
        return json.loads(json.dumps(value, ensure_ascii=False, default=str))


    @classmethod
    def from_dict(cls, data):
        values = dict(data)
        values.pop('status', None)
        values['predictions'] = tuple(Prediction.from_dict(row) for row in values.get('predictions', ()))
        for name in ('failures', 'traces', 'learning_curve'):
            values[name] = tuple(values.get(name, ()))
        return cls(**values)


class Benchmark(Protocol):
    """Concrete benchmarks supply cases and their original async scoring."""
    benchmark_id: str
    def cases(self, split: str = "test") -> Iterable[BenchmarkCase]: ...
    async def native_score_async(self, predictions: list[Prediction]) -> Mapping[str, Any]: ...
    def describe(self) -> Mapping[str, Any]: ...


def _fee_field(name):
    # Keep attribution and coverage even when their names mention cost.
    name = name.lower()
    if name.endswith(("_source", "_coverage")):
        return False
    return bool({"cost", "currency"} & set(name.replace(".", "_").split("_")))


def _resource_measurements(values):
    return {key: _resource_measurements(value) if isinstance(value, Mapping) else value
            for key, value in values.items() if not _fee_field(key)}
