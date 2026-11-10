"""One experiment configuration and one public evaluation entry."""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

from mematlas.core.config import MemAtlasConfig
from mematlas.core.evaluation import EvaluationResult
from mematlas.core.plugin import MethodServices
from mematlas.core.provider_config import ProviderConfig
from mematlas.core.registry import default_registry
from mematlas.evaluation.metrics import Evaluator, default_metric_registry

_PROTOCOLS = {"streaming_memory", "experience_train_test", "experience_learning_curve"}


@dataclass(frozen=True)
class ExperimentConfig:
    """Configure data, methods and measurement points for all experiments.

    answer_model configures the benchmark reader, separately from Memory's
    providers. Pass agent= to evaluate when using an existing Agent.
    Limits count BenchmarkCase instances; checkpoints count completed training
    cases. Omitting a limit selects the complete split.
    """
    methods: tuple[MemAtlasConfig, ...]
    benchmark: Any
    protocol: str = "streaming_memory"
    answer_model: ProviderConfig | Mapping[str, Any] | None = None
    train_split: str = "train"
    split: str = "test"
    train_limit: int | None = None
    test_limit: int | None = None
    checkpoints: tuple[int, ...] = ()
    output_dir: str | None = None
    train_benchmark: Any = None
    metrics: tuple[str, ...] = ()
    seed: int | None = None

    def __post_init__(self):
        methods = tuple(m if isinstance(m, MemAtlasConfig) else MemAtlasConfig.from_dict(m) for m in self.methods)
        if not methods:
            raise ValueError("methods must contain at least one Memory configuration")
        object.__setattr__(self, "methods", methods)
        object.__setattr__(self, "checkpoints", tuple(self.checkpoints))
        object.__setattr__(self, "metrics", tuple(self.metrics))
        if isinstance(self.answer_model, Mapping):
            object.__setattr__(self, "answer_model", ProviderConfig.from_dict(self.answer_model))
        if self.answer_model is not None and not isinstance(self.answer_model, ProviderConfig):
            raise ValueError("answer_model must be a ProviderConfig or provider configuration mapping")
        if self.protocol not in _PROTOCOLS:
            raise ValueError(f"protocol must be one of {sorted(_PROTOCOLS)}")
        for name in ("train_limit", "test_limit"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < 1):
                raise ValueError(f"{name} must be a positive integer")
        if any(type(point) is not int or point < 0 for point in self.checkpoints):
            raise ValueError("checkpoints must contain nonnegative integer training positions")
        if self.protocol != "experience_learning_curve" and self.checkpoints:
            raise ValueError("checkpoints are only used by experience_learning_curve")
        if self.protocol != "streaming_memory" and self.split == self.train_split and self.train_benchmark is None:
            raise ValueError("Training and test splits must be different")
        if self.protocol == "streaming_memory" and (self.train_limit is not None or self.train_benchmark is not None):
            raise ValueError("Training data and train_limit require a training experiment")
        if self.seed is not None and type(self.seed) is not int:
            raise ValueError("seed must be an integer")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExperimentConfig":
        unknown = set(data) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown experiment fields: {sorted(unknown)}")
        values = dict(data)
        values["methods"] = tuple(MemAtlasConfig.load(m) if isinstance(m, (str, Path)) else
                                   m if isinstance(m, MemAtlasConfig) else MemAtlasConfig.from_dict(m)
                                   for m in values.get("methods", ()))
        return cls(**values)

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentConfig":
        source = Path(path).expanduser()
        if source.is_dir():
            source = source / "experiment.json"
        text = source.read_text(encoding="utf-8-sig")
        if source.suffix.lower() == ".json":
            data = json.loads(text)
        elif source.suffix.lower() in {".yaml", ".yml"}:
            import yaml
            data = yaml.safe_load(text)
        elif source.suffix.lower() == ".toml":
            try:
                import tomllib
            except ImportError:
                import tomli as tomllib
            data = tomllib.loads(text)
        else:
            raise ValueError(f"Unsupported experiment format: {source.suffix}")
        if not isinstance(data, Mapping):
            raise ValueError("Experiment configuration must contain an object")
        # Config paths have the same meaning when invoked from Python or CLI.
        values = dict(data)
        values["methods"] = [
            str((source.parent / m).resolve()) if isinstance(m, str) and not Path(m).is_absolute() else m
            for m in values.get("methods", ())
        ]
        from mematlas.evaluation.scenes import resolve_paths
        for key in ("benchmark", "train_benchmark"):
            if key in values:
                values[key] = resolve_paths(values[key], source.parent)
        return cls.from_dict(values)

    def to_dict(self):
        def dataset(value):
            return dict(value) if isinstance(value, Mapping) else dict(value.describe()) if value is not None else None
        return {"methods": [m.to_dict() for m in self.methods], "benchmark": dataset(self.benchmark),
                "protocol": self.protocol, "answer_model": self.answer_model.to_dict() if self.answer_model else None,
                "train_split": self.train_split, "split": self.split, "train_limit": self.train_limit,
                "test_limit": self.test_limit, "checkpoints": list(self.checkpoints), "output_dir": self.output_dir,
                "train_benchmark": dataset(self.train_benchmark), "metrics": list(self.metrics), "seed": self.seed}


class ExperimentRunner:
    """check(config), await evaluate(config, agent=...), results(path)."""

    def __init__(self, *, registry=None, metric_registry=None):
        self.registry = registry or default_registry()
        self.metric_registry = metric_registry or default_metric_registry()

    @staticmethod
    def results(path):
        from mematlas.evaluation.artifacts import read_results
        return read_results(path)

    @staticmethod
    def _services(services, method):
        return (services.get(method.method.method_id) if isinstance(services, Mapping) else services) or MethodServices()

    @staticmethod
    def _dataset(spec):
        if not isinstance(spec, Mapping):
            if not callable(getattr(spec, "cases", None)) or not callable(getattr(spec, "describe", None)):
                raise ValueError("benchmark must be a dataset configuration or a Benchmark")
            return spec
        from mematlas.benchmarks.factory import load_benchmark
        allowed = {"kind", "path", "mode", "top_k", "max_questions_per_conversation", "official_judge"}
        unknown = set(spec) - allowed
        if unknown:
            raise ValueError(f"Unknown benchmark options: {sorted(unknown)}")
        if not spec.get("kind"):
            raise ValueError("benchmark requires kind")
        builtin_without_path = str(spec["kind"]).strip().lower() in {
            "reasoningbank_trajectory", "reasoningbank-trajectories",
            "ael_portfolio_slice", "ael-portfolio-slice",
            "memcon_action_observation", "memcon-action-observation",
        }
        if not spec.get("path") and not builtin_without_path:
            raise ValueError("benchmark requires path")
        return load_benchmark(spec["kind"], spec.get("path"), **{
            key: spec[key] for key in ("mode", "top_k", "max_questions_per_conversation") if key in spec})

    def _prepare(self, config, agent, services):
        from mematlas.evaluation.validation import check_models, memory_services, validate_cases
        from mematlas.runtime.configuration import check_config
        if not isinstance(config, ExperimentConfig):
            raise TypeError("evaluate expects ExperimentConfig")
        scene = None
        errors, reports = [], []
        scene_report = {}
        benchmark = train = tests = None
        try:
            if agent is None:
                from mematlas.evaluation.scenes import scene_for
                agent = scene_for(config, self.registry)
            config = getattr(agent, "config", config) if hasattr(agent, "_evaluate_experiment") else config
            scene = getattr(agent, "_evaluate_experiment", None)
            if scene is None:
                if agent is not None and not callable(agent):
                    raise ValueError("agent must be callable: agent(case, memory) -> MemoryEpisode")
                benchmark = self._dataset(config.benchmark)
                tests = tuple(benchmark.cases(config.split))[:config.test_limit]
                if not tests:
                    raise ValueError(f"No test cases in split {config.split!r}")
                train = ()
                if config.protocol != "streaming_memory":
                    training = self._dataset(config.train_benchmark) if config.train_benchmark is not None else benchmark
                    train = tuple(training.cases(config.train_split))[:config.train_limit]
                    if not train:
                        raise ValueError(f"No training cases in split {config.train_split!r}")
                    if {c.case_id for c in train} & {c.case_id for c in tests}:
                        raise ValueError("Training and test case IDs must be disjoint")
                    if any(p > len(train) for p in config.checkpoints):
                        raise ValueError("A checkpoint exceeds the selected training case count")
                check_models(config, agent, services, self._services, benchmark, self.registry)
            else:
                scene_report = agent._check_experiment(config) or {}
            if config.metrics:
                self.metric_registry.create(config.metrics)
        except Exception as exc:
            errors.append(str(exc))
        for method in config.methods:
            try:
                supplied = self._services(services, method)
                report = check_config(method, registry=self.registry, services=memory_services(supplied))
                if not errors and scene is None:
                    validate_cases(config, self.registry.get(method.method.method_id), train, tests, agent)
            except Exception as exc:
                report = {"ready": False, "valid": False, "errors": [str(exc)]}
            reports.append(report)
        return {**scene_report, "ready": not errors and all(r["ready"] for r in reports), "errors": errors, "methods": reports,
                "protocol_id": config.protocol, "case_count": len(tests) if tests is not None else scene_report.get("primary_test_count")}, (benchmark, train, tests, agent, config)

    def check(self, config, *, agent=None, services=None):
        return self._prepare(config, agent, services)[0]

    async def evaluate(self, config: ExperimentConfig, *, agent=None, services=None, resume_from=None):
        from mematlas.evaluation.execution import _EvaluationRun
        from mematlas.evaluation.reporting import ReportWriter
        report, prepared = self._prepare(config, agent, services)
        if not report["ready"]:
            raise ValueError("Experiment is not ready: " + json.dumps(report, ensure_ascii=False, default=str))
        benchmark, train, tests, agent, config = prepared
        from mematlas.evaluation.scenes import resolve_paths
        config = replace(config, benchmark=resolve_paths(config.benchmark, Path.cwd()),
                         train_benchmark=resolve_paths(config.train_benchmark, Path.cwd()))
        if config.seed is not None:
            random.seed(config.seed)
        root = Path(config.output_dir or Path(".mematlas/evaluations") / uuid4().hex).expanduser().resolve()
        if (root / "results.json").exists():
            raise FileExistsError("Output already contains results; choose a new directory")
        if resume_from and Path(resume_from).resolve() == root:
            raise ValueError("Resume into a new output directory")
        from mematlas.core.files import atomic_write_json
        from mematlas.core.provenance import redact_sensitive
        root.mkdir(parents=True, exist_ok=True)
        atomic_write_json(root / "experiment.json", redact_sensitive(config.to_dict()))
        results = []
        for index, method in enumerate(config.methods, start=1):
            directory = root / "runs" / f"{index:03d}-{method.method.method_id}"
            source = Path(resume_from) / "runs" / directory.name if resume_from else None
            supplied = self._services(services, method)
            evaluator = Evaluator(self.metric_registry.create(config.metrics)) if config.metrics else Evaluator()
            execution = None
            try:
                if hasattr(agent, "_evaluate_experiment"):
                    result = await agent._evaluate_experiment(self, config, method, directory, source, supplied, evaluator)
                else:
                    execution = _EvaluationRun(config, benchmark, train, tests, agent, supplied, evaluator)
                    result = await execution.run(method, self.registry, directory, source)
            except Exception as exc:
                result = EvaluationResult(method.method.method_id,
                    config.benchmark.get("kind", "scene") if hasattr(agent, "_evaluate_experiment") else benchmark.benchmark_id,
                    (), {}, {}, failures=({"stage": "start", "error_type": type(exc).__name__, "message": str(exc)},))
            finally:
                if execution is not None:
                    await execution.close()
            from mematlas.core.provenance import redact_sensitive
            result = replace(result, run_metadata={**result.run_metadata, "output_dir": str(root),
                "experiment": redact_sensitive(config.to_dict())})
            results.append(result)
            ReportWriter().write(results, root)
        return results
