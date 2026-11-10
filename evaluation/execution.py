"""One method experiment: Memory lifetime, learning, case execution and results.

ExperimentRunner calls run(). Every question calls execute_case(), which calls
the selected complete QA or user Agent directly. experience.py owns continuous
scheduling; validation.py owns preflight; qa.py owns question answering.
"""
from __future__ import annotations

import inspect
import platform
import sys
from collections.abc import Mapping
from copy import copy
from dataclasses import asdict, is_dataclass, replace
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from time import perf_counter
from types import CodeType
from uuid import uuid4

from mematlas.core.evaluation import EvaluationResult, Prediction
from mematlas.core.episodes import BatchResult, EpisodeResult
from mematlas.core.models import MemoryEpisode, content_text
from mematlas.core.provenance import git_revision, json_sha256, package_version, redact_sensitive
from mematlas.core.scope import scope_key
from mematlas.core.telemetry import ProcessResourceMonitor, usage_context, usage_scope
from mematlas.evaluation.experience import _ExperiencePlan, _ExperienceRunner
from mematlas.evaluation.experience_memory import _MemoryExperience
from mematlas.evaluation.qa import DefaultQA, _qa_type
from mematlas.evaluation.validation import history_episode, memory_services, training_episode, submit_episode
from mematlas.runtime.memory import Memory


class _EvaluationRun:
    def __init__(self, config, benchmark, train, tests, agent, services, evaluator):
        self.config, self.benchmark = config, copy(benchmark)
        self.train, self.tests, self.agent = train, tests, agent
        self.services, self.evaluator = services, evaluator
        self.training = {case.case_id: case for case in train}
        self._last_training_case = {scope_key(case.scope): case.case_id for case in train}
        self.testing = {case.case_id: case for case in tests}
        self.failures, self.predictions = [], []
        self.engine = self.qa = self.memory = None

    async def run(self, method, registry, directory, source=None):
        """Allocate one isolated Memory and select its method's QA once."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.memory = Memory(method, registry=registry, services=memory_services(self.services))
        from mematlas.benchmarks.factory import _configure_execution
        _configure_execution(self.benchmark, self.services)
        self.qa = _qa_type(self.memory.plugin)(self.config, self.services)
        from mematlas.benchmarks.factory import _configure_scoring
        _configure_scoring(self.config.benchmark, self.benchmark, self.services, self.qa.llm, self.qa.model)
        if self.config.protocol == "streaming_memory":
            if source is not None:
                raise ValueError("resume_from is available for training experiments")
            return await self._run_history()
        return await self._run_training(directory, source)

    async def close(self):
        if self.qa is not None:
            await self.qa.close()

    async def _execute_episode(self, case, memory):
        executor = self.agent if self.agent is not None else self.qa
        episode = executor(case, memory)
        if inspect.isawaitable(episode):
            episode = await episode
        if not isinstance(episode, MemoryEpisode):
            raise TypeError("QA or agent(case, memory) must return MemoryEpisode")
        if episode.scope is not None and episode.scope != case.scope:
            raise ValueError("Agent episode scope differs from its benchmark case")
        return replace(episode, scope=case.scope, task=episode.task or content_text(case.query))

    async def execute_case(self, memory, case):
        """The single case entry for history, frozen testing and external Agents."""
        started = perf_counter()
        episode = await self._execute_episode(case, memory)
        metadata = {"case_metadata": dict(case.metadata), "episode_id": episode.episode_id,
                    "response_time_s": perf_counter() - started}
        if self.qa.include_episode_in_prediction:
            metadata["episode"] = episode.to_dict()
        return Prediction(case.case_id, episode.output, case.reference, tuple(episode.applied_memory),
                          metadata=metadata)

    def _prepare_learning(self, case, episode, *, last_for_scope):
        episode = self.qa.prepare_episode(case, episode, last_for_scope=last_for_scope)
        if not isinstance(episode, MemoryEpisode):
            raise TypeError("QA.prepare_episode must return MemoryEpisode")
        return episode

    async def _run_history(self):
        memory = self.memory
        started_at, started, segment = datetime.now(timezone.utc), perf_counter(), uuid4().hex
        monitor = ProcessResourceMonitor()
        await monitor.start()
        try:
            with usage_context(stage="preparation", segment_id=segment):
                await memory._setup()
            for case in self.tests:
                try:
                    with usage_scope(memory.services.usage, "agent"), usage_context(
                            stage="evaluation", segment_id=segment, case_id=case.case_id):
                        with usage_context(stage="construction"):
                            if case.episode is not None or case.history:
                                episode = self._prepare_learning(case, history_episode(case), last_for_scope=True)
                                await submit_episode(memory, episode, scope=case.scope)
                            await memory.flush()
                        self.predictions.append(await self.execute_case(memory, case))
                except Exception as exc:
                    self.failures.append({"stage": "case", "case_id": case.case_id,
                        "error_type": type(exc).__name__, "message": str(exc)})
                    self.predictions.append(self._failed_prediction(case, "case", exc))
        except Exception as exc:
            self.failures.append({"stage": "start", "error_type": type(exc).__name__, "message": str(exc)})
            self.predictions = [self._failed_prediction(case, "start", exc) for case in self.tests]
        finally:
            try:
                with usage_context(stage="teardown", segment_id=segment):
                    await memory._close()
            except Exception as exc:
                self.failures.append({"stage": "close", "error_type": type(exc).__name__, "message": str(exc)})
        try:
            metrics, resources = await self.evaluator.evaluate(self.predictions, benchmark=self.benchmark,
                traces=memory.traces,
                artifact_dir=(
                    *(tuple(memory.services.resource_paths) or (memory.config.runtime.artifact_dir,)),
                    *memory.config.runtime.resource_paths,
                ),
                usage=memory.services.usage, expected_case_count=len(self.tests))
        finally:
            process = await monitor.stop()
        if metrics.get("native.scoring_error"):
            self.failures.append({"stage": "score", "message": metrics["native.scoring_error"]})
        if memory.config.runtime.executor == "service":
            resources.update(self.evaluator._storage_resources(getattr(memory._method, "storage_bytes", None)))
            resources["common.remote_usage_error"] = getattr(memory._method, "usage_error", None)
        resources.update({f"common.{key}": value for key, value in process.items()})
        resources.update({"common.run_wall_time_ms": (perf_counter() - started) * 1000,
            "common.process_memory_scope": "current_process_and_descendants_excludes_external_services",
            "common.operation_time_ms": resources.get("common.wall_time_ms"),
            "common.method_internal_token_tracking": memory.plugin.descriptor.internal_token_tracking})
        return EvaluationResult(memory.method_id, self.benchmark.benchmark_id, tuple(self.predictions), metrics,
            resources, tuple(self.failures), self._history_metadata(started_at, segment),
            tuple(trace.to_dict() for trace in memory.traces.all()))

    def _history_metadata(self, started_at, segment):
        memory = self.memory
        descriptor = memory.plugin.descriptor
        source = memory.services.source_root(memory.method_id)
        if source is None and descriptor.source_subdir:
            candidate = Path(__file__).resolve().parents[3] / "agent-memory-methods" / descriptor.source_subdir
            source = candidate.resolve() if candidate.exists() else None
        return {"run_id": memory.run_id, "experiment_id": segment, "started_at": started_at.isoformat(),
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "config": redact_sensitive(memory.config.to_dict()), "method": descriptor.to_dict(),
            "benchmark": dict(self.benchmark.describe()), "protocol_id": self.config.protocol,
            "protocol": {"protocol_id": self.config.protocol},
            "framework": {"version": package_version(), "git_revision": git_revision(Path(__file__).resolve().parents[1])},
            "method_source": {"path": str(source) if source else None, "git_revision": git_revision(source) if source else None},
            "python": sys.version, "platform": platform.platform()}

    @staticmethod
    def _failed_prediction(case, stage, error):
        return Prediction(case.case_id, None, case.reference, metadata={"case_metadata": dict(case.metadata),
            "evaluation_error": {"stage": stage, "error_type": type(error).__name__, "message": str(error)}})

    async def _train_batch(self, ids, experience, batch_index):
        memory, results = experience.core_memory, []
        offset = sum(len(batch) for batch in self.engine.plan.batches()[:batch_index - 1])
        for index, identity in enumerate(ids, start=1):
            case = self.training[identity]
            saved = experience.metadata.setdefault("training_episodes", {}).get(identity)
            if saved is not None:
                episode = MemoryEpisode.from_dict(saved)
            elif self.agent is not None or self.qa.collect_training_episode(case):
                try:
                    with usage_context(case_id=identity):
                        episode = await self._execute_episode(case, memory)
                except Exception as exc:
                    key = memory._history.identity("execution:" + identity, case.scope)
                    experience.history.record_result(EpisodeResult(key, case_id=identity,
                        execution="failed", learning="skipped", metadata={"scope": case.scope},
                        error={"type": type(exc).__name__, "message": str(exc)}))
                    if memory.state.value == "started":
                        await self.engine._save_progress_snapshot(self.engine._position)
                    raise
            else:
                episode = training_episode(case)
            if saved is None:
                episode = self._prepare_learning(case, episode,
                    last_for_scope=identity == self._last_training_case[scope_key(case.scope)])
            experience.metadata["training_episodes"][identity] = episode.to_dict()
            try:
                with usage_context(case_id=identity):
                    result = await submit_episode(memory, episode, scope=case.scope)
            except Exception as exc:
                key = memory._history.identity(episode.episode_id, case.scope)
                failed = memory._history.results.get(key) or EpisodeResult(key, case_id=identity,
                    feedback="invalid", learning="failed", metadata={"scope": case.scope},
                    error={"type": type(exc).__name__, "message": str(exc)})
                experience.history.record_result(failed)
                if memory.state.value == "started":
                    await self.engine._save_progress_snapshot(self.engine._position)
                raise
            experience.history.record_result(result)
            results.append(result)
            if offset + index > self.engine._position:
                await self.engine.record_case_progress(offset + index)
        return BatchResult(tuple(results))

    async def _test_batch(self, ids, experience, checkpoint, split):
        memory, predictions = experience.core_memory, []
        for identity in ids:
            try:
                with usage_context(case_id=identity, checkpoint=checkpoint):
                    fingerprint = await experience.state_fingerprint()
                    async with experience.frozen_operation(fingerprint):
                        prediction = await self.execute_case(memory, self.testing[identity])
            except Exception as exc:
                self.failures.append({"stage": "case", "case_id": identity, "checkpoint": checkpoint,
                    "error_type": type(exc).__name__, "message": str(exc)})
                raise
            predictions.append(prediction)
            self.predictions = list(predictions)
        metrics, resources = await self.evaluator.evaluate(predictions, benchmark=self.benchmark,
            traces=memory.traces, artifact_dir=tuple(memory.services.resource_paths),
            usage=memory.services.usage, expected_case_count=len(ids))
        if metrics.get("native.scoring_error"):
            raise ValueError(metrics["native.scoring_error"])
        return {**metrics, "metrics": metrics, "predictions": [p.to_dict() for p in predictions],
                "resource_metrics": resources}

    async def _run_training(self, directory, source):
        train_ids = tuple(c.case_id for c in self.train)
        points = (len(train_ids),) if self.config.protocol == "experience_train_test" else tuple(sorted({
            *(self.config.checkpoints or (0, len(train_ids))), len(train_ids)}))
        plan = _ExperiencePlan(train_ids, tuple(c.case_id for c in self.tests), (),
            batch_count=len({p for p in points if p > 0}), checkpoints=points, checkpoint_basis="completed")
        memory, experience = self.memory, _MemoryExperience(self.memory)
        recovery_error = None
        evaluation_contract = 3
        try:
            identity = {"agent": _agent_identity(self.agent),
                "injected_services": {name: _agent_identity(self.services.object(name))
                    for name in ("answer_llm", "native_judge") if self.services.object(name) is not None},
                "benchmark_judge": _agent_identity(getattr(self.config.benchmark, "judge", None)),
                "qa": {"type": f"{type(self.qa).__module__}.{type(self.qa).__qualname__}",
                       "steps": {name: _agent_identity(getattr(type(self.qa), name))
                           for name in ("__init__", "__call__", "retrieve", "retrieval_extensions", "messages", "prepare_episode")}}}
            # Only QA opting into the new execution/export behavior needs the
            # extended contract. Keep ordinary QA fingerprints byte-compatible
            # with pre-AWM v3 checkpoints (including MemRL and Mem0).
            if type(self.qa).collect_training_episode is not DefaultQA.collect_training_episode:
                identity["qa"]["steps"]["collect_training_episode"] = _agent_identity(
                    type(self.qa).collect_training_episode)
                evaluation_contract = 4
            if self.qa.include_episode_in_prediction:
                identity["qa"]["include_episode_in_prediction"] = True
                evaluation_contract = 4
        except (ValueError, TypeError, AttributeError) as exc:
            if source is not None:
                raise ValueError(f"Cannot verify recovery settings: {exc}") from exc
            recovery_error, identity = str(exc), {"unverifiable_run": uuid4().hex}
        fingerprint = json_sha256({"evaluation_contract": evaluation_contract, "protocol": self.config.protocol,
            "train": [case_fingerprint(c) for c in self.train], "test": [case_fingerprint(c) for c in self.tests],
            "reader": self.config.answer_model.to_dict() if self.config.answer_model else None,
            "metrics": self.config.metrics, "seed": self.config.seed,
            "execution": identity, "benchmark": self.benchmark.describe()})
        self.engine = _ExperienceRunner(plan, experience, directory, self._train_batch, self._test_batch,
            resume_from=source, execution_metadata={"case_resume": True, "protocol_fingerprint": fingerprint})
        try:
            report = await self.engine.run()
        except Exception as exc:
            report = self.engine.report
            self.failures.append({"stage": "experiment", "error_type": type(exc).__name__, "message": str(exc)})
        curve = tuple(row for row in self.engine._curve if "checkpoint" in row)
        metrics = {}
        if curve:
            self.predictions = [Prediction.from_dict(p) for p in curve[-1].get("predictions", [])]
            metrics = dict(curve[-1]["metrics"])
        resources = dict(report.get("resource_metrics", {}))
        resources.update(self.evaluator._operation_resources(memory.traces))
        resources.update(self.evaluator._response_resources(self.predictions))
        self.evaluator._add_coverage_resources(resources, self.predictions, len(self.tests))
        return EvaluationResult(memory.method_id, self.benchmark.benchmark_id, tuple(self.predictions),
            metrics, resources, failures=tuple(self.failures),
            run_metadata={**report.get("run_metadata", {}), "protocol_id": self.config.protocol,
                "recovery": {"supported": recovery_error is None, "reason": recovery_error},
                "progress": {name: report.get(name) for name in (
                    "attempted_exposures", "completed_exposures", "valid_feedback_exposures",
                    "learned_exposures", "failed_exposures", "pending_feedback")},
                "artifacts": report.get("artifacts", {}),
                "unreached_checkpoints": report.get("unreached_checkpoints", [])},
            traces=tuple(trace.to_dict() for trace in memory.traces.all()), learning_curve=curve)


def case_fingerprint(case):
    return {"case_id": case.case_id, "query": case.query, "scope": case.scope, "reference": case.reference,
            "history": [event.to_dict() for event in case.history],
            "episode": case.episode.to_dict() if case.episode else None, "metadata": dict(case.metadata)}


def _agent_identity(agent):
    """Hash executable code and configuration, excluding declared runtime state.

    An Agent may expose its existing ``config`` attribute to distinguish its
    settings from mutable counters, clients and execution logs. Otherwise we
    include function captures or the values of constructor-named attributes.
    """
    if agent is None:
        return "default_qa"
    active = set()
    def constant(value):
        if value is None or isinstance(value, (str, bool, int, float)):
            return value
        if isinstance(value, CodeType):
            return {"code": value.co_code.hex(), "names": value.co_names,
                    "constants": [constant(row) for row in value.co_consts]}
        if isinstance(value, bytes):
            return {"bytes": value.hex()}
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, type):
            return {"class": f"{value.__module__}.{value.__qualname__}"}
        if value is Ellipsis:
            return {"constant": "Ellipsis"}
        identity = id(value)
        if identity in active:
            raise ValueError("Cyclic Agent settings; expose serializable agent.config for recovery")
        active.add(identity)
        try:
            return structured(value)
        finally:
            active.remove(identity)

    def structured(value):
        if isinstance(value, Mapping):
            return {str(key): constant(row) for key, row in value.items()}
        if isinstance(value, (list, tuple, set, frozenset)):
            rows = [constant(row) for row in value]
            return sorted(rows, key=repr) if isinstance(value, (set, frozenset)) else rows
        if isinstance(value, partial):
            return {"callable": constant(value.func), "args": constant(value.args),
                    "keywords": constant(value.keywords)}
        if inspect.ismethod(value):
            return {"function": constant(value.__func__), "owner": settings(value.__self__)}
        if inspect.isfunction(value):
            result = {"module": value.__module__, "name": value.__qualname__,
                      "code": constant(value.__code__)}
            if hasattr(value, "config"):
                result["config"] = constant(value.config)
            else:
                result.update(defaults=constant(value.__defaults__), keywords=constant(value.__kwdefaults__),
                    captures={name: constant(cell.cell_contents) for name, cell in
                              zip(value.__code__.co_freevars, value.__closure__ or ())})
            return result
        if hasattr(value, "construction_spec"):
            return constant(value.construction_spec())
        if is_dataclass(value) and not callable(value):
            return constant(asdict(value))
        if hasattr(value, "model_dump"):
            return constant(value.model_dump(mode="json"))
        result = {"type": f"{type(value).__module__}.{type(value).__qualname__}",
                  "config": settings(value)}
        if callable(value):
            result["call"] = constant(type(value).__call__.__code__)
        return result

    def settings(value):
        if hasattr(value, "config"):
            return constant(value.config)
        try:
            parameters = inspect.signature(type(value).__init__).parameters
        except (TypeError, ValueError):
            parameters = {}
        names = [name for name, parameter in parameters.items() if name != "self"
                 and parameter.kind not in {parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD}]
        if names and all(hasattr(value, name) for name in names):
            return {name: constant(getattr(value, name)) for name in names}
        if hasattr(value, "__dict__") and not vars(value):
            return {}
        raise ValueError(f"Expose serializable agent.config for recovery: {type(value).__qualname__}")

    return json_sha256(constant(agent))

