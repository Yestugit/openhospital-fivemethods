"""Frozen pre-AWM training execution for cross-version recovery tests.

Copied verbatim from pre-native-fix/evaluation/execution.py::_run_training.
Source SHA256: b05bea4da06d3abf2a5133a10e886b8c14ec5e06c987ed19255ad84e2b9112e0
Only the surrounding imports are adapted. Do not update with production fixes.
"""
from uuid import uuid4
from mematlas.core.evaluation import EvaluationResult, Prediction
from mematlas.core.provenance import json_sha256
from mematlas.evaluation.execution import (
    _ExperiencePlan, _MemoryExperience, _ExperienceRunner, _agent_identity, case_fingerprint,
)


async def _run_training(self, directory, source):
    train_ids = tuple(c.case_id for c in self.train)
    points = (len(train_ids),) if self.config.protocol == "experience_train_test" else tuple(sorted({
        *(self.config.checkpoints or (0, len(train_ids))), len(train_ids)}))
    plan = _ExperiencePlan(train_ids, tuple(c.case_id for c in self.tests), (),
        batch_count=len({p for p in points if p > 0}), checkpoints=points, checkpoint_basis="completed")
    memory, experience = self.memory, _MemoryExperience(self.memory)
    recovery_error = None
    try:
        identity = {"agent": _agent_identity(self.agent),
            "injected_services": {name: _agent_identity(self.services.object(name))
                for name in ("answer_llm", "native_judge") if self.services.object(name) is not None},
            "benchmark_judge": _agent_identity(getattr(self.config.benchmark, "judge", None)),
            "qa": {"type": f"{type(self.qa).__module__}.{type(self.qa).__qualname__}",
                   "steps": {name: _agent_identity(getattr(type(self.qa), name))
                       for name in ("__init__", "__call__", "retrieve", "retrieval_extensions", "messages", "prepare_episode")}}}
    except (ValueError, TypeError, AttributeError) as exc:
        if source is not None:
            raise ValueError(f"Cannot verify recovery settings: {exc}") from exc
        recovery_error, identity = str(exc), {"unverifiable_run": uuid4().hex}
    fingerprint = json_sha256({"evaluation_contract": 3, "protocol": self.config.protocol,
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
