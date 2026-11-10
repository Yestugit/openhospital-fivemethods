from __future__ import annotations
import inspect
import json
import platform
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Mapping, Sequence
from mematlas.runtime.memory import Memory
from mematlas.core.provenance import file_sha256, git_revision, json_sha256, package_version, redact_sensitive
from mematlas.core.telemetry import ProcessResourceMonitor, UsageLedger, UsageRecord, usage_scope, usage_context
from mematlas.evaluation.accounting import AccountingWindow, checkpoint_costs, cost_summary
from uuid import uuid4
from time import perf_counter
from typing import Protocol
from mematlas.core.files import atomic_write_json, storage_bytes
from mematlas.core.episodes import BatchResult
from mematlas.evaluation.metrics import Evaluator


class ExperienceMemory(Protocol):
    core_memory: Memory
    history: Any
    async def start(self) -> None: ...
    async def commit(self) -> None: ...
    async def snapshot_to(self, path: Path, *, learned_count: int) -> Path: ...
    async def restore_from(self, path: Path) -> int: ...
    async def state_fingerprint(self) -> str: ...
    async def close(self) -> None: ...

TrainBatch = Callable[[Sequence[str], ExperienceMemory, int], BatchResult | Mapping[str, Any] | Awaitable[BatchResult | Mapping[str, Any]]]
TestBatch = Callable[[Sequence[str], ExperienceMemory, int, str], Mapping[str, Any] | Awaitable[Mapping[str, Any]]]

def _partition_evenly(values: Sequence[str], count: int) -> list[tuple[str, ...]]:
    if count <= 0:
        raise ValueError("Batch count must be positive")
    if count > len(values):
        raise ValueError("Batch count cannot exceed case count")
    quotient, remainder = divmod(len(values), count)
    result = []
    start = 0
    for index in range(count):
        size = quotient + (1 if index < remainder else 0)
        result.append(tuple(values[start:start + size]))
        start += size
    return result


@dataclass(frozen=True)
class _ExperiencePlan:
    train_ids: tuple[str, ...]
    primary_test_ids: tuple[str, ...]
    advanced_test_ids: tuple[str, ...]
    batch_count: int = 1
    checkpoints: tuple[int, ...] = (0,)
    checkpoint_basis: str = field(default="scheduled", kw_only=True)
    failure_policy: str = field(default="stop", kw_only=True)
    snapshot_every: int = field(default=1, kw_only=True)
    training_only: bool = field(default=False, kw_only=True)

    def __post_init__(self):
        self._validate_splits()
        self._validate_checkpoints()
        if not isinstance(self.batch_count, int) or isinstance(self.batch_count, bool) or not 1 <= self.batch_count <= len(self.train_ids):
            raise ValueError("batch_count must be between 1 and the number of training cases")

    def _validate_splits(self) -> None:
        if self.checkpoint_basis not in {"scheduled", "completed", "learned", "attempted"}:
            raise ValueError("Invalid checkpoint basis")
        if self.failure_policy not in {"stop", "skip"}:
            raise ValueError("failure_policy must be stop or skip")
        if isinstance(self.snapshot_every, bool) or not isinstance(self.snapshot_every, int) or self.snapshot_every < 1:
            raise ValueError("snapshot_every must be a positive integer")
        groups = [self.train_ids, self.primary_test_ids, self.advanced_test_ids]
        if any(len(set(group)) != len(group) for group in groups):
            raise ValueError("Experience splits cannot contain duplicate case IDs")
        if any(set(groups[i]) & set(groups[j]) for i in range(3) for j in range(i + 1, 3)):
            raise ValueError("Training and held-out experience splits must be disjoint")
        if type(self.training_only) is not bool or self.training_only and (self.primary_test_ids or self.advanced_test_ids):
            raise ValueError("training_only requires empty held-out splits")
        if not self.primary_test_ids and not self.training_only:
            raise ValueError("Experience evaluation requires a nonempty held-out primary test split")

    def _validate_checkpoints(self) -> None:
        if any(not isinstance(point, int) or isinstance(point, bool) or not 0 <= point <= self.training_count
               for point in self.checkpoints):
            raise ValueError("Checkpoints must be integer training exposures within the plan")

    @property
    def training_count(self) -> int:
        return len(self.train_ids)

    def protocol_details(self) -> dict[str, Any]:
        return {}

    def batches(self) -> tuple[tuple[str, ...], ...]:
        boundaries = [0, *sorted({point for point in self.checkpoints if 0 < point < len(self.train_ids)}), len(self.train_ids)]
        segment_lengths = [boundaries[index + 1] - boundaries[index] for index in range(len(boundaries) - 1)]
        if len(segment_lengths) > self.batch_count:
            raise ValueError("batch_count must accommodate every requested checkpoint boundary")
        raw_allocations = [self.batch_count * length / len(self.train_ids) for length in segment_lengths]
        allocations = [max(1, int(value)) for value in raw_allocations]
        while sum(allocations) < self.batch_count:
            candidates = [i for i in range(len(allocations)) if allocations[i] < segment_lengths[i]]
            index = max(candidates, key=lambda i: raw_allocations[i] - allocations[i])
            allocations[index] += 1
        while sum(allocations) > self.batch_count:
            candidates = [i for i, value in enumerate(allocations) if value > 1]
            index = min(candidates, key=lambda i: raw_allocations[i] - allocations[i])
            allocations[index] -= 1
        batches: list[tuple[str, ...]] = []
        for start, end, allocation in zip(boundaries, boundaries[1:], allocations):
            batches.extend(_partition_evenly(self.train_ids[start:end], allocation))
        return tuple(batches)

    def effective_checkpoints(self) -> tuple[int, ...]:
        if self.training_only:
            return ()
        return tuple(sorted({*self.checkpoints, self.training_count}))


@dataclass
class _ExperienceRunner:
    plan: _ExperiencePlan
    memory: ExperienceMemory
    output_dir: Path
    train_batch: TrainBatch
    test_batch: TestBatch
    execution_metadata: Mapping[str, Any] = field(default_factory=dict)
    resume_from: Path | None = None
    prepare: Callable | None = None
    preparation_usage: Sequence[Mapping[str, Any]] = ()
    report: dict[str, Any] = field(default_factory=dict, init=False)
    _curve: list[dict[str, Any]] = field(default_factory=list, init=False)
    _position: int = field(default=0, init=False)
    _progress: dict = field(default_factory=lambda: {name: 0 for name in ("attempted_exposures",
        "completed_exposures", "valid_feedback_exposures", "learned_exposures", "failed_exposures", "pending_feedback")}, init=False)
    _segment_id: str = field(default_factory=lambda: uuid4().hex, init=False)
    _episode_baseline: set[str] = field(default_factory=set, init=False)

    def _protocol_metadata(self) -> dict[str, Any]:
        return {"id": "experience_learning_curve", **self.plan.protocol_details()}

    def plan_signature(self) -> str:
        memory = self.memory.core_memory
        storage = memory.plugin.descriptor.storage
        storage_keys = set(storage.get("path_parameters", {})) | set(storage.get("namespace_parameters", ()))
        return json_sha256({
            "train": self.plan.train_ids, "primary_test": self.plan.primary_test_ids,
            "advanced_test": self.plan.advanced_test_ids, "batch_count": self.plan.batch_count,
            "training_batches": self.plan.batches(),
            "checkpoints": self.plan.effective_checkpoints(), "protocol": self._protocol_metadata(),
            "training_only": self.plan.training_only,
            "checkpoint_basis": self.plan.checkpoint_basis, "failure_policy": self.plan.failure_policy,
            "input_fingerprint": self.execution_metadata.get("protocol_fingerprint"),
            "execution_contract": self.execution_metadata.get("resume_contract", {
                name: self.execution_metadata[name] for name in ("doctor", "services", "openhospital_sources")
                if name in self.execution_metadata
            }),
            "scope": dict(memory.config.scope),
            "providers": memory.config.to_dict()["providers"],
            "method_contract": memory.plugin.descriptor.to_dict(),
            "method_id": memory.method_id,
            "method_params": {key: value for key, value in memory.config.method.params.items() if key not in storage_keys},
        })

    def _write_curves(self) -> dict[str, str]:
        return {}

    def _summarize_curve(self) -> dict[str, Any]:
        return {"checkpoint_count": sum(row.get("split") == "primary_test" for row in self._curve)}

    async def _round_start(self, index):
        pass

    async def _round_end(self, index):
        pass

    def _round_seal(self, index):
        pass

    def _round_complete(self, index):
        return True

    def _round_close(self):
        pass

    async def run(self) -> Mapping[str, Any]:
        self.plan.batches()
        self.output_dir = Path(self.output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if (self.output_dir / "results.json").exists() and self.resume_from is None:
            raise FileExistsError("Experience output already exists; choose a new directory or explicitly resume")
        if (self.output_dir / "curve.json").exists() and self.resume_from is None:
            raise FileExistsError("Experience progress already exists; choose a new directory or explicitly resume")
        ledger = self.memory.core_memory.services.usage
        ledger.bind_journal(self.output_dir / "usage.jsonl", run_id=self.memory.core_memory.run_id)
        trace_path = self.output_dir / "memory_traces.jsonl"
        self.memory.core_memory.traces.bind_journal(trace_path)
        trace_path.touch(exist_ok=True)
        pricing_file = self.output_dir / "accounting.json"
        if self.resume_from is not None and pricing_file.exists():
            if json.loads(pricing_file.read_text(encoding="utf-8"))["pricing"] != ledger.pricing_spec():
                raise ValueError("Resume requires the original accounting currency and model pricing")
        atomic_write_json(pricing_file, {"schema_version": 1,
                          "pricing": ledger.pricing_spec(), "segment_id": self._segment_id})
        process_monitor = ProcessResourceMonitor()
        await process_monitor.start()
        process_resources: dict[str, Any] = {}
        position = 0
        checkpoints = set(self.plan.effective_checkpoints())
        started_at = datetime.now(timezone.utc)
        started = perf_counter()
        error: BaseException | None = None
        try:
            with usage_context(stage="preparation", segment_id=self._segment_id):
                if self.resume_from is None:
                    for row in self.preparation_usage:
                        ledger.add(UsageRecord(**dict(row)))
                await self.memory.start()
                if self.prepare is not None and self.resume_from is None:
                    result = self.prepare(self.memory)
                    if inspect.isawaitable(result):
                        await result
                    await self.memory.commit()
            self._episode_baseline = set(self.memory.history.results)
            with usage_context(stage="recovery", segment_id=self._segment_id):
                position = await self._resume_if_requested()
            self._position = position
            # A resumed output must itself be recoverable before another case
            # starts, including training-only runs with no checkpoint at zero.
            await self._save_progress_snapshot(position)
            self._write_progress()
            unsettled_boundary = self.execution_metadata.get("case_resume") and position > 0 and not any(
                row.get("scheduled_exposures") == position and "batch_index" in row for row in self._curve)
            checkpoint = self._checkpoint_value()
            if checkpoint in checkpoints and not unsettled_boundary and not self._checkpoint_complete(checkpoint):
                await self._run_checkpoint(checkpoint)
            batch_start = 0
            for batch_index, case_ids in enumerate(self.plan.batches(), start=1):
                batch_end = batch_start + len(case_ids)
                batch_recorded = any(row.get("batch_index") == batch_index for row in self._curve)
                if batch_end <= position and ((batch_recorded and self._round_complete(batch_index)) or not self.execution_metadata.get("case_resume")):
                    batch_start = batch_end
                    continue
                if batch_start != position and not (self.execution_metadata.get("case_resume") and batch_start < position <= batch_end):
                    raise ValueError(
                        "Resume points must align with a completed training batch; "
                        f"got learned_count={position}, batch={batch_index} spans {batch_start}:{batch_end}"
                    )
                await self._round_start(batch_index)
                window = AccountingWindow(self.memory.core_memory)
                previous_episode_ids = set(self.memory.history.results)
                with usage_scope(ledger, "agent"), usage_context(stage="training", segment_id=self._segment_id):
                    result = self.train_batch(case_ids, self.memory, batch_index)
                    if inspect.isawaitable(result):
                        result = await result
                    batch_result = BatchResult.coerce(result)
                    # Method-specific learning/evolution boundaries belong to
                    # the training callback and its method usage contract.
                    await self.memory.commit()
                batch = BatchResult.coerce(result)
                increments = batch.progress(len(case_ids))
                for episode in batch.episodes:
                    self.memory.history.record_result(episode)
                if not batch.episodes and set(self.memory.history.results) == previous_episode_ids:
                    for name, count in increments.items():
                        previous = self._progress[name]
                        self._progress[name] = previous + count if previous is not None and count is not None else None
                failed = increments["failed_exposures"] or 0
                position = self._position if failed and self.plan.failure_policy == "stop" else batch_end
                point = {
                    **dict(batch.metrics),
                    **self._counts(),
                    "episode_results": [row.to_dict() for row in batch.episodes],
                    **window.finish(),
                    **checkpoint_costs(ledger),
                    "batch_index": batch_index,
                    "scheduled_exposures": batch_end,
                    "checkpoint_basis": self.plan.checkpoint_basis,
                    "case_count": len(case_ids),
                }
                self._curve = [row for row in self._curve if row.get("batch_index") != batch_index]
                self._curve.append(point)
                self._position = position
                if failed and self.plan.failure_policy == "stop":
                    raise RuntimeError("Training batch reported failed episodes; progress retained without counting them as learned")
                with usage_context(stage="measurement", segment_id=self._segment_id):
                    await self._save_progress_snapshot(position)
                await self._round_end(batch_index)
                # Commit round metrics with the same case/state checkpoint.
                await self._save_progress_snapshot(position)
                self._round_seal(batch_index)
                checkpoint = self._checkpoint_value()
                if checkpoint in checkpoints:
                    await self._run_checkpoint(checkpoint)
                self._write_progress()
                batch_start = batch_end
            final_checkpoint = self._checkpoint_value()
            if final_checkpoint is None:
                if self.plan.checkpoint_basis != "scheduled":
                    raise ValueError("Checkpoint basis is unknown; return structured episode results or explicit counts")
            elif not self.plan.training_only and not self._checkpoint_complete(final_checkpoint):
                await self._run_checkpoint(final_checkpoint)
        except BaseException as exc:
            error = exc
            # Preserve the original failure location before teardown/reporting
            # can allocate another large recovery object or mask the exception.
            import traceback
            try:
                with (self.output_dir / "failure-traceback.log").open("a", encoding="utf-8") as stream:
                    traceback.print_exception(type(exc), exc, exc.__traceback__, file=stream)
            except OSError:
                pass
        finally:
            try:
                with usage_context(stage="teardown", segment_id=self._segment_id):
                    await self.memory.close()
            except BaseException as exc:
                error = error or exc
            finally:
                self._round_close()
                process_resources = await process_monitor.stop()
                self._write_progress()
        curve_artifacts = self._write_curves()
        artifacts = {
            **curve_artifacts,
            "memory_traces_jsonl": str(trace_path),
            "usage_jsonl": str(self.output_dir / "usage.jsonl"),
        }
        completed_at = datetime.now(timezone.utc)
        report = {
            "status": "failed" if error else "completed",
            "error": {"type": type(error).__name__, "message": str(error)} if error else None,
            **self._counts(),
            "scheduled_exposures": self._position,
            "checkpoint_basis": self.plan.checkpoint_basis,
            "unreached_checkpoints": sorted(checkpoints - {row["checkpoint"] for row in self._curve if "checkpoint" in row}),
            "segment_id": self._segment_id,
            "method_id": self.memory.core_memory.config.method.method_id,
            "train_count": len(self.plan.train_ids),
            "train_exposure_count": self.plan.training_count,
            "primary_test_count": len(self.plan.primary_test_ids),
            "advanced_test_count": len(self.plan.advanced_test_ids),
            "batches": self._curve,
            "learning_curve": self._summarize_curve(),
            "cost_summary": cost_summary(ledger, segment_id=self._segment_id),
            "resource_metrics": {
                **{f"common.{key}": value for key, value in process_resources.items()},
                **ledger.resource_metrics(),
                **Evaluator._operation_resources(self.memory.core_memory.traces),
                "current_process_usage": ledger.summary(segment_id=self._segment_id),
                "common.run_wall_time_ms": (perf_counter() - started) * 1000,
                "common.process_memory_scope": "current_process_and_descendants_excludes_external_services",
                **Evaluator._storage_resources(getattr(self.memory.core_memory._method, "storage_bytes", None)
                if self.memory.core_memory.config.runtime.executor == "service" else storage_bytes(
                    tuple(self.memory.core_memory.services.resource_paths)
                    or (self.memory.core_memory.config.runtime.artifact_dir,)
                )),
            },
            "artifacts": artifacts,
            "run_metadata": self._run_metadata(
                started_at,
                completed_at,
                trace_path=trace_path,
            ),
        }
        if self._position:
            preparation = report["cost_summary"]["preparation"]["cost"]
            acquisition = report["cost_summary"]["acquisition"]["cost"]
            report["cost_summary"]["amortized_preparation_cost_per_exposure"] = preparation / self._position if preparation is not None else None
            report["cost_summary"]["acquisition_cost_per_exposure"] = acquisition / self._position if acquisition is not None else None
        if hasattr(self.memory, "reuse_statistics"):
            report["memory_reuse"] = self.memory.reuse_statistics()
        self.report = report
        if error is not None:
            raise error
        return report

    def _counts(self):
        history = self.memory.history
        tracked = history.counts(exclude=self._episode_baseline)
        return {name: tracked[name] + value if value is not None else None for name, value in self._progress.items()}

    def _checkpoint_value(self):
        if self.plan.checkpoint_basis == "scheduled":
            return self._position
        return self._counts().get(self.plan.checkpoint_basis + "_exposures")

    def _cumulative_usage(self) -> UsageLedger:
        return self.memory.core_memory.services.usage

    async def record_case_progress(self, completed_exposures: int) -> None:
        """Seal a case boundary after execution/feedback state is durable."""
        if not self.execution_metadata.get("case_resume") or completed_exposures != self._position + 1:
            raise ValueError("Case progress must be contiguous and explicitly enabled")
        self._position = completed_exposures
        with usage_context(stage="measurement", segment_id=self._segment_id):
            if completed_exposures % self.plan.snapshot_every == 0:
                await self._save_progress_snapshot(completed_exposures)
        self._write_progress()

    async def _save_progress_snapshot(self, position: int) -> None:
        snapshot = self.output_dir / "snapshots" / f"learned_{position}.json"
        pending = snapshot.with_suffix(".pending")
        await self.memory.snapshot_to(pending, learned_count=position)
        payload = json.loads(pending.read_text(encoding="utf-8"))
        payload.update(schema_version=6, committed=True, schedule_position=position, progress=self._counts(),
                       untracked_progress=self._progress,
                       episode_baseline=sorted(self._episode_baseline),
                       checkpoint_basis=self.plan.checkpoint_basis, plan_signature=self.plan_signature(), curve=self._curve,
                       accounting_cursor=self.memory.core_memory.services.usage.cursor(),
                       accounting=checkpoint_costs(self.memory.core_memory.services.usage))
        payload["payload_sha256"] = json_sha256(payload)
        atomic_write_json(snapshot, payload)
        pending.unlink()

    def _run_metadata(
        self,
        started_at: datetime,
        completed_at: datetime,
        *,
        trace_path: Path,
    ) -> dict[str, Any]:
        memory = self.memory.core_memory
        descriptor = memory.plugin.descriptor
        method_id = descriptor.method_id
        source_root = memory.services.source_root(method_id)
        framework_root = Path(__file__).resolve().parents[1]
        if source_root is None and descriptor.source_subdir:
            candidate = Path(__file__).resolve().parents[3] / "agent-memory-methods" / descriptor.source_subdir
            source_root = candidate.resolve() if candidate.exists() else None
        split_ids = {
            "train": self.plan.train_ids,
            "primary_test": self.plan.primary_test_ids,
            "advanced_test": self.plan.advanced_test_ids,
        }
        return {
            "run_id": memory.run_id,
            "started_at": started_at.isoformat(),
            "completed_at": completed_at.isoformat(),
            "config": redact_sensitive(memory.config.to_dict()),
            "method": descriptor.to_dict(),
            "protocol": {
                **self._protocol_metadata(),
                "checkpoint_basis": self.plan.checkpoint_basis,
                "failure_policy": self.plan.failure_policy,
                "snapshot_every": self.plan.snapshot_every,
                "batch_count": self.plan.batch_count,
                "checkpoints": list(self.plan.effective_checkpoints()),
                "split_counts": {name: len(ids) for name, ids in split_ids.items()},
                "train_exposure_count": self.plan.training_count,
                "training_batches_sha256": json_sha256(self.plan.batches()),
                "split_id_sha256": {name: json_sha256(list(ids)) for name, ids in split_ids.items()},
            },
            "framework": {
                "version": package_version(),
                "git_revision": git_revision(framework_root),
            },
            "method_source": {
                "path": str(source_root) if source_root else None,
                "git_revision": git_revision(source_root) if source_root else None,
            },
            "artifacts": {
                "memory_traces": {
                    "path": str(trace_path),
                    "sha256": file_sha256(trace_path),
                },
            },
            "execution": redact_sensitive(dict(self.execution_metadata)),
            "resume_from": str(self.resume_from) if self.resume_from else None,
            "python": sys.version,
            "platform": platform.platform(),
        }

    async def _resume_if_requested(self) -> int:
        if self.resume_from is None:
            return 0
        source = Path(self.resume_from).expanduser().resolve()
        curve_path = source / "curve.json"
        snapshot_dir = source / "snapshots"
        if not curve_path.is_file() or not snapshot_dir.is_dir():
            raise ValueError(f"Resume source must contain curve.json and snapshots/: {source}")
        from mematlas.evaluation.artifacts import recovery_checkpoint
        position, snapshot, payload = recovery_checkpoint(source, expected_signature=self.plan_signature())
        previous_trace = source / "memory_traces.jsonl"
        if previous_trace.exists():
            self.memory.core_memory.traces.import_journal(previous_trace)
        ledger = self.memory.core_memory.services.usage
        pricing_file = source / "accounting.json"
        if source != self.output_dir and pricing_file.exists() and json.loads(pricing_file.read_text(encoding="utf-8"))["pricing"] != ledger.pricing_spec():
            raise ValueError("Resume requires the original accounting currency and model pricing")
        batch_boundaries = {0}
        total = 0
        for batch in self.plan.batches():
            total += len(batch)
            batch_boundaries.add(total)
        if position not in batch_boundaries and not self.execution_metadata.get("case_resume"):
            raise ValueError(f"Snapshot learned_count {position} does not align with this RSI plan")
        raw_curve = payload.get("curve", json.loads(curve_path.read_text(encoding="utf-8")))
        if not isinstance(raw_curve, list):
            raise ValueError(f"Resume curve must be a JSON list: {curve_path}")
        self._progress = dict(payload["untracked_progress"])
        self._episode_baseline = set(payload["episode_baseline"])
        self._curve = [
            dict(point)
            for point in raw_curve
            if (
                (isinstance(point.get("scheduled_exposures"), int)
                 and point["scheduled_exposures"] <= position)
                or (isinstance(point.get("checkpoint"), int) and point["checkpoint"] <= position)
            )
        ]
        restored = await self.memory.restore_from(snapshot)
        if restored != position:
            raise RuntimeError(f"Snapshot restored {restored} cases, expected {position}")
        if not (source / "usage.jsonl").is_file():
            raise ValueError("Resume source must contain usage.jsonl")
        ledger.import_journal(source / "usage.jsonl")
        resume_payload = {
            "source_run": str(source),
            "snapshot": str(snapshot),
            "schedule_position": position,
            "resumed_at": datetime.now(timezone.utc).isoformat(),
        }
        (self.output_dir / "resume.json").write_text(
            json.dumps(resume_payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        self._write_progress()
        return position

    def _checkpoint_splits(self, position: int):
        if self.plan.training_only:
            return []
        values = [("primary_test", self.plan.primary_test_ids)]
        if self._position == self.plan.training_count and self.plan.advanced_test_ids:
            values.append(("advanced_test", self.plan.advanced_test_ids))
        return values

    def _checkpoint_complete(self, position: int) -> bool:
        completed = {point.get("split") for point in self._curve if point.get("checkpoint") == position}
        return all(split in completed for split, _ in self._checkpoint_splits(position))

    async def _run_checkpoint(self, position: int) -> None:
        ledger = self.memory.core_memory.services.usage
        with usage_context(stage="measurement", segment_id=self._segment_id):
            await self._save_progress_snapshot(self._position)
        self._write_progress()
        for split, case_ids in self._checkpoint_splits(position):
            if any(point.get("checkpoint") == position and point.get("split") == split for point in self._curve):
                continue
            costs_before_test = checkpoint_costs(ledger)
            window = AccountingWindow(self.memory.core_memory)
            with usage_context(stage="evaluation", segment_id=self._segment_id):
                await self.memory.commit()
                state_before = await self.memory.state_fingerprint()
                async with self.memory.core_memory.read_only():
                    with usage_scope(ledger, "agent"):
                        result = self.test_batch(case_ids, self.memory, position, split)
                        if inspect.isawaitable(result):
                            result = await result
                if await self.memory.state_fingerprint() != state_before:
                    raise RuntimeError(f"{split} mutated persistent memory at checkpoint {position}")
            self._curve.append({**dict(result), **window.finish(), **costs_before_test,
                                "checkpoint": position, "split": split,
                                **self._counts(), "scheduled_exposures": self._position,
                                "checkpoint_basis": self.plan.checkpoint_basis,
                                "frozen_verified":True, "state_fingerprint":state_before,
                                "cumulative_experiment_cost_after_test": ledger.summary()["cost"]})
            with usage_context(stage="measurement", segment_id=self._segment_id):
                await self._save_progress_snapshot(self._position)
            self._write_progress()

    def _write_progress(self) -> None:
        atomic_write_json(self.output_dir / "curve.json", self._curve)
        atomic_write_json(self.output_dir / "usage_records.json", self._cumulative_usage().to_dicts())
        self._write_curves()
