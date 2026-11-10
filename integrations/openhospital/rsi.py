"""OpenHospital metrics and presentation on the common experience runner."""
import json
from pathlib import Path
import asyncio
from mematlas.core.provenance import json_sha256, file_sha256
from mematlas.core.files import atomic_write_json
from mematlas.evaluation.experience import _ExperienceRunner
from .metrics import learning_curve_metrics
from .curves import write_openhospital_curves

class _HospitalRunner(_ExperienceRunner):
    async def _resume_if_requested(self):
        import os
        folder = os.environ.get("MEMATLAS_ACTIVITY_DIR")
        if self.resume_from is not None and folder:
            from mematlas.evaluation.artifacts import recovery_checkpoint
            from mematlas.core.activity import activity
            position, _, payload = recovery_checkpoint(self.resume_from)
            boundary = 0
            index = len(self.plan.batches())
            for candidate, batch in enumerate(self.plan.batches(), 1):
                boundary += len(batch)
                complete = any(row.get("batch_index") == candidate and "efficiency" in row
                    for row in payload.get("curve", ())) and (Path(folder) / f"measurement-{candidate}.json").exists()
                if boundary > position or boundary == position and not complete:
                    index = candidate
                    break
            atomic_write_json(Path(folder) / "context.json", {"method": self.memory.core_memory.method_id,
                "round": index, "segment_id": self._segment_id, "attempt": int(os.environ.get("MEMATLAS_ATTEMPT", "1"))})
            self._recovery_activity = activity("round", "checkpoint_recovery")
            self._recovery_activity.__enter__()
        return await super()._resume_if_requested()

    async def _round_start(self, index):
        import os
        if os.environ.get("MEMATLAS_ACTIVITY_DIR"):
            if not hasattr(self, "_round_meter"):
                from .efficiency_metrics import RoundMeter
                self._round_meter = RoundMeter(self)
            await self._round_meter.start(index)

    async def _round_end(self, index):
        if hasattr(self, "_round_meter"):
            await self._round_meter.end(index)

    async def record_case_progress(self, completed_exposures):
        await super().record_case_progress(completed_exposures)
        if hasattr(self, "_round_meter"):
            await self._round_meter.case(completed_exposures)

    def _round_seal(self, index):
        if hasattr(self, "_round_meter"):
            self._round_meter.seal(index)

    def _round_complete(self, index):
        import os
        folder = os.environ.get("MEMATLAS_ACTIVITY_DIR")
        return not folder or (Path(folder) / f"measurement-{index}.json").is_file() and any(
            row.get("batch_index") == index and "efficiency" in row for row in self._curve)

    def _round_close(self):
        recovery = getattr(self, "_recovery_activity", None)
        if recovery is not None:
            recovery.__exit__(None, None, None)
            self._recovery_activity = None
        if hasattr(self, "_round_meter"):
            self._round_meter.close()

    async def _save_progress_snapshot(self, position):
        await super()._save_progress_snapshot(position)
        recovery = getattr(self, "_recovery_activity", None)
        if recovery is not None:
            recovery.__exit__(None, None, None)
            self._recovery_activity = None
        from mematlas.core.models import utc_now
        atomic_write_json(self.output_dir / "latest_snapshot.json", {
            "position": position, "saved_at": utc_now().isoformat(),
            "path": str(self.output_dir / "snapshots" / f"learned_{position}.json")})
        keep_recent = self.execution_metadata.get("snapshot_keep_recent")
        if keep_recent is None:
            return
        # Only prune this run, after verifying the newly committed snapshot.
        folder = self.output_dir / "snapshots"
        current = folder / f"learned_{position}.json"
        payload = json.loads(current.read_text(encoding="utf-8"))
        digest = payload.pop("payload_sha256")
        if not payload.get("committed") or json_sha256(payload) != digest:
            raise ValueError("Snapshot checksum failed; retention was not applied")
        del payload
        candidates = sorted((int(p.stem.split("_")[1]), p) for p in folder.glob("learned_*.json")
                            if p.stem.split("_")[1].isdigit())
        recent = {p for _, p in candidates[-keep_recent:]}
        every = self.execution_metadata.get("snapshot_keep_every", 50)
        retained = {path for count, path in candidates if not count or not count % every
                    or path in recent or count >= position}
        memory_files = set()
        for path in retained:
            state = json.loads(path.read_text(encoding="utf-8"))
            checksum = state.pop("payload_sha256", None)
            if not state.get("committed") or json_sha256(state) != checksum:
                raise ValueError("Retained snapshot checksum failed; retention was not applied")
            memory_path = (folder / state["memory_file"]).resolve()
            if memory_path.parent != folder.resolve() or file_sha256(memory_path) != state["memory_sha256"]:
                raise ValueError("Retained memory checksum failed; retention was not applied")
            memory_files.add(memory_path)
            del state
        for count, path in candidates:
            if path not in retained:
                path.unlink()
        # A snapshot references a separate full Memory checkpoint. Replaced
        # checkpoints at the same position can leave these files orphaned too.
        import re
        for path in folder.glob("memory-*.json"):
            if re.fullmatch(r"memory-[0-9a-f]{32}\.json", path.name) and path.resolve() not in memory_files:
                path.unlink()

    def _protocol_metadata(self):
        return {"id": "openhospital_memory_rsi", "department": self.memory.department,
                **self.plan.protocol_details()}

    def _write_curves(self):
        return write_openhospital_curves(self._curve, self.output_dir)

    def _summarize_curve(self):
        return learning_curve_metrics(self._curve)


class _HospitalEvaluation:
    """Connect the hospital scene to the common public experiment entry."""
    def __init__(self, plan, memory, execution, *, execution_metadata=None, preparation_usage=()):
        self.plan, self.memory, self.execution = plan, memory, execution
        self.execution_metadata = execution_metadata or {}
        self.preparation_usage = preparation_usage

    def _check_experiment(self, config):
        if not isinstance(config.benchmark, dict) or config.benchmark.get("kind") != "openhospital":
            raise ValueError("Hospital execution requires benchmark.kind=openhospital")
        if config.protocol != "experience_learning_curve" or len(config.methods) != 1:
            raise ValueError("OpenHospital uses one method and experience_learning_curve")
        if config.methods[0].method.method_id != self.memory.core_memory.method_id:
            raise ValueError("Hospital Memory differs from the experiment method")
        if config.checkpoints and tuple(sorted(set(config.checkpoints))) != self.plan.effective_checkpoints():
            raise ValueError("Hospital checkpoints differ from the prepared schedule")
        if config.train_limit is not None and config.train_limit < self.plan.training_count:
            raise ValueError("Set hospital training counts when preparing its scene schedule")
        if config.test_limit is not None and config.test_limit < len(self.plan.primary_test_ids):
            raise ValueError("Set hospital held-out counts when preparing its scene schedule")
        self.plan.batches()
        from mematlas.core.contracts import ProtocolRequirements
        from mematlas.core.capabilities import Operation
        ProtocolRequirements(operations=frozenset({Operation.ADD, Operation.EPISODE_END, Operation.SEARCH, Operation.SAVE, Operation.LOAD}),
                             frozen=True).validate(self.memory.core_memory.plugin.descriptor)

    async def _evaluate_experiment(self, runner, config, method, directory, source, services, evaluator):
        from mematlas.core.evaluation import EvaluationResult, Prediction
        from mematlas.core.models import MemoryContribution
        from mematlas.core.provenance import json_sha256

        def predictions_for(point):
            predictions = []
            for row in point.get("case_reports", ()):
                patient = row["patient_id"]
                executed = self.execution._find(
                    f"test_execution/{point['checkpoint']}-{point['split']}-{json_sha256(patient)}.json") if point.get("split") != "train" else None
                executed = executed or {}
                predictions.append(Prediction(patient, row.get("final_result", executed.get("final_result")), None,
                    applied_memory=tuple(MemoryContribution.from_dict(item)
                        for item in row.get("applied_memory", executed.get("applied_memory", ()))),
                    metadata={"department": row["department"], "clinical_report": row["report"],
                              "response_time_s": row.get("response_time_s")}))
            return tuple(predictions)

        async def test_batch(ids, memory, checkpoint, split):
            result = dict(await self.execution.test_batch(ids, memory, checkpoint, split))
            if config.metrics:
                predictions = predictions_for({**result, "checkpoint": checkpoint, "split": split})
                for metric in evaluator.metrics:
                    result[metric.name] = metric.compute(predictions)
            return result

        engine = _HospitalRunner(self.plan, self.memory, directory,
            self.execution.train_batch, test_batch,
            resume_from=source, execution_metadata={**self.execution_metadata,
                "protocol_fingerprint": json_sha256({
                    "evaluation_contract": 2, "scene": self.execution_metadata.get("protocol_fingerprint"),
                    "metrics": list(config.metrics)})},
            preparation_usage=self.preparation_usage)
        self.execution.on_case_completed = engine.record_case_progress
        live = getattr(self.execution, "progress", None)
        async def heartbeat():
            while True:
                await asyncio.sleep(3)
                from mematlas.core.activity import append
                append("heartbeat")
                live.write()
        task = asyncio.create_task(heartbeat()) if live else None
        if live:
            live.bridge = self.memory
            # Recover display counters once; heartbeat never scans result directories.
            for root in reversed(self.execution.sources):
                for path in (root / "encounters").glob("*.json"):
                    bundle = json.loads(path.read_text(encoding="utf-8"))
                    timing_path = root / "case_usage" / path.name
                    timing = json.loads(timing_path.read_text(encoding="utf-8")) if timing_path.is_file() else {}
                    timeline = bundle.get("case_state", {}).get("timeline", ())
                    live.training[path.stem] = {"report": bundle.get("evaluation_report", {}),
                        "response_time_s": timing.get("response_time_s"),
                        "position": timing.get("position", 0),
                        "json_recovery": {"repairs": sum(row.get("json_repair_count", 0) for row in timeline),
                            "fallbacks": sum(bool(row.get("json_fallback")) for row in timeline)}}
                for path in (root / "test_cases").glob("*.json"):
                    saved = json.loads(path.read_text(encoding="utf-8"))
                    live.tests[path.stem] = {"json_recovery": saved.get("json_recovery", {})}
                for path in sorted((root / "checkpoints").glob("learned_*.json")):
                    _, checkpoint, split = path.stem.split("_", 2)
                    live.record_checkpoint(int(checkpoint), split, json.loads(path.read_text(encoding="utf-8")))
        failures = []
        try:
            report = await engine.run()
        except Exception as exc:
            report = engine.report
            from mematlas.core.recovery import error_info
            failures.append({"stage": "hospital", "error_type": type(exc).__name__, **error_info(exc)})
        finally:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            if live:
                paused = failures and failures[-1]["error_type"] in {"HospitalPaused", "DoctorOutputPaused", "LLMRequestPaused"}
                live.write(status="paused" if paused else "failed" if failures else "completed",
                           failure={"type": failures[-1]["error_type"], "message": failures[-1]["message"][:800]} if failures else None)
        curve = tuple(row for row in engine._curve if "checkpoint" in row)
        primary = [row for row in curve if row.get("split") == "primary_test"]
        training = [row for row in engine._curve if "batch_index" in row]
        last = primary[-1] if primary else ({"split": "train", "case_reports": [case
            for row in training for case in row.get("case_reports", ())]} if self.plan.training_only and training else {})
        predictions = predictions_for(last)
        from .metrics import aggregate_openhospital_evaluations
        metrics = {**aggregate_openhospital_evaluations(last.get("case_reports", ())),
                   "per_department": last.get("per_department", {})} if last else {}
        if last and self.plan.training_only:
            records = last["case_reports"]
            metrics["per_department"] = {department: aggregate_openhospital_evaluations(
                [row for row in records if row["department"] == department])
                for department in sorted({row["department"] for row in records})}
        if last:
            metrics.update({name: last[name] for name in config.metrics if name in last})
        metadata = {**report.get("run_metadata", {}), "protocol_id": config.protocol,
            "metrics_split": "train" if self.plan.training_only else "primary_test",
            "progress": {key: report.get(key) for key in (
                "attempted_exposures", "completed_exposures", "valid_feedback_exposures",
                "learned_exposures", "failed_exposures", "pending_feedback", "scheduled_exposures")},
            "cost_summary": report.get("cost_summary"), "memory_reuse": report.get("memory_reuse"),
            "learning_curve_summary": report.get("learning_curve"),
            "artifacts": report.get("artifacts", {}), "unreached_checkpoints": report.get("unreached_checkpoints", [])}
        return EvaluationResult(method.method.method_id, "openhospital", tuple(predictions), metrics,
            {**report.get("resource_metrics", {}), **evaluator._response_resources(predictions)}, tuple(failures), metadata,
            tuple(trace.to_dict() for trace in self.memory.core_memory.traces.all()), curve)
