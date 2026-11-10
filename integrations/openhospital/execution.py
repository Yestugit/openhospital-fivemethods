"""Ordered training, bounded parallel frozen tests, durable scoring and recovery."""
from pathlib import Path
import asyncio
import json
import os
from time import perf_counter

from mematlas.core.files import atomic_write_json
from mematlas.core.models import OperationTrace, utc_now
from mematlas.core.provenance import json_sha256
from mematlas.core.telemetry import usage_context, current_usage_context
from .accounting import HospitalUsageCollector, doctor_usage_observer, hospital_window_metrics
from .metrics import aggregate_openhospital_evaluations
from mematlas.core.episodes import BatchResult
from .progress import HospitalProgress


def _sdk_output_path(directory):
    """Allow SDK prompt/event files below long Windows experiment paths."""
    if os.name != "nt":
        return str(directory)
    absolute = str(Path(directory).resolve())
    if absolute.startswith("\\\\?\\"):
        return absolute
    if absolute.startswith("\\\\"):
        return "\\\\?\\UNC\\" + absolute[2:]
    return "\\\\?\\" + absolute


class HospitalExecution:
    def __init__(self, *, plan, output_dir, source_dir, agent_class, doctor_config, runtime_builder,
                 service_urls, team_id):
        self.plan, self.output = plan, Path(output_dir)
        self.sources = [self.output] + ([Path(source_dir)] if source_dir else [])
        self.agent_class, self.doctor_config = agent_class, doctor_config
        self.runtime_builder, self.urls, self.team_id = runtime_builder, service_urls, team_id
        self.on_case_completed = None
        self._collectors = {}
        self._ledger = None
        self._service_usage_locks = {}
        self.progress = HospitalProgress(self.output, plan) if plan is not None else None

    def _bind_metered(self, agent, bridge):
        self._ledger = bridge.core_memory.services.usage
        def bind(context):
            result = agent.bind_context(context)
            result.llm.model = os.environ.get("DOCTOR_MODEL", result.llm.model)
            result.llm.base_url = os.environ.get("DOCTOR_MODEL_BASE_URL", result.llm.base_url)
            result.llm.usage_observer = doctor_usage_observer(self._ledger)
            result.llm.request_context_provider = current_usage_context
            for client in vars(context.clients).values():
                if hasattr(client, "http"):
                    client.http.request_context_provider = current_usage_context
            return result
        return bind

    async def _service_usage(self, url):
        # Collector cursors are shared across cases. Never import the same
        # provider records twice when concurrent patients finish together.
        lock = self._service_usage_locks.setdefault(url, asyncio.Lock())
        async with lock:
            if url not in self._collectors:
                self._collectors[url] = HospitalUsageCollector(url, self.team_id, self._ledger)
            await asyncio.to_thread(self._collectors[url].sync)

    def _scoring_agent(self, bridge, patient, mode, split, directory):
        from hospital_agent_sdk.runtime import AgentContext
        from hospital_agent_sdk.actions import DoctorActions
        from hospital_agent_sdk.client import build_service_clients
        from hospital_agent_sdk.event_logger import EventLogger
        from hospital_agent_sdk.llm import LLMCaller
        config, runtime = self.runtime_builder(self.doctor_config, patient_ids=[patient],
            service_url=self.urls[split], mode=mode, team_id=self.team_id)
        clients = build_service_clients(base_url=runtime.service_base_url, token=runtime.service_token,
            model_api_key=runtime.model_api_key, team_id=self.team_id, mode=mode, send_model_api_key=True)
        logger = EventLogger(output_dir=_sdk_output_path(directory))
        actions = DoctorActions(patient_client=clients.patient_client, exam_client=clients.exam_client,
            evaluate_client=clients.evaluate_client, event_logger=logger, team_id=self.team_id)
        llm = LLMCaller(api_key=runtime.model_api_key, mode=mode, team_id=self.team_id)
        agent = self.agent_class(config=config, memory=bridge)
        self._bind_metered(agent, bridge)(AgentContext(mode, actions, logger, clients, llm, runtime))
        return agent

    async def _complete_feedback(self, bundle, bridge, occurrence=None):
        """Re-score an executed case without repeating its diagnosis or tool actions."""
        import importlib
        patient = bundle["patient_id"]
        occurrence = occurrence or bridge._occurrence(patient)
        journal = self._find(f"encounters/{json_sha256(occurrence)}.json")
        if journal and journal.get("final_result") == bundle["final_result"]:
            bundle = journal
        try:
            bridge.feedback_policy.evaluate(bundle.get("evaluation_report", {}))
            score = bundle["evaluation_report"]
        except ValueError:
            score = None
        if score is not None and not bundle.get("needs_reflection"):
            return bundle
        agent = self._scoring_agent(bridge, patient, "train", "train",
                    self.output / "regraded" / json_sha256(occurrence))
        actions = agent.actions
        await self._service_usage(self.urls["train"])
        try:
            with usage_context(case_id=occurrence, operation="clinical_feedback"):
                if score is None:
                    score = await actions.evaluation(patient, final_result=bundle["final_result"])
                    bridge.feedback_policy.evaluate(score)
                    atomic_write_json(bridge.journal_dir / f"{json_sha256(occurrence)}.json",
                                      {**bundle, "evaluation_report":score, "needs_reflection":True})
                prompts = importlib.import_module(self.agent_class.__module__.rsplit(".",1)[0] + ".prompt")
                reflection = await agent._call_llm(prompt=prompts.format_prompt(prompts.EVALUATION_REFLECTION_PROMPT,
                    {"chat_history":bundle["case_state"].get("chat_history", []), "evaluation_details":agent._evaluation_details(score)}),
                    default={}, prompt_name="evaluation_reflection", patient_id=patient)
            bundle = {**bundle, "evaluation_report":score, "reflection":reflection}
            bundle.pop("needs_reflection", None)
            atomic_write_json(bridge.journal_dir / f"{json_sha256(occurrence)}.json", bundle)
            return bundle
        finally:
            await self._service_usage(self.urls["train"])

    def _find(self, relative):
        for root in self.sources:
            path = root / relative
            if path.is_file():
                return json.loads(path.read_text(encoding="utf-8"))
        return None

    async def train_batch(self, ids, bridge, batch_index):
        from hospital_agent_sdk import run_train_async
        if self.progress:
            self.progress.check_limits()
        self._ledger = bridge.core_memory.services.usage
        cursor = self._ledger.cursor()
        bridge.feedback_resolver = self._complete_feedback
        bridge.configure_training_batch(ids, batch_index=batch_index,
            end_of_training=batch_index == len(self.plan.batches()))
        records = []
        offset = sum(len(batch) for batch in self.plan.batches()[:batch_index - 1])
        for index, patient in enumerate(ids, start=1):
            occurrence = bridge._occurrence(patient)
            if self.progress:
                self.progress.bridge = bridge
                self.progress.write(status="running", phase="train", patient_id=patient,
                                    department=bridge._department(patient), batch=batch_index, position=offset + index)
            key = json_sha256(occurrence)
            bundle = self._find(f"encounters/{key}.json")
            status = bridge.history.result(occurrence, scope=bridge.scope)
            if status.execution != "completed" or (
                    status.learning != "submitted" and status.episode_id not in bridge.history.pending_feedback):
                if bundle is not None:
                    try:
                        bridge.feedback_policy.evaluate(bundle["evaluation_report"])
                    except ValueError:
                        bundle = await self._complete_feedback(bundle, bridge)
                    if bundle.get("needs_reflection"):
                        bundle = await self._complete_feedback(bundle, bridge)
                    await bridge.learn_case(**bundle)
                else:
                    config, runtime = self.runtime_builder(self.doctor_config, patient_ids=[patient],
                        service_url=self.urls["train"], mode="train", team_id=self.team_id)
                    from dataclasses import replace
                    runtime = replace(runtime, output_dir=_sdk_output_path(runtime.output_dir))
                    agent = self.agent_class(config=config, memory=bridge)
                    await self._service_usage(self.urls["train"])
                    before = bridge.core_memory.services.usage.summary()
                    case_started = perf_counter()
                    try:
                        from mematlas.core.activity import activity
                        with activity("clinical", "clinical_sdk", case_id=occurrence), usage_context(case_id=occurrence):
                            summary = await run_train_async(agent_factory=self._bind_metered(agent, bridge), config=runtime)
                    finally:
                        await self._service_usage(self.urls["train"])
                    if summary["train_count"] != 1 or not all(c["finished"] for c in summary["cases"]):
                        raise RuntimeError("Hospital SDK did not complete the scheduled patient")
                    bundle = self._find(f"encounters/{key}.json")
                    if bundle is None:
                        raise RuntimeError("Doctor returned without recording clinical feedback")
                    after = bridge.core_memory.services.usage.summary()
                    atomic_write_json(self.output / "case_usage" / f"{key}.json", {
                        "tokens": (after["known_total_tokens"] - before["known_total_tokens"]),
                        "official_output_dir": summary["output_dir"],
                        "patient_id": patient, "occurrence": occurrence,
                        "position": offset + index, "response_time_s": perf_counter() - case_started,
                    })
                if self.on_case_completed:
                    await self.on_case_completed(offset + index)
            if bundle is None:
                raise RuntimeError("Completed case has no execution journal")
            records.append({"patient_id": patient, "department": bridge._department(patient),
                            "report": bundle["evaluation_report"] or {"evaluationStatus":"pending"},
                            "final_result": bundle["final_result"], "applied_memory": bundle.get("applied_memory", []),
                            "json_recovery": {"fallbacks": sum(bool(row.get("json_fallback")) for row in bundle["case_state"].get("timeline", ())),
                                "repairs": sum(row.get("json_repair_count", 0) for row in bundle["case_state"].get("timeline", ()))}})
            if self.progress:
                timing = self._find(f"case_usage/{key}.json") or {}
                records[-1]["response_time_s"] = timing.get("response_time_s")
                records[-1]["position"] = offset + index
                self.progress.training[json_sha256(occurrence)] = records[-1]
                self.progress.write()
                self.progress.check_stop(offset + index)
        boundaries = getattr(self.plan, "feedback_boundaries", None)
        await bridge.deliver_feedback(batch_end=boundaries is None or offset + len(ids) in boundaries)
        for row in records:
            bundle = self._find(f"encounters/{json_sha256(bridge._occurrence(row['patient_id']))}.json")
            row["report"] = bundle["evaluation_report"] or {"evaluationStatus":"pending"}
        return BatchResult(tuple(bridge.history.result(bridge._occurrence(patient), scope=bridge.scope) for patient in ids),
            {**self._metrics(records, bridge), **hospital_window_metrics(self._ledger, cursor)})

    async def test_batch(self, ids, bridge, learned, split):
        from hospital_agent_sdk import run_test_async
        batch_started = perf_counter()
        if self.progress:
            self.progress.check_limits()
        await bridge.start()
        self._ledger = bridge.core_memory.services.usage
        cursor = self._ledger.cursor()
        frozen_state = await bridge.state_fingerprint()
        ids = tuple(ids)
        if len(ids) != len(set(ids)):
            raise ValueError("Frozen test patients must be distinct within a checkpoint")
        slots = {patient: index for index, patient in enumerate(ids)}
        concurrency = int(os.environ.get("OPH_TEST_CONCURRENCY", "1"))
        if not 1 <= concurrency <= 16:
            raise ValueError("OPH_TEST_CONCURRENCY must be between 1 and 16")
        if concurrency > 1 and (bridge.core_memory.method_id != "mem0" or
                                bridge.core_memory.capabilities.read_only_mode not in {"native", "snapshot"}):
            raise ValueError("Parallel hospital tests require Mem0 protected read-only retrieval")
        rows_by_patient, active = {}, {}
        queue = iter(ids)
        stopping = False
        errors = []

        def show_progress():
            if self.progress:
                self.progress.bridge = bridge
                patients = list(active)
                current = patients[0] if patients else None
                self.progress.write(status="running", phase=split, patient_id=current,
                    department=bridge._department(current) if current else None,
                    checkpoint=learned, test_concurrency=concurrency,
                    active_test_patients=patients, active_test_count=len(patients),
                    test_batch_completed=len(rows_by_patient), test_batch_total=len(ids))

        async def run_patient(patient):
            # Agents, SDK logs and provenance are separate for every patient.
            # The common Memory facade serializes retrieval; only stateless
            # doctor/patient/exam/judge requests overlap. No training can run here.
            active[patient] = True
            show_progress()
            relative = f"test_cases/{learned}-{split}-{json_sha256(patient)}.json"
            execution_relative = f"test_execution/{learned}-{split}-{json_sha256(patient)}.json"
            saved = self._find(relative)
            if saved is not None and saved.get("frozen_state_fingerprint") != frozen_state:
                saved = None
            if saved is None:
                config, runtime = self.runtime_builder(self.doctor_config, patient_ids=[patient],
                    service_url=self.urls[split], mode="test", team_id=self.team_id)
                if runtime is not None:
                    from dataclasses import replace
                    sdk_directory = Path(runtime.output_dir)
                    if concurrency > 1:
                        # The complete patient identity remains in the execution
                        # journal; a roster slot isolates logs without a 64-char hash.
                        sdk_directory /= f"p{learned}-{split[0]}-{slots[patient]:03d}"
                    runtime = replace(runtime, output_dir=_sdk_output_path(sdk_directory))
                agent = self.agent_class(config=config, memory=bridge)
                await self._service_usage(self.urls[split])
                try:
                    executed = self._find(execution_relative)
                    if executed is not None and executed.get("frozen_state_fingerprint") != frozen_state:
                        executed = None
                    if executed is None:
                        bridge._begin_test_case(patient)
                        started = perf_counter()
                        async with bridge.frozen_operation(frozen_state):
                            with usage_context(case_id=f"{learned}:{split}:{patient}"):
                                summary = await run_test_async(agent_factory=self._bind_metered(agent, bridge), config=runtime)
                        response_time = perf_counter() - started
                        if len(summary["final_results"]) != 1 or not all(c["finished"] for c in summary["cases"]):
                            raise RuntimeError("Hospital test did not complete the scheduled patient")
                        executed = {"final_result":summary["final_results"][0], "output_dir":summary["output_dir"],
                                    "response_time_s": response_time,
                                    "applied_memory": bridge._test_contributions(patient),
                                    "frozen_state_fingerprint": frozen_state,
                                    "json_recovery": {"fallbacks": sum(bool(row.get("json_fallback")) for row in getattr(agent, "_timeline", ())),
                                        "repairs": sum(row.get("json_repair_count", 0) for row in getattr(agent, "_timeline", ()))}}
                        atomic_write_json(self.output / execution_relative, executed)
                    else:
                        agent = self._scoring_agent(bridge, patient, "test", split,
                            self.output / "regraded" / f"{learned}-{split}-{json_sha256(patient)}")
                    final = executed["final_result"]
                    scoring_at, scoring_started, scored = utc_now(), perf_counter(), False
                    try:
                        async with bridge.frozen_operation(frozen_state):
                            with usage_context(case_id=f"{learned}:{split}:{patient}", operation="native_scoring"):
                                score = await agent.actions.evaluation(patient, final_result=final)
                        # Failed scoring never seals a held-out result.
                        bridge.feedback_policy.evaluate(score)
                        scored = True
                    finally:
                        bridge.core_memory.traces.add(OperationTrace(bridge.core_memory.method_id,
                            "judge_scoring", scoring_at, (perf_counter() - scoring_started) * 1000, scored,
                            metadata={"stage": "evaluation", "phase": "judge", "case_id": patient,
                                      "checkpoint": learned}))
                    saved = {"patient_id": patient, "department": bridge._department(patient), "report": score,
                             "final_result": final, "applied_memory": executed.get("applied_memory", []),
                             "frozen_state_fingerprint": frozen_state,
                             "official_output_dir": executed["output_dir"],
                             "response_time_s": executed.get("response_time_s"),
                             "json_recovery": executed.get("json_recovery", {})}
                    atomic_write_json(self.output / relative, saved)
                finally:
                    await self._service_usage(self.urls[split])
            return saved

        async def worker():
            nonlocal stopping
            while not stopping:
                try:
                    if self.progress:
                        self.progress.check_stop(0)
                    patient = next(queue)
                except StopIteration:
                    return
                except Exception as exc:
                    errors.append(exc)
                    stopping = True
                    return
                try:
                    saved = await run_patient(patient)
                    rows_by_patient[patient] = saved
                    if self.progress:
                        self.progress.tests[f"{learned}-{split}-{json_sha256(patient)}"] = saved
                        self.progress.last_test = {"split": split, "checkpoint": learned,
                            **self.progress.metrics([rows_by_patient[p] for p in ids if p in rows_by_patient])}
                        self.progress.check_stop(0)
                except Exception as exc:
                    # Drain in-flight cases to durable files before propagating
                    # an error/stop. Resume only executes cases still missing.
                    errors.append(exc)
                    stopping = True
                finally:
                    active.pop(patient, None)
                    show_progress()

        async with bridge.core_memory.read_only():
            tasks = [asyncio.create_task(worker()) for _ in range(min(concurrency, len(ids)))]
            try:
                await asyncio.gather(*tasks)
            except BaseException:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                raise
        if errors:
            raise errors[0]
        if await bridge.state_fingerprint() != frozen_state:
            raise RuntimeError("Frozen test batch changed persistent memory")
        # Completion order never changes roster order or aggregate inputs.
        rows = [rows_by_patient[patient] for patient in ids]
        result = {**self._metrics(rows, bridge), **hospital_window_metrics(self._ledger, cursor)}
        result.update(test_concurrency=concurrency, test_batch_wall_time_s=perf_counter() - batch_started)
        from mematlas.core.evaluation import Prediction
        from mematlas.evaluation.metrics.common import Evaluator
        result["resource_metrics"] = Evaluator._response_resources([
            Prediction(row["patient_id"], None, None,
                       metadata={"response_time_s": row.get("response_time_s")}) for row in rows])
        atomic_write_json(self.output / "checkpoints" / f"learned_{learned}_{split}.json", result)
        if self.progress:
            self.progress.record_checkpoint(learned, split, result)
            self.progress.write()
        return result

    @staticmethod
    def _metrics(records, bridge):
        metrics = aggregate_openhospital_evaluations(records, expected_count=len(records))
        recoveries = [row.get("json_recovery", {}) for row in records]
        metrics["json_fallback_count"] = sum(row.get("fallbacks", 0) for row in recoveries)
        metrics["json_repair_count"] = sum(row.get("repairs", 0) for row in recoveries)
        departments = sorted({row["department"] for row in records})
        scope = bridge.scope
        results = tuple(row for row in bridge.history.results.values() if row.metadata.get("scope") == scope)
        metrics.update(case_reports=records, per_department={department: aggregate_openhospital_evaluations(
            [row for row in records if row["department"] == department]) for department in departments},
            learned_episodes=sum(row.learning == "submitted" for row in results),
            pending_feedback=sum(row.episode_id in bridge.history.pending_feedback for row in results))
        return metrics
