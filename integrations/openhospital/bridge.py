from __future__ import annotations
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from typing import Any

from mematlas.runtime.memory import Memory
from mematlas.evaluation.experience_memory import _MemoryExperience
from mematlas.evaluation.feedback import FeedbackSchedule
from mematlas.core.config import MemAtlasConfig
from mematlas.core.models import MemoryContribution, MemoryEpisode, Scope, content_text, utc_now
from mematlas.core.evaluation import BenchmarkCase
from mematlas.core.files import atomic_write_json
from mematlas.core.provenance import json_sha256
from mematlas.core.capabilities import Operation
from mematlas.core.contracts import ProtocolRequirements
from mematlas.core.contributions import applied_sources
from .encounter import encounter_events, task_description
from .feedback import FeedbackPolicy
from .context import bounded_text
from mematlas.evaluation.qa import _qa_type


def _flatten(value: Any) -> str:
    if value is None:
        return ""
    return value.strip() if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)


def _select_context(contribution, limit):
    """Keep only presented sources, retaining the method's full association data."""
    rows = contribution.context or contribution.items
    chosen = [i for i, row in enumerate(rows) if content_text(row["content"])]
    if limit is not None:
        chosen = chosen[:limit]
    texts = [content_text(rows[i]["content"]) for i in chosen]
    if contribution.context:
        indices = list(dict.fromkeys(i for n in chosen for i in rows[n]["source_item_indices"]))
        positions = {old: new for new, old in enumerate(indices)}
        contexts = [{**rows[n], "source_item_indices": [positions[i] for i in rows[n]["source_item_indices"]]}
                    for n in chosen]
    else:
        indices, contexts = chosen, []
    return texts, MemoryContribution(items=[contribution.items[i] for i in indices],
                                     context=contexts, extensions=contribution.extensions)


class OpenHospitalMemoryIntegration(_MemoryExperience):
    """Map doctor retrieval, real clinical feedback and recovery to Memory v3."""

    def __init__(self, memory: Memory, *, department="心内科", top_k=None, max_contexts=None,
                 max_query_chars=8000, patient_departments=None, feedback_policy=None,
                 feedback_timing="immediate", feedback_delay=0,
                 journal_dir=None):
        super().__init__(memory)
        self.department, self.top_k = department, top_k
        self.max_contexts, self.max_query_chars = max_contexts, max_query_chars
        self.patient_departments = dict(patient_departments or {})
        self.feedback_policy = feedback_policy or FeedbackPolicy()
        self.schedule = FeedbackSchedule(feedback_timing, feedback_delay)
        self.feedback_timing, self.feedback_delay = feedback_timing, feedback_delay
        self.feedback_resolver = None
        self.method_adapter = _qa_type(memory.plugin)
        self.journal_dir = Path(journal_dir) if journal_dir else None
        self._case_metrics, self._training_occurrences = [], {}
        self._applications, self._pending_applications, self._bundles = {}, {}, {}
        # Held-out provenance belongs to evaluation and never enters Memory snapshots.
        self._test_applications, self._pending_test_applications = {}, {}
        self._training_end = None

    @staticmethod
    def preflight(descriptor):
        ProtocolRequirements(operations=frozenset({Operation.SEARCH, Operation.ADD, Operation.EPISODE_END,
            Operation.SAVE, Operation.LOAD}), episode_fields=frozenset({"task", "trajectory", "success", "reward"}),
            frozen=True, recovery=True).validate(descriptor)

    async def start(self):
        self.preflight(self.core_memory.plugin.descriptor)
        await super().start()

    def configure_training_batch(self, patient_ids, *, batch_index, end_of_training=False):
        if batch_index < 1 or not patient_ids or len(patient_ids) != len(set(patient_ids)):
            raise ValueError("Training batches require a positive index and distinct patients")
        self._training_occurrences = {patient: f"batch:{batch_index}:patient:{patient}" for patient in patient_ids}
        self._training_end = self._training_occurrences[patient_ids[-1]] if end_of_training else None
        self._applications.clear()
        self._pending_applications.clear()
        self._test_applications.clear()
        self._pending_test_applications.clear()

    def _occurrence(self, patient_id):
        if self._training_occurrences and patient_id not in self._training_occurrences:
            raise ValueError(f"Patient {patient_id!r} is outside the active training batch")
        return self._training_occurrences.get(patient_id, patient_id)

    def _department(self, patient_id):
        return self.patient_departments.get(patient_id, self.department)

    @classmethod
    def from_config(cls, config, *, services=None, registry=None, **options):
        parsed = config if isinstance(config, MemAtlasConfig) else MemAtlasConfig.from_dict(config)
        from mematlas.core.registry import default_registry
        registry = registry or default_registry()
        cls.preflight(registry.get(parsed.method.method_id).descriptor)
        return cls(Memory(parsed, services=services, registry=registry), **options)

    @property
    def scope(self) -> Scope:
        return dict(self.core_memory.config.scope) or {
            "user_id": f"openhospital:{self.department}", "agent_id": "doctor"}

    async def retrieve(self, *, patient_id, phase, chat_history=(), examinations=None, mode="train"):
        await self.start()
        if mode not in {"train", "test"}:
            raise ValueError("OpenHospital retrieval mode must be train or test")
        training = mode == "train" and not self.core_memory.is_read_only
        occurrence = self._occurrence(patient_id) if training else None
        parts = [f"科室：{self._department(patient_id)}", f"当前诊疗阶段：{phase}"]
        if chat_history:
            parts.append("问诊记录：\n" + _flatten(list(chat_history)))
        if examinations:
            parts.append("检查信息：\n" + _flatten(examinations))
        if len(parts) == 2:
            parts.append("请检索可复用于当前病例的诊疗经验。")
        header = "\n\n".join(parts[:2])
        query = header + "\n\n" + bounded_text("\n\n".join(parts[2:]), self.max_query_chars-len(header)-2)
        extensions = self.method_adapter.retrieval_extensions({"task_type": "clinical_reasoning"})
        if mode == "test" and not self.core_memory.is_read_only:
            async with self.core_memory.read_only():
                contribution = await self.core_memory.search(query, scope=self.scope, top_k=self.top_k, extensions=extensions)
        else:
            contribution = await self.core_memory.search(query, scope=self.scope, top_k=self.top_k, extensions=extensions)
        texts, used = _select_context(contribution, self.max_contexts)
        self._pending_test_applications.pop((patient_id, phase), None)
        if training:
            self._pending_applications.pop((occurrence, phase), None)
            if texts:
                self._pending_applications[(occurrence, phase)] = used
            else:
                # Empty results can still carry method attribution for learning.
                self._applications.setdefault(occurrence, []).append(used)
        elif texts:
            self._pending_test_applications[(patient_id, phase)] = used
        return texts

    def confirm_application(self, patient_id, *, phase):
        key = (patient_id, phase)
        used = self._pending_test_applications.pop(key, None)
        if used is not None:
            self._test_applications.setdefault(patient_id, []).append(used)
            return
        if self.core_memory.is_read_only:
            return
        occurrence = self._occurrence(patient_id)
        used = self._pending_applications.pop((occurrence, phase), None)
        if used is not None:
            self._applications.setdefault(occurrence, []).append(used)

    def _begin_test_case(self, patient_id):
        self._test_applications.pop(patient_id, None)
        for key in tuple(self._pending_test_applications):
            if key[0] == patient_id:
                self._pending_test_applications.pop(key)

    def _test_contributions(self, patient_id):
        return [row.to_dict() for row in self._test_applications.get(patient_id, ())]

    def _bundle(self, patient_id, case_state, final_result, evaluation_report=None, reflection=None,
                applied_memory=None):
        occurrence = self._occurrence(patient_id)
        if applied_memory is None:
            applied_memory = self._applications.get(occurrence, [])
        state = deepcopy(case_state)
        previous = self._bundles.get(occurrence, {}).get("case_state", {})
        state.setdefault("encounter_timestamp", previous.get("encounter_timestamp") or utc_now().isoformat())
        return {"patient_id": patient_id, "case_state": state, "final_result": deepcopy(final_result),
                "evaluation_report": deepcopy(evaluation_report or {}), "reflection": deepcopy(reflection or {}),
                "applied_memory": [row.to_dict() if isinstance(row, MemoryContribution) else deepcopy(row)
                                   for row in applied_memory]}

    def _journal(self, occurrence, bundle):
        self._bundles[occurrence] = deepcopy(bundle)
        if self.journal_dir:
            atomic_write_json(self.journal_dir / f"{json_sha256(occurrence)}.json", bundle)

    async def append_case_reflection(self, *, patient_id, evaluation_reflection=None, case_state=None,
                                     evaluation_report=None, final_result=None, **_):
        await self.learn_case(patient_id=patient_id, reflection=evaluation_reflection or {},
            case_state=case_state or {}, evaluation_report=evaluation_report or {}, final_result=final_result or {})

    async def record_execution(self, *, patient_id, case_state, final_result):
        if self.core_memory.is_read_only:
            return
        occurrence = self._occurrence(patient_id)
        status = self.history.result(occurrence, scope=self.scope)
        if status.learning == "submitted":
            return
        if status.execution == "completed":
            await self.core_memory.episode_end(occurrence, scope=self.scope)
            return
        self._journal(occurrence, self._bundle(patient_id, case_state, final_result))
        self.record_completed(occurrence, scope=self.scope, case_id=patient_id)
        # The clinical task has ended even when feedback is delivered later.
        # No trajectory/outcome is learned until add() receives that unit.
        await self.core_memory.episode_end(occurrence, scope=self.scope)

    async def record_score(self, *, patient_id, case_state, final_result, evaluation_report):
        self.feedback_policy.evaluate(evaluation_report)
        if not self.core_memory.is_read_only:
            bundle = self._bundle(patient_id, case_state, final_result, evaluation_report)
            self._journal(self._occurrence(patient_id), {**bundle, "needs_reflection": True})

    async def learn_case(self, *, patient_id, reflection, case_state, evaluation_report, final_result,
                         applied_memory=None):
        await self.start()
        if self.core_memory.is_read_only:
            raise ValueError("Held-out feedback cannot enter the training queue")
        self.feedback_policy.evaluate(evaluation_report)
        occurrence = self._occurrence(patient_id)
        bundle = self._bundle(patient_id, case_state, final_result, evaluation_report, reflection, applied_memory)
        self._journal(occurrence, bundle)
        # Cached encounters may enter here without record_execution(). Keep the
        # boundary ahead of the feedback queue so it cannot release pending data.
        await self.core_memory.episode_end(occurrence, scope=self.scope)
        try:
            await self.schedule.accept(self, self._episode(occurrence, **bundle), self._deliver)
        finally:
            self._sync_learning()

    async def _deliver(self, episode):
        occurrence = episode.episode_id
        bundle = deepcopy(self._bundles[occurrence])
        if not bundle.get("evaluation_report") or bundle.get("needs_reflection"):
            if self.feedback_resolver is None:
                raise ValueError("Deferred clinical feedback requires an execution resolver")
            bundle = await self.feedback_resolver(bundle, self, occurrence)
            self._journal(occurrence, bundle)
        completed = self._episode(occurrence, **bundle)
        completed = replace(completed, extensions={**completed.extensions,
            **{key: value for key, value in episode.extensions.items() if key != "openhospital"}})
        return completed

    async def deliver_feedback(self, *, batch_end=False):
        try:
            await self.schedule.deliver_due(self, self._deliver, batch_end=batch_end)
        finally:
            self._sync_learning()

    def _episode(self, occurrence, patient_id, case_state, final_result, reflection,
                 evaluation_report, applied_memory):
        department = self._department(patient_id)
        reward, success = self.feedback_policy.evaluate(evaluation_report) if evaluation_report else (None, None)
        feedback = {"reflection": reflection, "evaluation": evaluation_report} if evaluation_report else {}
        extensions = {"openhospital": {"patient_id": patient_id, "department": department,
            "evaluation": dict(evaluation_report), "reward_policy": self.feedback_policy.to_dict()}}
        episode = MemoryEpisode(episode_id=occurrence, scope=self.scope,
            task=task_description(department, case_state),
            events=list(encounter_events(self.scope, occurrence, patient_id, department, case_state, final_result, feedback)),
            output=dict(final_result), reward=reward, success=success, feedback=_flatten(reflection) or None,
            applied_memory=[row if isinstance(row, MemoryContribution) else MemoryContribution.from_dict(row)
                            for row in applied_memory], extensions=extensions)
        case = BenchmarkCase(occurrence, episode.task, scope=self.scope,
                             metadata={"task_type": "clinical_reasoning"})
        return self.method_adapter.prepare_episode(case, episode, last_for_scope=occurrence == self._training_end)

    def _sync_learning(self):
        recorded = {row["episode_id"] for row in self._case_metrics}
        for occurrence, bundle in self._bundles.items():
            status = self.history.result(occurrence, scope=self.scope)
            if status.learning != "submitted" or occurrence in recorded:
                continue
            episode = self.episode(status.episode_id)
            self._case_metrics.append({"episode_id": occurrence, "patient_id": bundle["patient_id"],
                "department": self._department(bundle["patient_id"]), "reward": episode.reward,
                "success": episode.success, "applied_memory_ids": sorted({row["memory_id"]
                    for row in applied_sources(episode.applied_memory) if row.get("memory_id")})})
            self._applications.pop(occurrence, None)
            for key in list(self._pending_applications):
                if key[0] == occurrence:
                    self._pending_applications.pop(key)

    def snapshot_metadata(self):
        return {"case_metrics": deepcopy(self._case_metrics), "bundles": deepcopy(self._bundles),
                "feedback_schedule": {"timing": self.feedback_timing, "delay": self.feedback_delay}}

    def restore_metadata(self, metadata):
        if metadata.get("feedback_schedule") != {"timing": self.feedback_timing, "delay": self.feedback_delay}:
            raise ValueError("Snapshot feedback schedule differs from the active protocol")
        self._case_metrics = deepcopy(metadata["case_metrics"])
        self._bundles = deepcopy(metadata["bundles"])
        self._training_occurrences.clear()
        self._applications.clear()
        self._pending_applications.clear()
        self._test_applications.clear()
        self._pending_test_applications.clear()
        self._training_end = None

    def reuse_statistics(self):
        from collections import Counter
        counts = Counter(identity for row in self._case_metrics for identity in row["applied_memory_ids"])
        return {"learned_encounters":len(self._case_metrics),
            "encounters_with_applied_memory":sum(bool(row["applied_memory_ids"]) for row in self._case_metrics),
            "distinct_applied_memories":len(counts), "cross_encounter_reuses":sum(counts.values()) - len(counts),
            "interpretation":"observed_application_frequency_not_causal_effect"}


