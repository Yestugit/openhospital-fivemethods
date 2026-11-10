"""ReasoningBank's trajectory-oriented evaluation adapter.

ReasoningBank is trained from externally labelled trajectories rather than
ordinary question/answer pairs.  The adapter therefore prepares the native
episode fields and returns retrieved groups directly for trajectory
benchmarks; regular answer generation remains available when an answer model
is supplied.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

from mematlas.core.models import MemoryEpisode, MemoryEvent, content_text
from mematlas.evaluation.qa import DefaultQA


class ReasoningBankQA(DefaultQA):
    """Map benchmark trajectory metadata to ReasoningBank's public contract."""

    @classmethod
    def requires_answer_model_for(cls, config, benchmark=None) -> bool:
        # The native trajectory benchmark scores retrieved groups and does not
        # require an answer-model call. Other benchmark profiles use normal QA.
        return getattr(benchmark, "benchmark_id", None) != "reasoningbank_trajectory"

    def __init__(self, config, services):
        super().__init__(config, services)
        benchmark = getattr(config, "benchmark", None)
        if isinstance(benchmark, Mapping):
            self._native_trajectory = str(benchmark.get("kind", "")).lower() in {
                "reasoningbank_trajectory", "reasoningbank-trajectories"}
        else:
            self._native_trajectory = getattr(benchmark, "benchmark_id", None) == "reasoningbank_trajectory"

    @staticmethod
    def retrieval_extensions(metadata):
        extensions = DefaultQA.retrieval_extensions(metadata)
        native = metadata.get("reasoningbank")
        if isinstance(native, Mapping):
            extensions["reasoningbank"] = dict(native)
        return extensions

    @staticmethod
    def prepare_episode(case, episode, *, last_for_scope):
        metadata = getattr(case, "metadata", {})
        values = metadata.get("reasoningbank")
        if not isinstance(values, Mapping):
            values = {}
        # A trajectory benchmark supplies the complete labelled episode. For
        # JSON/custom inputs, allow metadata to provide the missing fields.
        updates: dict[str, Any] = {}
        if episode.task is None and metadata.get("task"):
            updates["task"] = str(metadata["task"])
        if episode.success is None and "success" in values:
            updates["success"] = bool(values["success"])
        if episode.feedback is None and values.get("feedback") is not None:
            updates["feedback"] = str(values["feedback"])
        merged = dict(episode.extensions)
        merged.update({"reasoningbank": dict(values)} if values else {})
        if merged != dict(episode.extensions):
            updates["extensions"] = merged
        return replace(episode, **updates) if updates else episode

    async def __call__(self, case, memory) -> MemoryEpisode:
        if not self._native_trajectory:
            return await super().__call__(case, memory)
        contribution = await self.retrieve(case, memory)
        # Native trajectory scoring consumes applied_memory, not prose.
        event = MemoryEvent("message", content_text(case.query), role="user", scope=case.scope)
        return MemoryEpisode(task=content_text(case.query), scope=case.scope,
                             events=[event], applied_memory=[contribution])


__all__ = ["ReasoningBankQA"]
