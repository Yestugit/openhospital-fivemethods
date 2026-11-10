"""MemCon evaluation adapter for event-driven task episodes."""
from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

from mematlas.evaluation.qa import DefaultQA


class MemConQA(DefaultQA):
    """Carry task metadata and MemCon retrieval limits through QA."""

    @classmethod
    def requires_answer_model_for(cls, config, benchmark=None) -> bool:
        return getattr(benchmark, "benchmark_id", None) != "memcon_action_observation"

    def __init__(self, config, services):
        super().__init__(config, services)
        benchmark = getattr(config, "benchmark", None)
        if isinstance(benchmark, Mapping):
            self._native_action_observation = str(benchmark.get("kind", "")).lower() in {
                "memcon_action_observation", "memcon-action-observation"}
        else:
            self._native_action_observation = getattr(benchmark, "benchmark_id", None) == "memcon_action_observation"

    async def __call__(self, case, memory):
        if not self._native_action_observation:
            return await super().__call__(case, memory)
        contribution = await self.retrieve(case, memory)
        from mematlas.core.models import MemoryEpisode, content_text
        return MemoryEpisode(task=content_text(case.query), scope=case.scope,
                             applied_memory=[contribution])

    @staticmethod
    def _native(metadata: Mapping[str, Any]) -> dict[str, Any]:
        value = metadata.get("memcon")
        return dict(value) if isinstance(value, Mapping) else {}

    @classmethod
    def retrieval_extensions(cls, metadata):
        extensions = DefaultQA.retrieval_extensions(metadata)
        native = cls._native(metadata)
        if native:
            existing = extensions.get("memcon")
            extensions["memcon"] = {**(existing if isinstance(existing, Mapping) else {}), **native}
        return extensions

    @classmethod
    def prepare_episode(cls, case, episode, *, last_for_scope):
        metadata = getattr(case, "metadata", {})
        native = cls._native(metadata)
        merged = dict(episode.extensions)
        existing = merged.get("memcon")
        combined = {**(existing if isinstance(existing, Mapping) else {}), **native}
        if combined:
            merged["memcon"] = combined
        updates: dict[str, Any] = {}
        if merged != dict(episode.extensions):
            updates["extensions"] = merged
        task = metadata.get("task") or combined.get("task")
        if episode.task is None and task:
            updates["task"] = str(task)
        return replace(episode, **updates) if updates else episode


__all__ = ["MemConQA"]
