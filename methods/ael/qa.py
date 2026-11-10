"""AEL evaluation adapter for structured portfolio episodes."""
from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

from mematlas.evaluation.qa import DefaultQA


class AELQA(DefaultQA):
    """Preserve AEL's structured query fields and numeric outcome score."""

    @classmethod
    def requires_answer_model_for(cls, config, benchmark=None) -> bool:
        return getattr(benchmark, "benchmark_id", None) != "ael_portfolio_slice"

    def __init__(self, config, services):
        super().__init__(config, services)
        benchmark = getattr(config, "benchmark", None)
        if isinstance(benchmark, Mapping):
            self._native_portfolio = str(benchmark.get("kind", "")).lower() in {
                "ael_portfolio_slice", "ael-portfolio-slice"}
        else:
            self._native_portfolio = getattr(benchmark, "benchmark_id", None) == "ael_portfolio_slice"

    async def __call__(self, case, memory):
        if not self._native_portfolio:
            return await super().__call__(case, memory)
        contribution = await self.retrieve(case, memory)
        from mematlas.core.models import MemoryEpisode, content_text
        return MemoryEpisode(task=content_text(case.query), scope=case.scope,
                             applied_memory=[contribution])

    @staticmethod
    def _native(metadata: Mapping[str, Any]) -> dict[str, Any]:
        value = metadata.get("ael")
        return dict(value) if isinstance(value, Mapping) else {}

    @classmethod
    def retrieval_extensions(cls, metadata):
        extensions = DefaultQA.retrieval_extensions(metadata)
        native = cls._native(metadata)
        # Native portfolio benchmark cases commonly keep ticker/sector/task
        # fields at case metadata top level. Preserve them under AEL's
        # structured extension namespace for controller selection.
        for key in ("ticker", "sector", "task_type", "planner", "market_regime", "tool_name", "memory_query"):
            if key in metadata and metadata[key] not in (None, ""):
                native.setdefault(key, metadata[key])
        if native:
            existing = extensions.get("ael")
            extensions["ael"] = {**(existing if isinstance(existing, Mapping) else {}), **native}
        return extensions

    @classmethod
    def prepare_episode(cls, case, episode, *, last_for_scope):
        metadata = getattr(case, "metadata", {})
        native = cls._native(metadata)
        merged = dict(episode.extensions)
        existing = merged.get("ael")
        combined = {**(existing if isinstance(existing, Mapping) else {}), **native}
        # Keep the native score numeric. AEL intentionally does not infer it
        # from a boolean success label.
        if episode.reward is None and "outcome_score" in combined:
            try:
                reward = float(combined["outcome_score"])
            except (TypeError, ValueError):
                reward = None
        else:
            reward = episode.reward
        if combined:
            merged["ael"] = combined
        updates: dict[str, Any] = {}
        if merged != dict(episode.extensions):
            updates["extensions"] = merged
        if reward is not None and episode.reward is None:
            updates["reward"] = reward
        if episode.task is None and metadata.get("task"):
            updates["task"] = str(metadata["task"])
        return replace(episode, **updates) if updates else episode


__all__ = ["AELQA"]
