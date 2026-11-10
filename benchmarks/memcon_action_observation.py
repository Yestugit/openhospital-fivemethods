"""Compact MemCon action/observation benchmark for ALFWorld/PDDL-shaped data."""
from __future__ import annotations

from typing import Any, Iterable, Mapping

from mematlas.benchmarks.base import BenchmarkAdapter
from mematlas.core.evaluation import BenchmarkCase, Prediction
from mematlas.core.models import MemoryEpisode, MemoryEvent


class MemConActionObservationBenchmark(BenchmarkAdapter):
    benchmark_id = "memcon_action_observation"

    def __init__(self, *, scope: Mapping[str, str] | None = None) -> None:
        self.scope = dict(scope or {"user_id": "memcon-action-slice"})
        self._cases = self._build_cases()

    def _trajectory(self, case_id: str, task: str, rows: tuple[tuple[str, str, Mapping[str, Any]], ...], success: bool,
                    feedback: str) -> MemoryEpisode:
        events = [MemoryEvent(kind, text, event_id=f"{case_id}:{index}", episode_id=case_id,
                              extensions=dict(values)) for index, (kind, text, values) in enumerate(rows)]
        return MemoryEpisode(episode_id=case_id, task=task, events=events, success=success,
                             feedback=feedback, scope=self.scope)

    def _build_cases(self) -> tuple[BenchmarkCase, ...]:
        success = self._trajectory("alf-1", "Put the apple in the box", (
            ("episode_start", "Put the apple in the box", {"task": "Put the apple in the box"}),
            ("action", "take apple", {"action": "take apple"}),
            ("observation", "You pick up the apple", {}),
            ("action", "put apple in box", {"action": "put apple in box"}),
            ("observation", "The apple is in the box", {}),
        ), True, "goal reached")
        failure = self._trajectory("pddl-1", "Open the locked door", (
            ("episode_start", "Open the locked door", {"task": "Open the locked door"}),
            ("action", "open door", {"action": "open door"}),
            ("observation", "The door is locked", {}),
        ), False, "precondition failed")
        return (
            BenchmarkCase("train-alfworld-success", success.task or "", self.scope, episode=success,
                          metadata={"split": "train", "label": True}),
            BenchmarkCase("train-pddl-failure", failure.task or "", self.scope, episode=failure,
                          metadata={"split": "train", "label": False}),
            BenchmarkCase("test-alfworld-success", "Put the apple in the box", self.scope,
                          metadata={"split": "test", "expected_memory_type": "success_experience", "top_k": 5}),
            BenchmarkCase("test-pddl-failure", "Open the locked door", self.scope,
                          metadata={"split": "test", "expected_memory_type": "failure_experience", "top_k": 5}),
        )

    def cases(self, split: str = "test") -> Iterable[BenchmarkCase]:
        return (case for case in self._cases
                if split == "all" or case.metadata.get("split") == split)

    def native_score(self, predictions: list[Prediction]) -> Mapping[str, Any]:
        if not predictions:
            return {"memcon.label_recall": 0.0, "memcon.source_provenance": 0.0,
                    "memcon.evaluated_cases": 0}
        labels = provenance = 0
        for prediction in predictions:
            expected = prediction.metadata.get("case_metadata", {}).get("expected_memory_type")
            items = [item for contribution in prediction.applied_memory for item in contribution.items]
            matching = [item for item in items if item.get("memory_type") == expected]
            labels += int(bool(matching))
            provenance += int(any(item.get("source_event_ids") for item in matching))
        total = len(predictions)
        return {"memcon.label_recall": labels / total, "memcon.source_provenance": provenance / total,
                "memcon.evaluated_cases": total, "memcon.train_episodes": len(tuple(self.cases("train")))}

    def describe(self) -> Mapping[str, Any]:
        return {"benchmark_id": self.benchmark_id, "kind": "native_action_observation",
                "train_cases": len(tuple(self.cases("train"))), "test_cases": len(tuple(self.cases("test"))),
                "source": "synthetic ALFWorld/PDDL-shaped trajectories",
                "scoring": "success/failure memory label and event provenance",
                "upstream_equivalents": ["ALFWorld", "PDDL"]}


__all__ = ["MemConActionObservationBenchmark"]
