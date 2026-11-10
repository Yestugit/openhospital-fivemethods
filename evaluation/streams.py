"""Explicit task schedules with independent patient rosters and exposures."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Mapping

from mematlas.evaluation.experience import _ExperiencePlan


@dataclass(frozen=True)
class _ExperienceStreamPlan(_ExperiencePlan):
    """Training IDs form a disjoint roster; batches define replay/order.

    A batch contains distinct IDs so environments keyed by patient ID can
    execute it safely. The same ID may reappear in a later batch. Checkpoints
    count completed exposures and must coincide with batch boundaries.
    """

    checkpoint_basis: str = field(default="completed", kw_only=True)
    training_batches: tuple[tuple[str, ...], ...] = ()
    stream_metadata: Mapping[str, Any] = field(default_factory=dict)
    feedback_boundaries: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        self._validate_splits()
        if (not isinstance(self.batch_count, int) or isinstance(self.batch_count, bool)
                or self.batch_count < 1 or self.batch_count != len(self.training_batches)):
            raise ValueError("batch_count must equal the nonempty training schedule length")
        roster = set(self.train_ids)
        observed: set[str] = set()
        boundaries = {0}
        total = 0
        for batch in self.training_batches:
            if not batch or len(set(batch)) != len(batch):
                raise ValueError("Each training batch must contain distinct, nonempty case IDs")
            if not set(batch).issubset(roster):
                raise ValueError("Training schedule contains IDs outside the training roster")
            observed.update(batch)
            total += len(batch)
            boundaries.add(total)
        if observed != roster:
            raise ValueError("Training roster must match the distinct IDs in the schedule")
        if not self.feedback_boundaries:
            object.__setattr__(self, "feedback_boundaries", tuple(sorted(boundaries - {0})))
        if (tuple(sorted(set(self.feedback_boundaries))) != self.feedback_boundaries
                or any(isinstance(value, bool) or not isinstance(value, int) for value in self.feedback_boundaries)
                or not set(self.feedback_boundaries).issubset(boundaries - {0})
                or self.feedback_boundaries[-1] != total):
            raise ValueError("Feedback boundaries must be ordered execution boundaries including the final exposure")
        self._validate_checkpoints()
        if not set(self.effective_checkpoints()).issubset(boundaries):
            raise ValueError("Every checkpoint must align with a training batch boundary")

    @property
    def training_count(self) -> int:
        return sum(len(batch) for batch in self.training_batches)

    def batches(self) -> tuple[tuple[str, ...], ...]:
        return self.training_batches

    def with_checkpoints(self, checkpoints: tuple[int, ...], *,
                         feedback_boundaries: tuple[int, ...] | None = None) -> "_ExperienceStreamPlan":
        """Split execution windows while retaining the original feedback schedule."""
        position, batches = 0, []
        feedback = self.feedback_boundaries if feedback_boundaries is None else feedback_boundaries
        for original in self.training_batches:
            start = 0
            cuts = sorted({point - position for point in (*checkpoints, *feedback)
                           if position < point < position + len(original)} | {len(original)})
            for cut in cuts:
                batches.append(original[start:cut])
                start = cut
            position += len(original)
        return replace(self, training_batches=tuple(batches), batch_count=len(batches), checkpoints=checkpoints,
                       feedback_boundaries=feedback)

    def protocol_details(self) -> dict[str, Any]:
        return {"schedule": "explicit_task_stream", "stream": dict(self.stream_metadata),
                "feedback_boundaries": self.feedback_boundaries}
