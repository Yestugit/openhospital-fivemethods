"""Observable episode outcomes; execution, feedback and learning are separate."""
from dataclasses import dataclass, field, asdict
from typing import Any, Mapping


def episode_counts(rows):
    rows = tuple(rows)
    return {"attempted_exposures": sum(row.execution != "pending" for row in rows),
            "completed_exposures": sum(row.execution == "completed" for row in rows),
            "valid_feedback_exposures": sum(row.feedback == "valid" for row in rows),
            "learned_exposures": sum(row.learning == "submitted" for row in rows),
            "failed_exposures": sum(row.execution == "failed" or row.learning in {"failed", "unknown"} for row in rows),
            "pending_feedback": sum(row.execution == "completed" and row.feedback == "pending" for row in rows)}


@dataclass(frozen=True)
class EpisodeResult:
    episode_id: str
    case_id: str | None = None
    execution: str = "completed"
    feedback: str = "pending"
    learning: str = "pending"
    error: Mapping[str, Any] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.episode_id or self.execution not in {"completed", "failed", "pending"}:
            raise ValueError("Invalid episode identity or execution status")
        if self.feedback not in {"pending", "valid", "invalid", "not_required"}:
            raise ValueError("Invalid feedback status")
        if self.learning not in {"pending", "submitted", "skipped", "failed", "unknown"}:
            raise ValueError("Invalid learning status")
        if self.learning == "submitted" and (self.execution != "completed" or self.feedback not in {"valid", "not_required"}):
            raise ValueError("Submitted learning requires completed execution and valid feedback")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        return cls(**dict(data))


@dataclass(frozen=True)
class BatchResult:
    episodes: tuple[EpisodeResult, ...] = ()
    metrics: Mapping[str, Any] = field(default_factory=dict)
    counts: Mapping[str, int | None] = field(default_factory=dict)

    def __post_init__(self):
        if len({episode.episode_id for episode in self.episodes}) != len(self.episodes):
            raise ValueError("Batch results cannot contain duplicate episode identities")

    def progress(self, expected_count: int) -> dict[str, int | None]:
        if self.episodes:
            if len(self.episodes) != expected_count:
                raise ValueError("Return one episode result for each scheduled case")
            counts = episode_counts(self.episodes)
        else:
            counts = {name: self.counts.get(name) for name in (
                "attempted_exposures", "completed_exposures", "valid_feedback_exposures",
                "learned_exposures", "failed_exposures", "pending_feedback")}
        for name, value in counts.items():
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= expected_count):
                raise ValueError(f"Invalid {name}; expected a count within the scheduled batch")
        completed, learned = counts["completed_exposures"], counts["learned_exposures"]
        if completed is not None and learned is not None and learned > completed:
            raise ValueError("Learned exposures cannot exceed completed exposures")
        attempted = counts["attempted_exposures"]
        if attempted is not None and completed is not None and completed > attempted:
            raise ValueError("Completed exposures cannot exceed attempted exposures")
        return counts

    @classmethod
    def coerce(cls, value):
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError("Training callbacks must return BatchResult or a result mapping")
        data = dict(value)
        if "episodes" in data:
            rows = data.pop("episodes")
            return cls(tuple(row if isinstance(row, EpisodeResult) else EpisodeResult.from_dict(row) for row in rows), data)
        counts = {key: data[key] for key in data
                  if key in {"attempted_exposures", "completed_exposures", "valid_feedback_exposures",
                             "learned_exposures", "failed_exposures", "pending_feedback"}}
        return cls(metrics=data, counts=counts)
