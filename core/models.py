"""Shared event, episode and memory output data. No Agent execution objects."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
import math
from typing import Any, Mapping, TypeAlias
from uuid import uuid4

Content: TypeAlias = str | list[dict[str, Any]]
Scope: TypeAlias = dict[str, str]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Modality(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    ARTIFACT = "artifact"


class EventKind(str, Enum):
    MESSAGE = "message"
    OBSERVATION = "observation"
    ACTION = "action"
    TOOL_RESULT = "tool_result"
    FEEDBACK = "feedback"
    EPISODE_START = "episode_start"
    EPISODE_END = "episode_end"
    CUSTOM = "custom"


def content_value(value: Content) -> Content:
    if isinstance(value, str):
        return value
    if not isinstance(value, (list, tuple)):
        raise TypeError("content must be text or a list of typed content blocks")
    blocks = []
    for row in value:
        if not isinstance(row, Mapping):
            raise TypeError("each content block must be a mapping")
        block = deepcopy(dict(row))
        kind = Modality(block.get("type"))
        required = "text" if kind == Modality.TEXT else "uri"
        if not isinstance(block.get(required), str):
            raise ValueError(f"{kind.value} content requires {required}")
        blocks.append(block)
    return blocks


def content_text(value: Content) -> str:
    if isinstance(value, str):
        return value
    return "\n".join(row["text"] for row in value if row["type"] == "text")


def content_modalities(value: Content) -> frozenset[Modality]:
    return frozenset({Modality.TEXT}) if isinstance(value, str) else frozenset(Modality(row["type"]) for row in value)


def _extensions(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("extensions must be a mapping")
    return deepcopy(dict(value))


@dataclass(frozen=True)
class MemoryEvent:
    kind: EventKind | str
    content: Content
    role: str | None = None
    event_id: str = field(default_factory=lambda: uuid4().hex)
    episode_id: str | None = None
    timestamp: datetime = field(default_factory=utc_now)
    extensions: Mapping[str, Any] = field(default_factory=dict)
    scope: Scope | None = None

    def __post_init__(self):
        from mematlas.core.scope import normalize_scope
        object.__setattr__(self, "kind", EventKind(self.kind))
        object.__setattr__(self, "content", content_value(self.content))
        object.__setattr__(self, "extensions", _extensions(self.extensions))
        if self.role is not None and not isinstance(self.role, str):
            raise TypeError("role must be text when supplied")
        if not isinstance(self.event_id, str) or not self.event_id:
            raise ValueError("event_id must be a nonempty string")
        if self.episode_id is not None and (not isinstance(self.episode_id, str) or not self.episode_id):
            raise ValueError("episode_id must be a nonempty string when supplied")
        timestamp = self.timestamp
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        if not isinstance(timestamp, datetime):
            raise TypeError("timestamp must be a datetime or ISO timestamp")
        object.__setattr__(self, "timestamp", timestamp if timestamp.tzinfo else timestamp.replace(tzinfo=timezone.utc))
        if self.scope is not None:
            object.__setattr__(self, "scope", normalize_scope(self.scope))

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "kind": self.kind.value, "timestamp": self.timestamp.isoformat()}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MemoryEvent:
        return cls(**deepcopy(dict(data)))


@dataclass(frozen=True)
class MemoryContribution:
    items: list[dict[str, Any]] = field(default_factory=list)
    context: list[dict[str, Any]] = field(default_factory=list)
    extensions: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        items, contexts = [], []
        for row in self.items:
            item = deepcopy(dict(row))
            item["content"] = content_value(item["content"])
            identity = item.get("memory_id")
            if identity is not None and (not isinstance(identity, str) or not identity):
                raise ValueError("memory_id must be a nonempty string when supplied")
            score = item.get("score")
            if score is not None and (isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score)):
                raise ValueError("memory score must be a finite number")
            if "source_event_ids" in item and (not isinstance(item["source_event_ids"], (list, tuple)) or any(not isinstance(x, str) for x in item["source_event_ids"])):
                raise ValueError("source_event_ids must be a list of strings")
            items.append(item)
        for row in self.context:
            context = deepcopy(dict(row))
            context["content"] = content_value(context["content"])
            indices = context.get("source_item_indices", [])
            if not isinstance(indices, (list, tuple)) or any(type(i) is not int or not 0 <= i < len(items) for i in indices):
                raise ValueError("context source_item_indices must refer to returned items")
            if items and not indices:
                raise ValueError("formatted context requires source_item_indices")
            context["source_item_indices"] = list(indices)
            contexts.append(context)
        object.__setattr__(self, "items", items)
        object.__setattr__(self, "context", contexts)
        object.__setattr__(self, "extensions", _extensions(self.extensions))

    def to_dict(self) -> dict[str, Any]:
        return deepcopy(asdict(self))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MemoryContribution:
        return cls(**deepcopy(dict(data)))


@dataclass(frozen=True)
class MemoryEpisode:
    episode_id: str = field(default_factory=lambda: uuid4().hex)
    task: str | None = None
    events: list[MemoryEvent] | None = None
    output: Any = None
    success: bool | None = None
    reward: float | None = None
    feedback: str | None = None
    applied_memory: list[MemoryContribution] = field(default_factory=list)
    extensions: Mapping[str, Any] = field(default_factory=dict)
    scope: Scope | None = None

    def __post_init__(self):
        from mematlas.core.scope import normalize_scope
        if not isinstance(self.episode_id, str) or not self.episode_id:
            raise ValueError("episode_id must be a nonempty string")
        if self.success is not None and not isinstance(self.success, bool):
            raise ValueError("success must be a boolean")
        if self.reward is not None and (isinstance(self.reward, bool) or not isinstance(self.reward, (int, float)) or not math.isfinite(self.reward)):
            raise ValueError("reward must be a finite number")
        if self.task is not None and not isinstance(self.task, str):
            raise TypeError("task must be text")
        if self.feedback is not None and not isinstance(self.feedback, str):
            raise TypeError("feedback must be text")
        if self.events is not None:
            if any(not isinstance(event, MemoryEvent) for event in self.events):
                raise TypeError("events must contain MemoryEvent objects")
            object.__setattr__(self, "events", deepcopy(list(self.events)))
        if any(not isinstance(row, MemoryContribution) for row in self.applied_memory):
            raise TypeError("applied_memory must contain complete MemoryContribution objects")
        object.__setattr__(self, "applied_memory", deepcopy(list(self.applied_memory)))
        object.__setattr__(self, "extensions", _extensions(self.extensions))
        if self.scope is not None:
            object.__setattr__(self, "scope", normalize_scope(self.scope))

    def to_dict(self) -> dict[str, Any]:
        return {**deepcopy(asdict(self)),
                "events": None if self.events is None else [event.to_dict() for event in self.events],
                "applied_memory": [row.to_dict() for row in self.applied_memory]}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MemoryEpisode:
        values = deepcopy(dict(data))
        if values.get("events") is not None:
            values["events"] = [MemoryEvent.from_dict(row) for row in values["events"]]
        values["applied_memory"] = [MemoryContribution.from_dict(row) for row in values.get("applied_memory", [])]
        return cls(**values)


@dataclass(frozen=True)
class MemoryUnit:
    """One independently understandable submission; many units may share a task.

    Feedback uses ``success``, ``reward`` and ``text`` when available. Additional
    evaluation details are preserved as a feedback event, without inferring an
    outcome from the task's completion status.
    """

    events: list[MemoryEvent] = field(default_factory=list)
    type: str = "message"
    write_ref: str = field(default_factory=lambda: uuid4().hex)
    episode_ref: str | None = None
    task: str | None = None
    output: Any = None
    feedback: Mapping[str, Any] = field(default_factory=dict)
    related_refs: tuple[str, ...] = ()
    applied_memory: list[MemoryContribution] = field(default_factory=list)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    extensions: Mapping[str, Any] = field(default_factory=dict)
    scope: Scope | None = None

    def __post_init__(self):
        if not isinstance(self.type, str) or not self.type.strip():
            raise ValueError("MemoryUnit.type must be nonempty text")
        if not isinstance(self.write_ref, str) or not self.write_ref:
            raise ValueError("write_ref must be nonempty text")
        if self.episode_ref is not None and (not isinstance(self.episode_ref, str) or not self.episode_ref):
            raise ValueError("episode_ref must be nonempty text when supplied")
        feedback = _extensions(self.feedback)
        # Reuse the existing outcome and trajectory validation.
        episode = MemoryEpisode(self.episode_ref or self.write_ref, task=self.task,
            events=self.events, output=self.output, success=feedback.get("success"),
            reward=feedback.get("reward"), feedback=feedback.get("text"),
            applied_memory=self.applied_memory, extensions=self.extensions, scope=self.scope)
        refs = tuple(self.related_refs)
        if any(not isinstance(ref, str) or not ref for ref in refs):
            raise ValueError("related_refs must contain nonempty references")
        if not episode.events and self.output is None and not feedback and not self.task:
            raise ValueError("A memory unit requires content, a task, an output or feedback")
        for name in ("events", "applied_memory", "extensions", "scope"):
            object.__setattr__(self, name, getattr(episode, name))
        object.__setattr__(self, "feedback", feedback)
        object.__setattr__(self, "metadata", _extensions(self.metadata))
        object.__setattr__(self, "related_refs", refs)

    def to_dict(self):
        return {**deepcopy(asdict(self)), "events": [event.to_dict() for event in self.events],
                "applied_memory": [row.to_dict() for row in self.applied_memory]}

    @classmethod
    def from_dict(cls, data):
        values = deepcopy(dict(data))
        values["events"] = [MemoryEvent.from_dict(row) for row in values.get("events", ())]
        values["applied_memory"] = [MemoryContribution.from_dict(row) for row in values.get("applied_memory", ())]
        return cls(**values)

    @classmethod
    def from_episode(cls, episode: MemoryEpisode):
        """Submit an existing complete execution record without losing attribution."""
        feedback = {key: value for key, value in {
            "success": episode.success, "reward": episode.reward, "text": episode.feedback,
        }.items() if value is not None}
        return cls(events=list(episode.events or ()), type="experience",
            write_ref=episode.episode_id, episode_ref=episode.episode_id,
            task=episode.task, output=episode.output, feedback=feedback,
            applied_memory=episode.applied_memory, extensions=episode.extensions, scope=episode.scope)

    def as_episode(self):
        return MemoryEpisode(self.episode_ref or self.write_ref, task=self.task, events=self.events,
            output=self.output, success=self.feedback.get("success"), reward=self.feedback.get("reward"),
            feedback=self.feedback.get("text"), applied_memory=self.applied_memory,
            extensions=self.extensions, scope=self.scope)


@dataclass(frozen=True)
class WriteReceipt:
    """An accepted submission reference, not a promise of a new stored item."""

    write_ref: str
    episode_ref: str


@dataclass(frozen=True)
class OperationTrace:
    method_id: str
    operation: str
    started_at: datetime
    duration_ms: float
    success: bool
    input_size: int = 0
    output_size: int = 0
    error_type: str | None = None
    error_message: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self):
        return {**asdict(self), "started_at": self.started_at.isoformat()}
