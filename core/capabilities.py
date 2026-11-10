from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Mapping

from mematlas.core.models import EventKind, Modality


class IntegrationSlot(str, Enum):
    CONTEXT = "context"
    TOOLS = "tools"
    MODEL_HOOK = "model_hook"
    ARTIFACT = "artifact"
    EPISODE = "episode"


class MemoryForm(str, Enum):
    RECORD = "record"
    GRAPH = "graph"
    HIERARCHICAL = "hierarchical"
    PROCEDURAL = "procedural"
    MULTIMODAL = "multimodal"
    PARAMETRIC = "parametric"
    LATENT = "latent"
    POLICY = "policy"


class Operation(str, Enum):
    INDEX = "index"
    ADD = "add"
    SEARCH = "search"
    EPISODE_END = "episode_end"
    DETAIL = "detail"
    UPDATE = "update"
    DELETE = "delete"
    FLUSH = "flush"
    RESET = "reset"
    SAVE = "save"
    LOAD = "load"
    COMMAND = "command"
    NATIVE_RUN = "native_run"
    TRAIN = "train"


@dataclass(frozen=True)
class MethodCapabilities:
    modalities: frozenset[Modality]
    forms: frozenset[MemoryForm]
    operations: frozenset[Operation]
    integration_slots: frozenset[IntegrationSlot]
    accepted_events: frozenset[EventKind] = field(default_factory=frozenset)
    scopes: frozenset[str] = field(default_factory=lambda: frozenset({"user", "agent", "session"}))
    default_scope: str = "user"
    requires_training: bool = False
    requires_gpu: bool = False
    supports_persistence: bool = True
    supports_async: bool = True
    read_only_mode: str = "unsupported"
    read_only_commands: frozenset[str] = field(default_factory=frozenset)
    required_episode_fields: frozenset[str] = field(default_factory=frozenset)
    max_scopes: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "modalities": sorted(x.value for x in self.modalities),
            "forms": sorted(x.value for x in self.forms),
            "operations": sorted(x.value for x in self.operations),
            "integration_slots": sorted(x.value for x in self.integration_slots),
            "accepted_events": sorted(x.value for x in self.accepted_events),
            "scopes": sorted(self.scopes),
            "default_scope": self.default_scope,
            "requires_training": self.requires_training,
            "requires_gpu": self.requires_gpu,
            "supports_persistence": self.supports_persistence,
            "supports_async": self.supports_async,
            "read_only_mode": self.read_only_mode,
            "read_only_commands": sorted(self.read_only_commands),
            "required_episode_fields": sorted(self.required_episode_fields),
            "max_scopes": self.max_scopes,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MethodCapabilities":
        return cls(
            modalities=frozenset(Modality(x) for x in data.get("modalities", [])),
            forms=frozenset(MemoryForm(x) for x in data.get("forms", [])),
            operations=frozenset(Operation(x) for x in data.get("operations", [])),
            integration_slots=frozenset(IntegrationSlot(x) for x in data.get("integration_slots", [])),
            accepted_events=frozenset(EventKind(x) for x in data.get("accepted_events", [])),
            scopes=frozenset(data.get("scopes", [])),
            default_scope=str(data.get("default_scope", "user")),
            requires_training=bool(data.get("requires_training", False)),
            requires_gpu=bool(data.get("requires_gpu", False)),
            supports_persistence=bool(data.get("supports_persistence", True)),
            supports_async=bool(data.get("supports_async", True)),
            read_only_mode=str(data.get("read_only_mode", "unsupported")),
            read_only_commands=frozenset(data.get("read_only_commands", [])),
            required_episode_fields=frozenset(data.get("required_episode_fields", [])),
            max_scopes=data.get("max_scopes"),
        )


@dataclass(frozen=True)
class MethodDescriptor:
    method_id: str
    name: str
    summary: str
    capabilities: MethodCapabilities
    manifest_version: int = 1
    version: str = "1"
    paper_url: str | None = None
    repository_url: str | None = None
    source_subdir: str | None = None
    recommended_executor: str = "in_process"
    fidelity: str = "unspecified"
    implementation_strategy: str = "native_plugin"
    internal_token_tracking: str = "untracked"
    implementation_notes: tuple[str, ...] = ()
    config_schema: Mapping[str, Any] = field(default_factory=dict)
    limitations: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    requirements: Mapping[str, Any] = field(default_factory=dict)
    route_hints: tuple[str, ...] = ()
    storage: Mapping[str, Any] = field(default_factory=dict)
    interface_version: int = 3
    commands: Mapping[str, Any] = field(default_factory=dict)
    input_schemas: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not self.method_id.strip():
            errors.append("method_id is required")
        if not self.name.strip():
            errors.append(f"{self.method_id}: name is required")
        if not self.summary.strip():
            errors.append(f"{self.method_id}: summary is required")
        if self.manifest_version < 1:
            errors.append(f"{self.method_id}: manifest_version must be >= 1")
        if self.capabilities.max_scopes is not None and (isinstance(self.capabilities.max_scopes, bool)
                or not isinstance(self.capabilities.max_scopes, int) or self.capabilities.max_scopes < 1):
            errors.append(f"{self.method_id}: max_scopes must be a positive integer")
        if self.capabilities.required_episode_fields - {"success", "reward", "task", "trajectory"}:
            errors.append(f"{self.method_id}: unknown required episode field")
        if self.capabilities.read_only_mode not in {"native", "snapshot", "unsupported"}:
            errors.append(f"{self.method_id}: invalid read_only_mode")
        if self.capabilities.read_only_mode == "snapshot" and not {
            Operation.SAVE, Operation.LOAD
        }.issubset(self.capabilities.operations):
            errors.append(f"{self.method_id}: snapshot read-only mode requires save/load state")
        if self.capabilities.default_scope not in self.capabilities.scopes and self.capabilities.default_scope not in {"native", "custom", "runtime"}:
            errors.append(f"{self.method_id}: default_scope must be listed in capabilities.scopes")
        return errors

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["capabilities"] = self.capabilities.to_dict()
        return result

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MethodDescriptor":
        values = dict(data)
        values["capabilities"] = MethodCapabilities.from_dict(values["capabilities"])
        values["limitations"] = tuple(values.get("limitations", []))
        values["tags"] = tuple(values.get("tags", []))
        values["route_hints"] = tuple(values.get("route_hints", []))
        values["implementation_notes"] = tuple(values.get("implementation_notes", []))
        return cls(**values)
