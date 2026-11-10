from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Mapping

from mematlas.core.capabilities import IntegrationSlot, MemoryForm
from mematlas.core.config import MemAtlasConfig
from mematlas.core.models import Modality
from mematlas.core.plugin import LLMService
from mematlas.core.registry import PluginRegistry, default_registry
from mematlas.catalog import MethodCatalog


@dataclass(frozen=True)
class TaskSpec:
    modalities: frozenset[Modality] = field(default_factory=lambda: frozenset({Modality.TEXT}))
    forms: frozenset[MemoryForm] = field(default_factory=frozenset)
    required_slots: frozenset[IntegrationSlot] = field(default_factory=lambda: frozenset({IntegrationSlot.CONTEXT}))
    allow_training: bool = False
    gpu_available: bool = False
    persistence_required: bool = False
    preferred_tags: frozenset[str] = field(default_factory=frozenset)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TaskSpec":
        return cls(
            modalities=frozenset(Modality(x) for x in data.get("modalities", ["text"])),
            forms=frozenset(MemoryForm(x) for x in data.get("forms", [])),
            required_slots=frozenset(IntegrationSlot(x) for x in data.get("required_slots", ["context"])),
            allow_training=bool(data.get("allow_training", False)),
            gpu_available=bool(data.get("gpu_available", False)),
            persistence_required=bool(data.get("persistence_required", False)),
            preferred_tags=frozenset(data.get("preferred_tags", [])),
            metadata=data.get("metadata", {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "modalities": sorted(x.value for x in self.modalities),
            "forms": sorted(x.value for x in self.forms),
            "required_slots": sorted(x.value for x in self.required_slots),
            "allow_training": self.allow_training,
            "gpu_available": self.gpu_available,
            "persistence_required": self.persistence_required,
            "preferred_tags": sorted(self.preferred_tags),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class Recommendation:
    method_id: str
    score: float
    reasons: tuple[str, ...]
    description: str = ""


@dataclass(frozen=True)
class SelectionResult:
    recommendations: tuple[Recommendation, ...]
    excluded: Mapping[str, tuple[str, ...]]


class MethodSelector:
    def __init__(self, registry: PluginRegistry | None = None):
        self.registry = registry or default_registry()

    def recommend(self, spec: TaskSpec, *, limit: int = 5) -> SelectionResult:
        eligible = []
        excluded = {}
        for plugin in self.registry.plugins():
            descriptor = plugin.descriptor
            capabilities = descriptor.capabilities
            blockers = []
            if descriptor.interface_version != 3:
                blockers.append("method requires migration to Memory interface v2")
            missing_modalities = spec.modalities - capabilities.modalities
            if missing_modalities:
                blockers.append(f"missing modalities: {sorted(x.value for x in missing_modalities)}")
            missing_slots = spec.required_slots - capabilities.integration_slots
            if missing_slots:
                blockers.append(f"missing integration slots: {sorted(x.value for x in missing_slots)}")
            if capabilities.requires_training and not spec.allow_training:
                blockers.append("training is disabled")
            if capabilities.requires_gpu and not spec.gpu_available:
                blockers.append("GPU is unavailable")
            if spec.persistence_required and not capabilities.supports_persistence:
                blockers.append("persistence is required")
            if blockers:
                excluded[descriptor.method_id] = tuple(blockers)
                continue
            form_overlap = len(spec.forms & capabilities.forms)
            tag_overlap = len(spec.preferred_tags & set(descriptor.tags))
            score = 1.0 + 0.4 * form_overlap + 0.1 * tag_overlap
            reasons = ["all hard capability constraints are satisfied"]
            if form_overlap:
                reasons.append(f"matches {form_overlap} requested memory form(s)")
            if tag_overlap:
                reasons.append(f"matches {tag_overlap} preferred tag(s)")
            eligible.append(Recommendation(
                descriptor.method_id,
                score,
                tuple(reasons),
                descriptor.summary,
            ))
        eligible.sort(key=lambda item: (-item.score, item.method_id))
        return SelectionResult(tuple(eligible[:limit]), excluded)


class ConfigCompiler:
    def __init__(self, registry: PluginRegistry | None = None):
        self.registry = registry or default_registry()

    def compile(
        self,
        method_id: str,
        *,
        method_overrides: Mapping[str, Any] | None = None,
        runtime: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        providers: Mapping[str, Any] | None = None,
        scope: Mapping[str, str] | None = None,
    ) -> MemAtlasConfig:
        plugin = self.registry.get(method_id)
        schema = plugin.descriptor.config_schema
        params = {
            name: field_schema["default"]
            for name, field_schema in schema.get("properties", {}).items()
            if "default" in field_schema
        }
        from mematlas.core.provider_config import ALIASES
        explicit = dict(method_overrides or {})
        for role, fields in ALIASES.items():
            for field_name, names in fields.items():
                if any(name in explicit for name in names) or field_name in (providers or {}).get(role, {}):
                    for name in names:
                        params.pop(name, None)
        params.update(method_overrides or {})
        runtime_values = {"executor": plugin.descriptor.recommended_executor, **dict(runtime or {})}
        config = MemAtlasConfig.from_dict({
            "method": {"method_id": method_id, "params": params},
            "runtime": runtime_values,
            "metadata": dict(metadata or {}),
            "providers": dict(providers or {}),
            "scope": dict(scope or {}),
        })
        errors = plugin.validate_config(config.method)
        if errors:
            raise ValueError("; ".join(errors))
        return config

class NaturalLanguageTaskParser:
    def __init__(self, llm: LLMService):
        self.llm = llm

    async def parse(self, request: str) -> TaskSpec:
        prompt = (
            "Convert the agent-memory requirement into one JSON object. "
            "Allowed modalities: text,image,audio,video,artifact. Allowed forms: "
            "record,graph,hierarchical,procedural,multimodal,parametric,latent,policy. "
            "Allowed required_slots: context,tools,model_hook,artifact,episode. Include "
            "allow_training, gpu_available, persistence_required, and preferred_tags.\n\n"
            f"Requirement: {request}"
        )
        output = await self.llm.complete([{"role": "user", "content": prompt}], temperature=0)
        match = re.search(r"\{.*\}", output, flags=re.S)
        if not match:
            raise ValueError("Task parser did not return a JSON object")
        return TaskSpec.from_dict(json.loads(match.group(0)))


class NaturalLanguageMethodRouter:
    """Optional LLM router over one-sentence method descriptions."""

    def __init__(self, llm: LLMService, registry: PluginRegistry | None = None):
        self.llm = llm
        self.registry = registry or default_registry()
        self.catalog = MethodCatalog(self.registry)

    async def route(
        self,
        request: str,
        *,
        limit: int = 5,
        include_controls: bool = False,
    ) -> SelectionResult:
        descriptions = self.catalog.descriptions(include_controls=include_controls)
        descriptions = [item for item in descriptions if self.registry.get(item["method_id"]).descriptor.interface_version == 3]
        if not descriptions:
            raise ValueError("No methods currently satisfy Memory interface v2; integrate a method before recommendation")
        prompt = (
            "Choose the most suitable agent-memory methods for the user request from the "
            "provided catalog. Use only listed method_id values. Return JSON in the form "
            '{"recommendations":[{"method_id":"...","reason":"..."}]}. '
            f"Return at most {limit} methods in best-first order.\n\n"
            f"Catalog: {json.dumps(descriptions, ensure_ascii=False)}\n\n"
            f"User request: {request}"
        )
        output = await self.llm.complete([{"role": "user", "content": prompt}], temperature=0)
        match = re.search(r"\{.*\}", output, flags=re.S)
        if not match:
            raise ValueError("Method router did not return a JSON object")
        payload = json.loads(match.group(0))
        candidates = payload.get("recommendations", [])
        if not isinstance(candidates, list):
            raise ValueError("Method router recommendations must be a list")
        available = {item["method_id"]: item for item in descriptions}
        selected = []
        seen = set()
        for candidate in candidates:
            if not isinstance(candidate, Mapping):
                continue
            method_id = str(candidate.get("method_id", ""))
            if method_id not in available or method_id in seen:
                continue
            seen.add(method_id)
            reason = str(candidate.get("reason") or "selected from the method description")
            selected.append(Recommendation(
                method_id=method_id,
                score=float(limit - len(selected)),
                reasons=(reason,),
                description=str(available[method_id]["description"]),
            ))
            if len(selected) >= limit:
                break
        return SelectionResult(tuple(selected), {})
