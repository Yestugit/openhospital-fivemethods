from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

from mematlas.core.capabilities import (
    IntegrationSlot,
    MemoryForm,
    MethodCapabilities,
    MethodDescriptor,
    Operation,
)
from mematlas.core.config import MethodConfig
from mematlas.core.models import (
    ContentBlock,
    DecisionContext,
    EventKind,
    MemoryContribution,
    MemoryEvent,
    MemoryItem,
    MemoryScope,
    Modality,
)
from mematlas.core.plugin import BaseMemoryMethod, MethodPlugin, MethodContext, MethodServices


TEXT_EVENTS = frozenset({
    EventKind.MESSAGE,
    EventKind.OBSERVATION,
    EventKind.ACTION,
    EventKind.TOOL_RESULT,
    EventKind.FEEDBACK,
})
BASELINE_OPERATIONS = frozenset({
    Operation.OBSERVE,
    Operation.RETRIEVE,
    Operation.LEARN,
    Operation.COMMIT,
    Operation.RESET,
    Operation.SAVE_STATE,
    Operation.LOAD_STATE,
})


def _descriptor(
    method_id: str,
    name: str,
    summary: str,
    *,
    accepted_events: frozenset[EventKind],
    config_schema: Mapping[str, Any] | None = None,
) -> MethodDescriptor:
    return MethodDescriptor(
        method_id=method_id,
        name=name,
        summary=summary,
        capabilities=MethodCapabilities(
            modalities=frozenset({Modality.TEXT}),
            forms=frozenset({MemoryForm.RECORD}),
            operations=BASELINE_OPERATIONS,
            integration_slots=frozenset({IntegrationSlot.CONTEXT, IntegrationSlot.EPISODE}),
            accepted_events=accepted_events,
            supports_persistence=True,
            read_only_mode="native",
        ),
        storage={"kind": "local_files", "path_parameters": {"path": "file"}} if method_id == "baseline_markdown" else {"kind": "memory"},
        fidelity="reference_baseline",
        implementation_strategy="mematlas_native",
        internal_token_tracking="none",
        implementation_notes=("Internal comparison control; no paper implementation is claimed.",),
        config_schema=config_schema or {"type": "object", "properties": {}, "additionalProperties": False},
        tags=("baseline", "control"),
    )


class NoMemory(BaseMemoryMethod):
    async def observe(self, event: MemoryEvent) -> None:
        return None

    async def retrieve(self, context: DecisionContext) -> MemoryContribution:
        return MemoryContribution.empty(baseline="none")


class NoMemoryPlugin(MethodPlugin):
    descriptor = _descriptor(
        "baseline_none",
        "No Memory",
        "Control that never stores or retrieves cross-step memory.",
        accepted_events=TEXT_EVENTS,
    )

    def create_method(self, config: MethodConfig, services: MethodServices) -> BaseMemoryMethod:
        return NoMemory()


class MarkdownReflectionMemory(BaseMemoryMethod):
    def __init__(self, config: MethodConfig, services: MethodServices) -> None:
        self.params = dict(config.params)
        configured = self.params.get("path")
        self.path = Path(configured).expanduser().resolve() if configured else services.artifact_dir / "memory.md"
        services.register_resource(self.path)
        self.max_notes = int(self.params.get("max_notes", 3))
        self._entries: list[dict[str, str]] = []

    async def setup(self, context: MethodContext) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.is_file():
            self._load()
        else:
            self._write()

    def _load(self) -> None:
        text = self.path.read_text(encoding="utf-8")
        entries = []
        for raw in text.split("\n## ")[1:]:
            lines = raw.strip().splitlines()
            if not lines:
                continue
            entries.append({"event_id": lines[0].strip(), "text": "\n".join(lines[1:]).strip()})
        self._entries = entries

    def _write(self) -> None:
        blocks = ["# MemAtlas Markdown Reflection Baseline"]
        blocks.extend(f"## {item['event_id']}\n\n{item['text']}" for item in self._entries)
        self.path.write_text("\n\n".join(blocks).rstrip() + "\n", encoding="utf-8")

    async def observe(self, event: MemoryEvent) -> None:
        if not event.text:
            return
        self._entries = [item for item in self._entries if item["event_id"] != event.event_id]
        self._entries.append({"event_id": event.event_id, "text": event.text})
        self._write()

    async def retrieve(self, context: DecisionContext) -> MemoryContribution:
        limit = int(context.top_k or self.max_notes)
        selected = list(reversed(self._entries[-limit:]))
        return MemoryContribution.from_items([
            MemoryItem(
                content=(ContentBlock.text(item["text"]),),
                memory_id=item["event_id"],
                source_event_ids=(item["event_id"],),
                memory_type="reflection",
            )
            for item in selected
        ], metadata={"baseline": "markdown_reflection"})

    async def reset(self, scope: MemoryScope | None = None) -> None:
        self._entries.clear()
        self._write()

    async def save_state(self) -> Mapping[str, Any]:
        return {"entries": list(self._entries)}

    async def load_state(self, state: Mapping[str, Any]) -> None:
        self._entries = [dict(item) for item in state.get("entries", [])]
        self._write()


class MarkdownReflectionPlugin(MethodPlugin):
    descriptor = _descriptor(
        "baseline_markdown",
        "Markdown Reflection",
        "OpenHospital-style recent reflection notes stored in one Markdown file.",
        accepted_events=frozenset({EventKind.FEEDBACK}),
        config_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "max_notes": {"type": "integer", "minimum": 1, "default": 3},
            },
            "additionalProperties": False,
        },
    )

    def create_method(self, config: MethodConfig, services: MethodServices) -> BaseMemoryMethod:
        return MarkdownReflectionMemory(config, services)


def _terms(text: str) -> list[str]:
    lowered = text.casefold()
    latin = re.findall(r"[a-z0-9]+", lowered)
    cjk = re.findall(r"[\u3400-\u9fff]", lowered)
    return latin + cjk


class PersistentTextIndexMemory(BaseMemoryMethod):
    def __init__(self, config: MethodConfig, services: MethodServices) -> None:
        self.params = dict(config.params)
        configured = self.params.get("path")
        self.path = Path(configured).expanduser().resolve() if configured else services.artifact_dir / "index.json"
        self.entries: list[dict[str, Any]] = []

    async def setup(self, context: MethodContext) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.is_file():
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            self.entries = list(payload.get("entries", []))

    def _save(self) -> None:
        self.path.write_text(
            json.dumps({"entries": self.entries}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    async def reset(self, scope: MemoryScope | None = None) -> None:
        if scope is None:
            self.entries.clear()
        else:
            self.entries = [item for item in self.entries if item["scope"] != scope.key()]
        self._save()

    async def save_state(self) -> Mapping[str, Any]:
        return {"entries": list(self.entries)}

    async def load_state(self, state: Mapping[str, Any]) -> None:
        self.entries = [dict(item) for item in state.get("entries", [])]
        self._save()


class BM25Memory(PersistentTextIndexMemory):
    async def observe(self, event: MemoryEvent) -> None:
        if not event.text:
            return
        self.entries = [item for item in self.entries if item["event_id"] != event.event_id]
        self.entries.append({
            "event_id": event.event_id,
            "scope": event.scope.key(),
            "text": event.text,
            "terms": _terms(event.text),
            "kind": event.kind.value,
        })
        self._save()

    async def retrieve(self, context: DecisionContext) -> MemoryContribution:
        entries = [item for item in self.entries if item["scope"] == context.scope.key()]
        query = _terms(context.text)
        if not entries or not query:
            return MemoryContribution.empty(baseline="bm25")
        document_frequency = Counter()
        for item in entries:
            document_frequency.update(set(item["terms"]))
        average_length = sum(len(item["terms"]) for item in entries) / len(entries)
        k1 = float(self.params.get("k1", 1.5))
        b = float(self.params.get("b", 0.75))
        scored = []
        for item in entries:
            counts = Counter(item["terms"])
            score = 0.0
            for term in query:
                frequency = counts[term]
                if not frequency:
                    continue
                inverse_document_frequency = math.log(
                    1 + (len(entries) - document_frequency[term] + 0.5) / (document_frequency[term] + 0.5)
                )
                denominator = frequency + k1 * (
                    1 - b + b * len(item["terms"]) / max(average_length, 1)
                )
                score += inverse_document_frequency * frequency * (k1 + 1) / denominator
            if score > 0:
                scored.append((score, item))
        scored.sort(key=lambda value: value[0], reverse=True)
        top_k = int(context.top_k or self.params.get("top_k", 5))
        return MemoryContribution.from_items([
            MemoryItem(
                content=(ContentBlock.text(item["text"]),),
                memory_id=item["event_id"],
                score=score,
                source_event_ids=(item["event_id"],),
                memory_type=item["kind"],
            )
            for score, item in scored[:top_k]
        ], metadata={"baseline": "bm25"})


class BM25Plugin(MethodPlugin):
    descriptor = _descriptor(
        "baseline_bm25",
        "BM25 Memory",
        "Persistent lexical retrieval control over observed text events.",
        accepted_events=TEXT_EVENTS,
        config_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "top_k": {"type": "integer", "minimum": 1, "default": 5},
                "k1": {"type": "number", "minimum": 0, "default": 1.5},
                "b": {"type": "number", "minimum": 0, "maximum": 1, "default": 0.75},
            },
            "additionalProperties": False,
        },
    )

    def create_method(self, config: MethodConfig, services: MethodServices) -> BaseMemoryMethod:
        return BM25Memory(config, services)


def _cosine(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("Embedding dimensions differ")
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    return numerator / (left_norm * right_norm) if left_norm and right_norm else 0.0


class VectorMemory(PersistentTextIndexMemory):
    def __init__(self, config: MethodConfig, services: MethodServices) -> None:
        super().__init__(config, services)
        if services.embedder is None:
            raise ValueError("baseline_vector requires MethodServices.embedder")
        self.services = services

    @property
    def embedder(self):
        return self.services.embedder

    async def observe(self, event: MemoryEvent) -> None:
        if not event.text:
            return
        vector = (await self.embedder.embed([event.text]))[0]
        self.entries = [item for item in self.entries if item["event_id"] != event.event_id]
        self.entries.append({
            "event_id": event.event_id,
            "scope": event.scope.key(),
            "text": event.text,
            "embedding": list(vector),
            "kind": event.kind.value,
        })
        self._save()

    async def retrieve(self, context: DecisionContext) -> MemoryContribution:
        entries = [item for item in self.entries if item["scope"] == context.scope.key()]
        if not entries or not context.text:
            return MemoryContribution.empty(baseline="vector")
        query = (await self.embedder.embed([context.text]))[0]
        scored = sorted(
            ((_cosine(list(query), list(item["embedding"])), item) for item in entries),
            key=lambda value: value[0],
            reverse=True,
        )
        top_k = int(context.top_k or self.params.get("top_k", 5))
        return MemoryContribution.from_items([
            MemoryItem(
                content=(ContentBlock.text(item["text"]),),
                memory_id=item["event_id"],
                score=score,
                source_event_ids=(item["event_id"],),
                memory_type=item["kind"],
            )
            for score, item in scored[:top_k]
        ], metadata={"baseline": "vector"})


class VectorPlugin(MethodPlugin):
    descriptor = replace(_descriptor(
        "baseline_vector",
        "Vector Memory",
        "Persistent cosine-similarity retrieval control using a supplied embedder.",
        accepted_events=TEXT_EVENTS,
        config_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "top_k": {"type": "integer", "minimum": 1, "default": 5},
            },
            "additionalProperties": False,
        },
    ), requirements={"services": ["embedding"]})

    def create_method(self, config: MethodConfig, services: MethodServices) -> BaseMemoryMethod:
        from mematlas.methods.services import prepare_services
        prepare_services(config.params, services, method_id="baseline_vector", llm=False, embedding=True)
        return VectorMemory(config, services)


PLUGINS = (
    NoMemoryPlugin(),
    MarkdownReflectionPlugin(),
    BM25Plugin(),
    VectorPlugin(),
)
