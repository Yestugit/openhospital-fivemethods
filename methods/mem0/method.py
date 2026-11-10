"""Official Mem0 vector-memory adapter for the public Memory interface."""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
import json
from urllib.parse import parse_qsl
from typing import Any, Mapping

from mematlas.core.config import MethodConfig
from mematlas.core.errors import ConfigurationError
from mematlas.core.models import MemoryContribution, MemoryEpisode, MemoryEvent, content_text
from mematlas.core.plugin import BaseMemoryMethod, MethodPlugin, MethodContext, MethodServices
from mematlas.core.provider_config import resolve_providers
from mematlas.core.scope import normalize_scope, scope_key
from mematlas.methods.common import normalize_items
from mematlas.methods.core_utils import vision_content
from mematlas.methods.services import prepare_services
from .config import _DESCRIPTOR as DESCRIPTOR
from .native import NativeBackend
from .provenance import IDENTITY, verify_source


def native_identity(scope):
    scope = normalize_scope(scope)
    if "team_id" in scope:
        raise ValueError("Native Mem0 does not expose team ownership; use user_id/agent_id/session_id/run_id")
    identity = {key: scope[key] for key in ("user_id", "agent_id") if key in scope}
    # A complete public scope is one owner. A partial user-only native filter
    # would otherwise also match a different user+agent/session owner.
    identity["run_id"] = "mematlas.scope:" + sha256(scope_key(scope).encode("utf-8")).hexdigest()
    return identity


class Mem0Memory(BaseMemoryMethod):
    def __init__(self, config: MethodConfig, services: MethodServices):
        self.config, self.params, self.services = config, dict(config.params), services
        self.backend = None
        self._seen_events: set[tuple[str, str]] = set()
        self._buffers: dict[tuple[str, str | None], list[MemoryEvent]] = defaultdict(list)

    @property
    def memory(self):
        return self.backend.memory if self.backend else None

    async def setup(self, context: MethodContext):
        report = Mem0Plugin().healthcheck(self.config, self.services)
        if not report["ready"]:
            raise ConfigurationError(f"Mem0 native requirements are not satisfied: {report}")
        prepare_services(self.params, self.services, method_id="mem0", llm=self.params.get("infer", True), embedding=True)
        providers = resolve_providers(self.params, self.services.provider_configs)
        dimensions = self.params.get("embedding_dims") or providers["embedding"].dimensions or getattr(self.services.embedder, "dimensions", None)
        if not isinstance(dimensions, int) or dimensions < 1:
            raise ConfigurationError("Mem0 native Qdrant requires providers.embedding.dimensions (or embedding_dims); no dimension-probe model call is made")
        llm_options = {"model": providers["llm"].model or getattr(self.services.llm, "model", "injected")}
        allowed = {"temperature", "top_p", "top_k", "max_tokens", "vision_details", "reasoning_effort", "is_reasoning_model"}
        llm_options.update({key: value for key, value in providers["llm"].options.items() if key in allowed})
        self.backend = NativeBackend(self.services, self.params, dimensions, llm_options)
        await self.backend.setup()

    def _message(self, event):
        text = content_text(event.content)
        images = [block["uri"] for block in event.content if block["type"] == "image"] if isinstance(event.content, list) else []
        if images and (not self.params.get("enable_vision", False) or not self.params.get("infer", True)):
            raise ValueError("Mem0 images require enable_vision=true, infer=true and a vision-capable LLM")
        if not text and not images:
            return None
        role = (event.role or "user").lower()
        role = "assistant" if role in {"assistant", "agent", "ai"} else "system" if role == "system" else "user"
        return {"role": role, "content": vision_content(text, images)}

    async def _write(self, events):
        messages = [message for event in events if (message := self._message(event)) is not None]
        if not messages:
            return
        metadata = {"mematlas_event_ids": [event.event_id for event in events],
            "mematlas_event_kinds": [event.kind.value for event in events],
            "mematlas_scope": normalize_scope(events[0].scope or {})}
        field_sources = {event.event_id: {"field": event.extensions["_unit_field"], "episode_ref": event.episode_id}
                         for event in events if "_unit_field" in event.extensions}
        if field_sources:
            metadata["mematlas_field_sources"] = field_sources
        await self.backend.call("add", messages, infer=self.params.get("infer", True),
            metadata=metadata,
            **native_identity(events[0].scope or {}))
        self._seen_events.update((scope_key(event.scope or {}), event.event_id) for event in events)

    async def _flush(self, key):
        events = self._buffers.get(key, [])
        if events:
            await self._write(events)
            del self._buffers[key]

    async def _observe(self, event):
        owner = scope_key(event.scope or {})
        if (owner, event.event_id) in self._seen_events or any(
            item.event_id == event.event_id for key, rows in self._buffers.items() if key[0] == owner for item in rows
        ):
            return
        message = self._message(event)
        if message is None:
            return
        key = owner, event.episode_id
        self._buffers[key].append(event)
        if self.params.get("write_policy", "turn") == "turn" and message["role"] == "assistant":
            await self._flush(key)

    async def _learn(self, episode: MemoryEpisode, *, observed_event_ids):
        owner = scope_key(episode.scope or {})
        source_events = list(episode.events or [])
        # Public units can carry task/output/feedback alongside their events.
        # Encode only real caller fields that the trajectory does not already
        # contain. Do not serialize applied memories, extensions or references.
        def serialized(value):
            return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)

        def value_tree(value):
            if isinstance(value, str):
                try:
                    return json.loads(value)
                except (ValueError, TypeError):
                    return value.strip()
            return value

        def contains(value, expected):
            if value == expected:
                return True
            if isinstance(value, Mapping):
                return any(contains(item, expected) for item in value.values())
            if isinstance(value, (list, tuple)):
                return any(contains(item, expected) for item in value)
            return False

        fields, aliases, covered_generated = {}, [], set()
        for field, value, role in (("task", episode.task, "user"), ("output", episode.output, "assistant"),
                                   ("feedback", episode.feedback, "assistant")):
            if value is None or isinstance(value, str) and not value.strip():
                continue
            text = serialized(value)
            expected = value_tree(text)
            identity = "mematlas:mem0:field:" + sha256(json.dumps({
                "episode_id": episode.episode_id, "field": field, "value": expected,
            }, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")).hexdigest()
            aliases.append((owner, identity))
            if (owner, identity) in self._seen_events:
                for event in source_events:
                    if event.extensions.get("_unit_field") != field:
                        continue
                    generated_value = value_tree(content_text(event.content))
                    # Public feedback-only units wrap text in a generated JSON
                    # event. A fresh write_ref must not bypass field dedup.
                    if generated_value == expected or field == "feedback" and generated_value == {"text": value}:
                        covered_generated.add(event.event_id)
            if (owner, identity) in self._seen_events or any(
                contains(value_tree(content_text(event.content)), expected) for event in source_events
            ):
                continue
            fields[field] = MemoryEvent("feedback" if field == "feedback" else "message", text,
                role=role, event_id=identity, episode_id=episode.episode_id, scope=episode.scope,
                timestamp=source_events[-1].timestamp if source_events else datetime(1970, 1, 1, tzinfo=timezone.utc),
                extensions={"_unit_field": field})
        ordered = ([fields["task"]] if "task" in fields else []) + source_events + [
            fields[field] for field in ("output", "feedback") if field in fields]
        events = [event for event in ordered if (owner, event.event_id) not in self._seen_events
                  and event.event_id not in covered_generated]
        if not events:
            self._seen_events.update(aliases)
            return
        # One submitted learning unit maps to one native add batch.
        await self._write(events)
        # Remember aliases even when their full value was already present in
        # a trajectory event, so later feedback cannot resubmit that field.
        self._seen_events.update(aliases)
        identities = {event.event_id for event in events}
        for key in list(self._buffers):
            if key[0] == owner:
                remaining = [event for event in self._buffers[key] if event.event_id not in identities]
                if remaining:
                    self._buffers[key] = remaining
                else:
                    del self._buffers[key]

    async def search(self, query, *, level_ref=None, top_k, scope, extensions):
        text = content_text(query)
        if not text.strip():
            raise ValueError("Mem0 search requires text")
        raw = await self.backend.call("search", text, filters=native_identity(scope),
            top_k=top_k if top_k is not None else self.params.get("top_k", 20),
            threshold=self.params.get("threshold", 0.1), rerank=False)
        return MemoryContribution(items=normalize_items(raw, method_id="mem0", memory_type="semantic"))

    async def _owned_item(self, memory_id, scope):
        item = await self.backend.call("get", memory_id)
        if item is None:
            raise KeyError(memory_id)
        metadata = item.get("metadata") or {}
        if any(item.get(key, metadata.get(key)) != value for key, value in native_identity(scope).items()):
            raise ValueError("Memory item belongs to a different scope")
        return item

    async def detail(self, memory_id, *, scope):
        return MemoryContribution(items=normalize_items([await self._owned_item(memory_id, scope)], method_id="mem0", memory_type="semantic"))

    async def update(self, memory_id, content, *, scope):
        await self._owned_item(memory_id, scope)
        if not isinstance(content, str):
            raise ValueError("Native Mem0 explicit update accepts text")
        await self.backend.call("update", memory_id, content)

    async def delete(self, memory_id, *, scope):
        await self._owned_item(memory_id, scope)
        await self.backend.call("delete", memory_id)

    async def flush(self):
        for key in list(self._buffers):
            await self._flush(key)

    async def reset(self, scope):
        identity = native_identity(scope)
        await self.backend.call("delete_all", **identity)
        # Native delete_all leaves recent messages and entity links. Public
        # reset must also remove these resources for the selected ownership.
        def clear_context():
            entities = self.memory.entity_store
            while rows := entities.list(filters=identity, top_k=1000)[0]:
                for row in rows:
                    entities.delete(row.id)
            with self.memory.db._lock:
                db = self.memory.db.connection
                scopes = [row[0] for row in db.execute("SELECT DISTINCT session_scope FROM messages")]
                for stored in scopes:
                    parsed = dict(parse_qsl(stored))
                    if all(parsed.get(key) == value for key, value in identity.items()):
                        db.execute("DELETE FROM messages WHERE session_scope=?", (stored,))
                db.commit()
        import asyncio
        await asyncio.to_thread(clear_context)
        def matches(owner):
            return all(native_identity(json.loads(owner)).get(key) == value for key, value in identity.items())
        self._seen_events = {row for row in self._seen_events if not matches(row[0])}
        self._buffers = defaultdict(list, {key: rows for key, rows in self._buffers.items() if not matches(key[0])})

    async def save(self) -> Mapping[str, Any]:
        return {"version": "native-wrapper-v1", "source": IDENTITY, "native": await self.backend.save(),
                "seen_events": sorted(self._seen_events),
                "buffer_events": [event.to_dict() for key in sorted(self._buffers, key=str) for event in self._buffers[key]]}

    async def load(self, state):
        if state.get("version") != "native-wrapper-v1" or state.get("source") != IDENTITY:
            raise ValueError("Mem0 native checkpoint identity mismatch; old classic/offline/reimplementation states cannot be loaded")
        seen = set()
        for owner, event_id in state["seen_events"]:
            if scope_key(json.loads(owner)) != owner or not isinstance(event_id, str) or not event_id:
                raise ValueError("Invalid native Mem0 event identity")
            seen.add((owner, event_id))
        buffers = defaultdict(list)
        buffered_ids = set()
        for row in state["buffer_events"]:
            event = MemoryEvent.from_dict(row)
            owner = scope_key(event.scope or {})
            identity = owner, event.event_id
            if identity in seen or identity in buffered_ids or self._message(event) is None:
                raise ValueError("Duplicate/invalid Mem0 buffered event")
            buffered_ids.add(identity)
            buffers[(owner, event.episode_id)].append(event)
        await self.backend.load(deepcopy(state["native"]))
        self._seen_events, self._buffers = seen, buffers

    async def close(self):
        if self.backend is not None:
            await self.backend.close()


class Mem0Plugin(MethodPlugin):
    descriptor = DESCRIPTOR

    def create_method(self, config, services):
        return Mem0Memory(config, services)

    def healthcheck(self, config, services):
        report = dict(super().healthcheck(config, services))
        report["config_errors"].extend(verify_source())
        if services.object("mem0.instance") is not None:
            report["config_errors"].append("mem0.instance legacy/custom engines are unsupported; inject LLM/embedding services into the pinned native wrapper")
        report["source_identity"] = IDENTITY
        report["ready"] = not any(report[key] for key in ("config_errors", "missing_packages", "missing_services"))
        report["status"] = "ready" if report["ready"] else "unavailable"
        return report


PLUGIN = Mem0Plugin()
