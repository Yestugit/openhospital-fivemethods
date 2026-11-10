"""MemoryOS adaptation to the public event, learning and evolution contract."""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import json
from typing import Any, Mapping

from mematlas.core.config import MethodConfig
from mematlas.core.errors import ConfigurationError
from mematlas.core.files import atomic_write_json
from mematlas.core.models import MemoryEvent, MemoryEpisode, MemoryContribution, content_text
from mematlas.core.plugin import BaseMemoryMethod, MethodPlugin, MethodServices, MethodContext, call_maybe_async
from mematlas.core.scope import normalize_scope, scope_key
from mematlas.methods.services import prepare_services

from .config import _DESCRIPTOR as DESCRIPTOR
from .core import MemoryOSCore
from .utils import safe_storage_user_id, format_contribution


class MemoryOSMemory(BaseMemoryMethod):
    def __init__(self, config: MethodConfig, services: MethodServices):
        self.config, self.services = config, services
        self.params = dict(config.params)
        self.instances: dict[str, Any] = {}
        self._injected = services.object("memoryos.instance")
        self._seen_events: set[tuple[str, str]] = set()
        self._buffers: dict[tuple[str, str | None], list[MemoryEvent]] = defaultdict(list)

    async def setup(self, context: MethodContext) -> None:
        report = PLUGIN.healthcheck(self.config, self.services)
        if not report["ready"]:
            raise ConfigurationError(f"MemoryOS requirements are not satisfied: {report}")
        if self._injected is None:
            prepare_services(self.params, self.services, method_id="memoryos", llm=True, embedding=True)

    def _get_instance(self, scope):
        key = scope_key(scope)
        if key not in self.instances:
            if self._injected is not None:
                if self.instances:
                    raise ValueError("A single memoryos.instance supports one scope")
                self.instances[key] = self._injected
            else:
                names = ("short_term_capacity", "mid_term_capacity", "long_term_knowledge_capacity",
                         "retrieval_queue_capacity", "mid_term_heat_threshold", "mid_term_similarity_threshold")
                self.instances[key] = MemoryOSCore(
                    user_id=safe_storage_user_id(key), assistant_id=scope.get("agent_id", "assistant"),
                    services=self.services, storage_path=self.params.get("storage_path"),
                    **{name: self.params[name] for name in names if name in self.params},
                )
        return self.instances[key]

    @staticmethod
    def _role(event):
        if event.kind.value != "message":
            raise ValueError("MemoryOS accepts message events")
        role = (event.role or "user").lower()
        if role in {"user", "human", "patient"}:
            return "user"
        if role in {"assistant", "agent", "ai", "doctor"}:
            return "assistant"
        if role == "system":
            return "system"
        raise ValueError("MemoryOS message role must identify the user or assistant")

    async def _write(self, users, assistant=None):
        events = [*users, *([assistant] if assistant is not None else [])]
        if not events:
            return
        memory = self._get_instance(events[0].scope)
        await call_maybe_async(
            memory.add_memory, "\n".join(content_text(event.content) for event in users),
            content_text(assistant.content) if assistant is not None else "",
            timestamp=events[-1].timestamp.isoformat(),
            meta_data={"mematlas_event_ids": [event.event_id for event in events],
                       "episode_id": events[0].episode_id},
            run_sync_in_thread=True,
        )
        owner = scope_key(events[0].scope)
        self._seen_events.update((owner, event.event_id) for event in events)

    async def _observe(self, event: MemoryEvent) -> None:
        """Pair real user/assistant messages; native ingestion keeps automatic evolution."""
        owner = scope_key(event.scope)
        if (owner, event.event_id) in self._seen_events or any(
            event.event_id == buffered.event_id
            for key, values in self._buffers.items() if key[0] == owner for buffered in values
        ):
            return
        role = self._role(event)
        if role == "system" or not content_text(event.content).strip():
            return
        key = owner, event.episode_id
        if role == "user":
            self._buffers[key].append(event)
        else:
            await self._write(self._buffers.get(key, []), event)
            self._buffers.pop(key, None)

    async def _learn(self, episode: MemoryEpisode, *, observed_event_ids: frozenset[str]) -> None:
        """Consume the complete learning unit, preserving messages already written online."""
        # Validate the whole unit before beginning native writes.
        for event in episode.events:
            self._role(event)
        owner = scope_key(episode.scope)
        pending = [event for event in episode.events if (owner, event.event_id) not in self._seen_events]
        pending_ids = {event.event_id for event in pending}
        for key in list(self._buffers):
            if key[0] == owner:
                remaining = [event for event in self._buffers[key] if event.event_id not in pending_ids]
                if remaining:
                    self._buffers[key] = remaining
                else:
                    del self._buffers[key]
        for event in pending:
            await self._observe(event)
        key = owner, episode.episode_id
        if key in self._buffers:
            # A learning unit may contain a user message without a reply.
            # Preserve the missing side as empty text, without inventing a response.
            await self._write(self._buffers[key])
            del self._buffers[key]

    async def index(self, *, scope):
        return [{"level_ref": ref, "scope": dict(scope), "description": description, "granularity": granularity}
                for ref, description, granularity in (
                    ("default", "Short-, mid- and long-term memory", "native combined context"),
                    ("short_term", "Recent conversation", "user/assistant pairs"),
                    ("mid_term", "Retrieved historical conversation", "conversation pages"),
                    ("long_term", "User profile and reusable knowledge", "profile and knowledge entries"))]

    async def search(self, query, *, level_ref=None, top_k, scope, extensions) -> MemoryContribution:
        text = content_text(query)
        if not text.strip():
            raise ValueError("MemoryOS retrieval requires a nonempty text query")
        key = scope_key(scope)
        if extensions.get("read_only") and key not in self.instances:
            return MemoryContribution()
        memory = self._get_instance(scope)
        names = ("segment_similarity_threshold", "page_similarity_threshold", "knowledge_threshold",
                 "top_k_sessions", "top_k_knowledge")
        raw = {} if level_ref == "short_term" else await call_maybe_async(
            memory.retriever.retrieve_context, user_query=text, user_id=memory.user_id,
            read_only=bool(extensions.get("read_only")),
            **{name: self.params[name] for name in names if name in self.params},
            run_sync_in_thread=True,
        )
        profile = memory.user_long_term_memory.get_raw_user_profile(memory.user_id) if level_ref in {None, "default", "long_term"} else None
        short_term = memory.short_term_memory.get_all() if level_ref in {None, "default", "short_term"} else []
        if level_ref == "mid_term":
            raw = {"retrieved_pages": raw.get("retrieved_pages", [])}
        elif level_ref == "long_term":
            raw = {key: value for key, value in raw.items() if key != "retrieved_pages"}
        result = format_contribution(short_term, raw, profile, memory.user_id, top_k)
        return MemoryContribution(items=[{**row, "level_ref": "short_term" if row["memory_type"] == "short_term"
            else "mid_term" if row["memory_type"] == "mid_term" else "long_term"} for row in result.items],
            context=result.context, extensions=result.extensions)

    async def flush(self) -> None:
        """Persist current layers and pending messages without ingesting or evolving them."""
        atomic_write_json(self.services.artifact_dir / "memoryos" / "adapter_state.json", await self.save())

    async def reset(self, scope) -> None:
        owner = scope_key(scope)
        instance = self.instances.get(owner)
        if instance is not None:
            await call_maybe_async(instance.import_state, {}, run_sync_in_thread=True)
            if instance is not self._injected:
                await call_maybe_async(instance.close, run_sync_in_thread=True)
            del self.instances[owner]
        self._buffers = defaultdict(list, {key: values for key, values in self._buffers.items() if key[0] != owner})
        self._seen_events = {identity for identity in self._seen_events if identity[0] != owner}

    async def save(self) -> Mapping[str, Any]:
        states = {key: await call_maybe_async(instance.export_state, run_sync_in_thread=True)
                  for key, instance in self.instances.items()}
        return deepcopy({"version": 2, "instances": states, "seen_events": sorted(self._seen_events),
                         "buffer_events": [event.to_dict() for values in self._buffers.values() for event in values]})

    async def load(self, state: Mapping[str, Any]) -> None:
        if state.get("version") != 2:
            raise ValueError("MemoryOS state requires version 2")
        states = state.get("instances")
        if not isinstance(states, Mapping):
            raise ValueError("MemoryOS instances must be mappings")
        scopes = {}
        for key, value in states.items():
            owner = normalize_scope(json.loads(key))
            if scope_key(owner) != key or not isinstance(value, Mapping):
                raise ValueError("Invalid MemoryOS scope state")
            scopes[key] = owner
        if self._injected is not None and len(scopes) > 1:
            raise ValueError("A single memoryos.instance supports one scope")
        seen = set()
        for owner, event_id in state["seen_events"]:
            if scope_key(normalize_scope(json.loads(owner))) != owner or not isinstance(event_id, str) or not event_id:
                raise ValueError("Invalid MemoryOS written event identity")
            seen.add((owner, event_id))
        buffers = defaultdict(list)
        buffered_ids = set()
        for row in state["buffer_events"]:
            event = MemoryEvent.from_dict(row)
            if event.scope is None or self._role(event) != "user":
                raise ValueError("Invalid MemoryOS pending message")
            identity = scope_key(event.scope), event.event_id
            if identity in seen or identity in buffered_ids:
                raise ValueError("Duplicate MemoryOS pending message")
            buffered_ids.add(identity)
            buffers[(identity[0], event.episode_id)].append(event)
        for key in list(self.instances):
            await self.reset(normalize_scope(json.loads(key)))
        for key, payload in states.items():
            await call_maybe_async(self._get_instance(scopes[key]).import_state, deepcopy(payload), run_sync_in_thread=True)
        self._seen_events, self._buffers = seen, buffers

    async def close(self) -> None:
        for instance in self.instances.values():
            if instance is not self._injected:
                await call_maybe_async(instance.close, run_sync_in_thread=True)


class MemoryOSPlugin(MethodPlugin):
    descriptor = DESCRIPTOR

    def create_method(self, config: MethodConfig, services: MethodServices) -> BaseMemoryMethod:
        return MemoryOSMemory(config, services)

    def healthcheck(self, config, services):
        report = dict(super().healthcheck(config, services))
        instance = services.object("memoryos.instance")
        if instance is not None:
            for name in ("add_memory", "export_state", "import_state"):
                if not callable(getattr(instance, name, None)):
                    report["config_errors"].append(f"memoryos.instance requires {name}()")
            for obj, name in ((getattr(instance, "retriever", None), "retrieve_context"),
                              (getattr(instance, "short_term_memory", None), "get_all"),
                              (getattr(instance, "user_long_term_memory", None), "get_raw_user_profile")):
                if not callable(getattr(obj, name, None)):
                    report["config_errors"].append(f"memoryos.instance requires {name}()")
            report["missing_packages"], report["missing_services"] = [], []
        report["ready"] = not (report["config_errors"] or report["missing_packages"] or report["missing_services"])
        report["status"] = "ready" if report["ready"] else "unavailable"
        return report


PLUGIN = MemoryOSPlugin()
