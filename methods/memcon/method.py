from __future__ import annotations

from copy import deepcopy
import logging
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

from mematlas.core.config import MethodConfig
from mematlas.core.models import EventKind, MemoryContribution, MemoryEpisode, MemoryEvent, MemoryUnit, content_text
from mematlas.core.plugin import BaseMemoryMethod, MethodContext, MethodPlugin, MethodServices, call_maybe_async
from mematlas.core.scope import normalize_scope, scope_key
from mematlas.methods.manifest import load_method_descriptor

from .core import MemConSettings, create_native, resolve_settings, scope_key as native_scope_key

DESCRIPTOR = load_method_descriptor(__file__)
LOGGER = logging.getLogger("mematlas.methods.memcon.adapter")


class MemConMemory(BaseMemoryMethod):
    def __init__(self, config: MethodConfig, services: MethodServices):
        self.params = dict(config.params)
        self.services = services
        self.settings: MemConSettings = resolve_settings(self.params, DESCRIPTOR.config_schema["x-variants"])
        self._instances: dict[str, Any] = {}
        self._scopes: dict[str, dict[str, str]] = {}
        self._initial_states: dict[str, Any] = {}
        self._active_tasks: set[str] = set()
        self._ended_tasks: set[str] = set()
        self._task_main: dict[str, str] = {}
        self._task_identity: dict[str, str] = {}
        self._event_buffers: dict[str, list[MemoryEvent]] = {}
        self._seen_events: dict[str, set[str]] = {}
        self._pending_actions: dict[str, tuple[str, dict[str, Any], list[str]]] = {}
        self._task_start_events: dict[str, str] = {}
        self._injected_instance = services.object("memcon.instance")
        self._service_factory = services.object("memcon.service_factory")

    async def setup(self, context: MethodContext) -> None:
        if self._injected_instance is None and self._service_factory is None:
            return

    def _storage_path(self, key: str) -> Path:
        root = Path(str(self.settings.values.get("storage_path") or (self.services.artifact_dir / "memcon"))).expanduser().resolve()
        path = root / "scopes" / key[:24]
        path.mkdir(parents=True, exist_ok=True)
        return self.services.register_resource(path)

    async def _service_for_scope(self, scope: Mapping[str, str]):
        owner = normalize_scope(scope)
        key = native_scope_key(owner)
        self._scopes[key] = owner
        if key in self._instances:
            return key, self._instances[key]
        if self._injected_instance is not None:
            if self._instances and key not in self._instances:
                raise ValueError("memcon.instance supports one scope; use memcon.service_factory for multiple scopes")
            memory = self._injected_instance
        elif self._service_factory is not None:
            memory = await call_maybe_async(self._service_factory, settings=self.settings, services=self.services, scope=owner, storage_path=self._storage_path(key), run_sync_in_thread=True)
        else:
            memory = create_native(self.settings, self._storage_path(key))
        self._instances[key] = memory
        self._initial_states[key] = await self._snapshot_native(memory)
        return key, memory

    async def _snapshot_native(self, memory: Any) -> Any:
        for name in ("snapshot", "export_state"):
            function = getattr(memory, name, None)
            if callable(function):
                return deepcopy(await call_maybe_async(function, run_sync_in_thread=True))
        return None

    async def _restore_native(self, memory: Any, state: Any) -> None:
        if state is None:
            return
        for name in ("restore", "import_state"):
            function = getattr(memory, name, None)
            if callable(function):
                await call_maybe_async(function, deepcopy(state), run_sync_in_thread=True)
                return
        raise ValueError("MemCon native instance does not support state restoration")

    async def _begin_task(self, scope: Mapping[str, str], task: str, task_description: str | None = None, task_identity: str | None = None, *, force_new: bool = False):
        key, memory = await self._service_for_scope(scope)
        if key in self._active_tasks and not force_new:
            previous = self._task_identity.get(key)
            if previous is None and task_identity is not None and self._task_main.get(key) == task:
                self._task_identity[key] = str(task_identity)
                return key, memory
            if (task_identity is not None and previous == str(task_identity)) or (task_identity is None and self._task_main.get(key) == task):
                return key, memory
            self._pending_actions.pop(key, None)
        self._task_start_events.pop(key, None)
        await call_maybe_async(memory.init_task_context, task, task_description, run_sync_in_thread=True)
        self._active_tasks.add(key); self._task_main[key] = task
        self._ended_tasks.discard(key)
        if task_identity is not None: self._task_identity[key] = str(task_identity)
        else: self._task_identity.pop(key, None)
        self._seen_events[key] = set()
        buffered = self._event_buffers.pop(key, [])
        for buffered_event in buffered:
            await self._apply_event(key, memory, buffered_event)
        return key, memory

    @staticmethod
    def _event_values(event: MemoryEvent) -> dict[str, Any]:
        return dict(event.extensions)

    async def add(self, unit: MemoryUnit, *, learning_episode=None) -> None:
        if learning_episode is None and unit.task:
            options = unit.extensions.get("memcon", {})
            await self._begin_task(unit.scope, unit.task, options.get("task_description"), unit.episode_ref)
        await super().add(unit, learning_episode=learning_episode)

    async def episode_end(self, episode_ref, *, scope, status, learning_episode=None) -> None:
        await super().episode_end(episode_ref, scope=scope, status=status, learning_episode=learning_episode)
        key = native_scope_key(scope)
        if key in self._active_tasks and self._task_identity.get(key) == episode_ref:
            self._ended_tasks.add(key)

    async def _observe(self, event: MemoryEvent) -> None:
        key = native_scope_key(event.scope)
        values = self._event_values(event)
        task = values.get("task") or values.get("task_main")
        if event.kind == EventKind.EPISODE_START and isinstance(task, str) and task.strip():
            if self._task_start_events.get(key) == event.event_id:
                return
            await self._begin_task(event.scope, task, values.get("task_description"), event.episode_id)
            self._task_start_events[key] = event.event_id
            return
        if key not in self._active_tasks:
            self._event_buffers.setdefault(key, []).append(event)
            return
        await self._apply_event(key, self._instances[key], event)

    async def _apply_event(self, key: str, memory: Any, event: MemoryEvent) -> None:
        seen = self._seen_events.setdefault(key, set())
        if event.event_id in seen:
            return
        seen.add(event.event_id)
        values = self._event_values(event)
        if event.kind == EventKind.ACTION:
            action = str(values.get("action") or content_text(event.content)).strip()
            if not action or action.lower().startswith(("think", "thought")):
                return
            if "observation" in values:
                if key in self._pending_actions: await self._flush_pending_action(key, memory)
                await call_maybe_async(memory.move_memory_state, action, str(values.get("observation") or ""), reward=values.get("reward", 0.0), done=values.get("done", False), run_sync_in_thread=True)
                return
            if key in self._pending_actions: await self._flush_pending_action(key, memory)
            self._pending_actions[key] = (action, values, [event.event_id]); return
        if event.kind in {EventKind.OBSERVATION, EventKind.TOOL_RESULT}:
            pending = self._pending_actions.pop(key, ("", {}, []))
            action = str(values.get("action") or pending[0])
            if not action: return
            merged = {**pending[1], **values}
            await call_maybe_async(memory.move_memory_state, action, content_text(event.content), reward=merged.get("reward", 0.0), done=merged.get("done", False), run_sync_in_thread=True)

    async def _flush_pending_action(self, key: str, memory: Any) -> None:
        pending = self._pending_actions.pop(key, None)
        if pending is None: return
        action, values, _ = pending
        await call_maybe_async(memory.move_memory_state, action, "", reward=values.get("reward", 0.0), done=values.get("done", False), run_sync_in_thread=True)

    @staticmethod
    def _record_item(value: Any, memory_type: str) -> dict[str, Any]:
        if isinstance(value, Mapping):
            raw = dict(value); content = str(raw.get("content") or raw.get("trajectory") or raw.get("task_description") or raw.get("task") or "")
            memory_id = raw.get("memory_id") or raw.get("id")
            score = raw.get("score", raw.get("relevance_score"))
            source = raw.get("source_event_ids", [])
            metadata = {k: v for k, v in raw.items() if k not in {"content", "trajectory", "memory_id", "id", "score", "relevance_score", "source_event_ids"}}
        else:
            content = str(getattr(value, "content", getattr(value, "task_trajectory", value)))
            memory_id = getattr(value, "memory_id", None); score = getattr(value, "relevance_score", None); source = []
            metadata = {"task": getattr(value, "task_main", None), "success": getattr(value, "label", None)}
        result = {"content": content, "source_event_ids": list(source or []), "memory_type": memory_type, "metadata": {"method_id": "memcon", **metadata}}
        if memory_id is not None: result["memory_id"] = str(memory_id)
        if score is not None:
            try: result["score"] = float(score)
            except (TypeError, ValueError): pass
        return result

    async def search(self, query, *, level_ref=None, top_k, scope, extensions) -> MemoryContribution:
        ext = dict(extensions or {}); method_ext = ext.get("memcon", {}) if isinstance(ext.get("memcon"), Mapping) else {}
        owner = native_scope_key(scope)
        current = self._task_main.get(owner) if owner not in self._ended_tasks else None
        task = str(method_ext.get("task") or ext.get("task") or current or content_text(query)).strip()
        if not task:
            return MemoryContribution(extensions={"memcon": {"native_item_count": 0, "native_action": None}})
        key, memory = await self._begin_task(scope, task, method_ext.get("task_description") or ext.get("task_description"), method_ext.get("episode_id") or ext.get("episode_id"), force_new=owner in self._ended_tasks)
        settings = self.settings.values
        native = await call_maybe_async(memory.retrieve_memory, query_task=content_text(query) or task, successful_topk=int(method_ext.get("successful_topk", settings.get("successful_topk", 1))), failed_topk=int(method_ext.get("failed_topk", settings.get("failed_topk", 0))), insight_topk=int(method_ext.get("insight_topk", settings.get("insight_topk", 5))), threshold=float(method_ext.get("threshold", settings.get("threshold", 0.0))), run_sync_in_thread=True)
        successful = list(native[0]) if len(native) > 0 else []; failed = list(native[1]) if len(native) > 1 else []; insights = list(native[2]) if len(native) > 2 else []
        rows = [self._record_item(item, "success_experience") for item in successful] + [self._record_item(item, "failure_experience") for item in failed] + [{"content": str(item), "source_event_ids": [], "memory_type": "insight", "metadata": {"method_id": "memcon", "kind": "insight"}} for item in insights]
        limit = int(top_k if top_k is not None else self.params.get("top_k", 5))
        rows = rows[:max(0, limit)]
        action = None
        try:
            actions = memory._mdp.get_episode_actions()
            if actions: action = str(actions[-1][1])
        except Exception: pass
        return MemoryContribution(items=rows, extensions={"memcon": {"profile": self.settings.profile, "native_item_count": len(rows), "native_action": action, "retrieval_ref": uuid4().hex, "task": task, "scope_key": key}})

    async def _learn(self, episode: MemoryEpisode, *, observed_event_ids=frozenset()) -> None:
        if not isinstance(episode.success, bool):
            raise ValueError("MemCon requires external boolean MemoryEpisode.success")
        task = episode.task or self._task_main.get(native_scope_key(episode.scope))
        if not isinstance(task, str) or not task.strip():
            raise ValueError("MemCon requires MemoryEpisode.task")
        values = dict(episode.extensions.get("memcon", {})) if isinstance(episode.extensions.get("memcon"), Mapping) else {}
        owner = native_scope_key(episode.scope)
        current_id = self._task_identity.get(owner)
        replay = owner not in self._active_tasks or not (
            current_id == episode.episode_id or current_id is None and self._task_main.get(owner) == task)
        key, memory = await self._begin_task(episode.scope, task, values.get("task_description"), episode.episode_id)
        if replay:
            # Deferred feedback can arrive after another task has begun. Replay
            # the actual retrieval decisions carried by the used contributions.
            record_action = getattr(getattr(memory, "_mdp", None), "record_action", None)
            if callable(record_action):
                for contribution in episode.applied_memory:
                    action = contribution.extensions.get("memcon", {}).get("native_action")
                    if action is not None:
                        await call_maybe_async(record_action, action, run_sync_in_thread=True)
        for event in episode.events or (): await self._apply_event(key, memory, event)
        await self._flush_pending_action(key, memory)
        await call_maybe_async(memory.save_task_context, label=episode.success, feedback=episode.feedback, run_sync_in_thread=True)
        # Preserve event provenance for the dependency-free backend and any
        # native backend exposing its last record as a mutable mapping.
        if hasattr(memory, "records") and memory.records:
            memory.records[-1]["source_event_ids"] = [event.event_id for event in (episode.events or ())]
            persist = getattr(memory, "_persist", None)
            if callable(persist): await call_maybe_async(persist, run_sync_in_thread=True)
        await call_maybe_async(memory.backward, episode.success, run_sync_in_thread=True)
        self._active_tasks.discard(key); self._task_main.pop(key, None); self._task_identity.pop(key, None); self._task_start_events.pop(key, None); self._seen_events.pop(key, None); self._pending_actions.pop(key, None)
        self._ended_tasks.discard(key)

    async def save(self) -> Mapping[str, Any]:
        instances = {}
        for key, memory in self._instances.items():
            instances[key] = {"scope": self._scopes[key], "native": await self._snapshot_native(memory)}
        return {"schema_version": 2, "method_id": "memcon", "instances": instances,
                "initial_states": deepcopy(self._initial_states), "scopes": deepcopy(self._scopes),
                "active_tasks": sorted(self._active_tasks), "ended_tasks": sorted(self._ended_tasks), "task_main": deepcopy(self._task_main),
                "task_identity": deepcopy(self._task_identity), "event_buffers": {k: [e.to_dict() for e in v] for k, v in self._event_buffers.items()},
                "seen_events": {k: sorted(v) for k, v in self._seen_events.items()},
                "pending_actions": {k: [a, m, ids] for k, (a, m, ids) in self._pending_actions.items()}, "task_start_events": deepcopy(self._task_start_events)}

    async def load(self, state: Mapping[str, Any]) -> None:
        if state.get("schema_version") != 2 or state.get("method_id") != "memcon": raise ValueError("Invalid MemCon state")
        entries = state.get("instances", {})
        for key, row in entries.items():
            scope = normalize_scope(row["scope"])
            if native_scope_key(scope) != key: raise ValueError("MemCon scope identity mismatch")
        for key in list(self._instances):
            if key not in entries:
                self._instances.pop(key, None); self._initial_states.pop(key, None)
        for key, row in entries.items():
            _, memory = await self._service_for_scope(row["scope"])
            await self._restore_native(memory, row.get("native"))
        self._initial_states = deepcopy(dict(state.get("initial_states", {}))); self._scopes = {k: normalize_scope(v) for k, v in state.get("scopes", {}).items()}
        self._active_tasks = set(state.get("active_tasks", [])); self._task_main = dict(state.get("task_main", {})); self._task_identity = dict(state.get("task_identity", {})); self._task_start_events = dict(state.get("task_start_events", {}))
        self._ended_tasks = set(state.get("ended_tasks", []))
        self._event_buffers = {k: [MemoryEvent.from_dict(row) for row in rows] for k, rows in state.get("event_buffers", {}).items()}; self._seen_events = {k: set(v) for k, v in state.get("seen_events", {}).items()}; self._pending_actions = {k: (v[0], dict(v[1]), list(v[2])) for k, v in state.get("pending_actions", {}).items()}

    async def reset(self, scope) -> None:
        key = native_scope_key(scope)
        if key in self._instances:
            memory = self._instances[key]
            reset = getattr(memory, "reset", None)
            if callable(reset): await call_maybe_async(reset, run_sync_in_thread=True)
            elif key in self._initial_states: await self._restore_native(memory, self._initial_states[key])
        self._active_tasks.discard(key); self._task_main.pop(key, None); self._task_identity.pop(key, None); self._task_start_events.pop(key, None); self._event_buffers.pop(key, None); self._seen_events.pop(key, None); self._pending_actions.pop(key, None)
        self._ended_tasks.discard(key)

    async def close(self) -> None:
        self._instances.clear(); self._initial_states.clear()


class MemConPlugin(MethodPlugin):
    descriptor = DESCRIPTOR
    qa_factory = "mematlas.methods.memcon.qa:MemConQA"
    def create_method(self, config: MethodConfig, services: MethodServices) -> BaseMemoryMethod:
        return MemConMemory(config, services)


PLUGIN = MemConPlugin()
