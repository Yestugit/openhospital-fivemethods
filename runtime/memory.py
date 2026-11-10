"""The public Memory API. Agent execution remains with the caller."""
from __future__ import annotations

import asyncio
import base64
from contextlib import asynccontextmanager
from copy import deepcopy
from dataclasses import replace
from enum import Enum
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping
from uuid import uuid4

from mematlas.core.capabilities import Operation
from mematlas.core.config import MemAtlasConfig
from mematlas.core.contracts import episode_fields, validate_episode, validate_modalities
from mematlas.core.episodes import EpisodeResult
from mematlas.core.errors import CapabilityMismatchError, ConfigurationError, MemoryStateError, MethodOperationError
from mematlas.core.files import atomic_write_json
from mematlas.core.models import MemoryContribution, MemoryEpisode, MemoryEvent, MemoryUnit, WriteReceipt, OperationTrace, content_value, utc_now
from mematlas.core.plugin import BaseMemoryMethod, MethodContext, MethodServices
from mematlas.core.provenance import json_sha256
from mematlas.core.registry import default_registry
from mematlas.core.scope import scope_key
from mematlas.core.telemetry import TraceCollector, UsageLedger, current_usage_context, usage_context, usage_scope
from mematlas.runtime.contract import MemoryContract
from mematlas.runtime.execution import default_executors
from mematlas.runtime.history import EpisodeHistory
from mematlas.runtime.workspace import allocate_workspace


class MemoryState(str, Enum):
    CREATED = "created"
    STARTED = "started"
    CLOSED = "closed"
    FAILED = "failed"


class Memory:
    def __init__(self, config, *, registry=None, services=None, executors=None):
        requested = config if isinstance(config, MemAtlasConfig) else MemAtlasConfig.from_dict(config)
        self.registry = registry or default_registry()
        self.plugin = self.registry.get(requested.method.method_id)
        if self.plugin.descriptor.interface_version != 3:
            raise ConfigurationError(f"{requested.method.method_id}: method requires migration to Memory interface v3")
        errors = self.plugin.validate_config(requested.method)
        if errors:
            raise ConfigurationError("; ".join(errors))
        self._contract = MemoryContract(requested, self.plugin.descriptor)
        self._contract.scope()
        self.executors = executors or default_executors()
        executor = self.executors.get(requested.runtime.executor)
        executor_status = executor.check(requested)
        if not executor_status.get("ready"):
            raise ConfigurationError(str(executor_status))
        self._requested_config = requested
        self.config, self.run_id = allocate_workspace(requested, self.plugin)
        supplied = services or MethodServices()
        from mematlas.core.provider_config import resolve_providers
        self.services = replace(supplied, artifact_dir=Path(self.config.runtime.artifact_dir),
            objects=dict(supplied.objects), usage=UsageLedger(**supplied.usage.pricing_spec()), resource_paths=set(),
            source_roots={**supplied.source_roots, **self.config.runtime.source_roots},
            provider_configs=resolve_providers(self.config.method.params, self.config.providers))
        self.services.usage.bind_journal(self.services.artifact_dir / "usage.jsonl", run_id=self.run_id)
        self.traces = TraceCollector()
        self.traces.bind_journal(self.services.artifact_dir / "operations.jsonl")
        self._history = EpisodeHistory(self.services.artifact_dir / "episodes.jsonl")
        self._method = None
        self._lock = asyncio.Lock()
        self._setup_lock = asyncio.Lock()
        self._read_only_depth = 0
        self._backend_pause_depth = 0
        self._frozen_state = self._frozen_history = None
        self._scopes = set()
        self.state = MemoryState.CREATED
        self.backend_status = {"state": "available", "activity": "idle", "can_continue": True}

    @staticmethod
    def check(config, **options):
        from mematlas.runtime.configuration import check_config
        return check_config(config, **options)

    @property
    def method_id(self):
        return self.config.method.method_id

    @property
    def capabilities(self):
        return self.plugin.descriptor.capabilities

    @property
    def is_read_only(self):
        return self._read_only_depth > 0

    def _scope(self, explicit=None, recorded=None):
        scope = self._contract.scope(explicit, recorded)
        keys = self._scopes | {scope_key(scope)}
        if self.capabilities.max_scopes is not None and len(keys) > self.capabilities.max_scopes:
            raise CapabilityMismatchError("Method scope limit exceeded")
        return scope

    async def _setup(self):
        async with self._setup_lock:
            if self.state == MemoryState.STARTED:
                return
            if self.state != MemoryState.CREATED:
                raise MemoryStateError(f"Memory is {self.state.value}; restore in a new Memory")
            self._method = self.executors.get(self.config.runtime.executor).create(self.config, self.plugin, self.services)
            from mematlas.providers.instrumentation import instrument_services
            instrument_services(self.services)
            try:
                for operation in self.capabilities.operations:
                    if operation in {Operation.INDEX, Operation.ADD, Operation.EPISODE_END, Operation.FLUSH, Operation.NATIVE_RUN, Operation.TRAIN}:
                        continue
                    implementation = getattr(type(self._method), operation.value, None)
                    if implementation is None or implementation is getattr(BaseMemoryMethod, operation.value, None):
                        raise ConfigurationError(f"Method declares {operation.value} but does not implement it")
                await self._invoke("setup", lambda: self._method.setup(MethodContext(
                    self.run_id, self.config.method, self.config.runtime, self.config.metadata)))
            except BaseException:
                self._failed_write()
                try:
                    await asyncio.wait_for(self._method.close(), 5)
                except BaseException:
                    self.services.logger.debug("Failed to release partial method setup", exc_info=True)
                raise
            instrument_services(self.services)
            self.state = MemoryState.STARTED

    async def _invoke(self, operation, callback, *, episode_id=None):
        started_at, started, operation_id = utc_now(), perf_counter(), uuid4().hex
        context = {**current_usage_context(), "parent_operation_id": current_usage_context().get("operation_id"), "operation": operation, "operation_id": operation_id, "run_id": self.run_id}
        if episode_id is not None:
            context["case_id"] = episode_id
        error = None
        try:
            from mematlas.core.activity import activity
            category = "foreground" if operation in {"search", "add", "episode_end", "flush", "index", "detail"} else "persistence"
            with activity(category, operation, method=self.method_id, **{k:v for k,v in context.items() if k != "operation"}), usage_scope(self.services.usage, "memory"), usage_context(**context):
                timeout = self.config.runtime.operation_timeout_s
                return await asyncio.wait_for(callback(), timeout) if timeout else await callback()
        except BaseException as exc:
            error = exc
            if isinstance(exc, (asyncio.CancelledError, TimeoutError)) or getattr(self._method, "_broken", False):
                self.state = MemoryState.FAILED
                guarantees = self.executors.get(self.config.runtime.executor).guarantees
                self.backend_status = {"state": "uncertain", "can_continue": False,
                    "activity": "may_continue" if guarantees.backend_may_continue else "worker_stopped",
                    "operation_id": operation_id, "recovery": "load_in_new_memory"}
            if isinstance(exc, (asyncio.CancelledError, MemoryStateError, CapabilityMismatchError)):
                raise
            raise MethodOperationError(self.method_id, operation, str(exc), operation_id=operation_id,
                                       backend_status=dict(self.backend_status)) from exc
        finally:
            self.traces.add(OperationTrace(self.method_id, operation, started_at, (perf_counter() - started) * 1000,
                error is None, error_type=type(error).__name__ if error else None,
                error_message=str(error) if error else None, metadata={**context, "backend_status": dict(self.backend_status)}))

    def _usable(self):
        if self.state not in {MemoryState.CREATED, MemoryState.STARTED}:
            raise MemoryStateError(f"Memory is {self.state.value}; restore in a new Memory")

    def _writable(self):
        self._usable()
        if self.is_read_only:
            raise MemoryStateError("Memory updates are forbidden during read_only()")

    def _failed_write(self):
        self.state = MemoryState.FAILED
        if self.backend_status["state"] != "uncertain":
            self.backend_status = {"state": "uncertain", "activity": "idle", "can_continue": False,
                                   "recovery": "load_in_new_memory"}

    def _check_input(self, operation, extensions):
        errors = self.plugin.validate_input(operation, extensions)
        if errors:
            raise CapabilityMismatchError("; ".join(errors))

    def _reference(self, memory_id, scope, level_ref="default"):
        payload = json.dumps({"method": self.method_id, "scope": scope, "id": memory_id,
                              "level": level_ref}, sort_keys=True, ensure_ascii=False).encode()
        return "mem:" + base64.urlsafe_b64encode(payload).decode().rstrip("=")

    def _resolve_reference(self, ref, scope):
        if not isinstance(ref, str) or not ref.startswith("mem:"):
            raise ValueError("Expected a memory_ref returned by search() or detail()")
        try:
            raw = ref[4:]
            data = json.loads(base64.urlsafe_b64decode(raw + "=" * (-len(raw) % 4)))
        except Exception as exc:
            raise ValueError("Invalid memory_ref") from exc
        if data.get("method") != self.method_id or data.get("scope") != scope:
            raise ValueError("Memory reference belongs to a different method or scope")
        if not isinstance(data.get("id"), str) or not data["id"]:
            raise ValueError("Memory reference has no item identity")
        return data

    def _referenced(self, contribution, scope, level_ref="default"):
        if not isinstance(contribution, MemoryContribution):
            raise TypeError("Memory reads must return MemoryContribution")
        items = []
        for row in contribution.items:
            item = deepcopy(row)
            item["level_ref"] = item.get("level_ref", level_ref)
            if item.get("memory_id") is not None:
                item["memory_ref"] = self._reference(item["memory_id"], scope, item["level_ref"])
            items.append(item)
        return MemoryContribution(items=items, context=contribution.context, extensions=contribution.extensions)

    async def _read(self, operation, callback):
        snapshot = await self._capture_state() if self.is_read_only and self.capabilities.read_only_mode == "snapshot" else None
        try:
            return await self._invoke(operation, callback)
        finally:
            if snapshot is not None and self.state == MemoryState.STARTED:
                await self._restore_state(snapshot)

    async def index(self, *, scope=None):
        self._usable()
        self._contract.require(Operation.INDEX)
        async with self._lock:
            await self._setup()
            owner = self._scope(scope)
            levels = await self._read("index", lambda: self._method.index(scope=owner))
            if not isinstance(levels, list) or any(not isinstance(row, Mapping) or not row.get("level_ref") for row in levels):
                raise TypeError("index() must return level descriptions with level_ref")
            return deepcopy(levels)

    async def search(self, query, *, level_ref=None, top_k=None, scope=None, extensions=None):
        self._usable()
        self._contract.require(Operation.SEARCH)
        query = content_value(query)
        validate_modalities(self.capabilities, query)
        if top_k is not None and (type(top_k) is not int or top_k < 1):
            raise ValueError("top_k must be a positive integer")
        extra = deepcopy(dict(extensions or {}))
        self._check_input("search", extra)
        async with self._lock:
            await self._setup()
            owner = self._scope(scope)
            if level_ref is not None:
                levels = await self._read("index", lambda: self._method.index(scope=owner))
                if level_ref not in {row["level_ref"] for row in levels}:
                    raise CapabilityMismatchError(f"Unknown or unsupported memory level: {level_ref}")
            if self.is_read_only:
                extra["read_only"] = True
            result = await self._read("search", lambda: self._method.search(query,
                level_ref=level_ref, top_k=top_k, scope=owner, extensions=extra))
            if not self.is_read_only:
                self._scopes.add(scope_key(owner))
            return self._referenced(result, owner, level_ref or "default")

    async def detail(self, memory_ref, *, scope=None):
        self._usable()
        self._contract.require(Operation.DETAIL)
        async with self._lock:
            await self._setup()
            owner = self._scope(scope)
            ref = self._resolve_reference(memory_ref, owner)
            result = await self._read("detail", lambda: self._method.detail(ref["id"], scope=owner))
            return self._referenced(result, owner, ref["level"])

    async def _write_call(self, operation, callback, *, episode_ref=None, history_before=None):
        snapshot = None
        invoked = False
        try:
            snapshot = await self._capture_state() if {Operation.SAVE, Operation.LOAD}.issubset(self.capabilities.operations) else None
            invoked = True
            return await self._invoke(operation, callback, episode_id=episode_ref)
        except BaseException:
            restored = not invoked
            if snapshot is not None and self.state == MemoryState.STARTED:
                try:
                    await self._restore_state(snapshot)
                    restored = await self._capture_state() == snapshot
                except BaseException:
                    pass
            if restored and history_before is not None:
                self._history.restore(history_before)
                self._history.persist()
            if not restored:
                self._failed_write()
            raise

    async def update(self, memory_ref, content, *, scope=None):
        self._writable()
        self._contract.require(Operation.UPDATE)
        content = content_value(content)
        validate_modalities(self.capabilities, content)
        async with self._lock:
            self._writable()
            await self._setup()
            owner = self._scope(scope)
            ref = self._resolve_reference(memory_ref, owner)
            await self._write_call("update", lambda: self._method.update(ref["id"], content, scope=owner))

    async def delete(self, memory_ref, *, scope=None):
        self._writable()
        self._contract.require(Operation.DELETE)
        async with self._lock:
            self._writable()
            await self._setup()
            owner = self._scope(scope)
            ref = self._resolve_reference(memory_ref, owner)
            await self._write_call("delete", lambda: self._method.delete(ref["id"], scope=owner))

    def _normalize_unit(self, unit, owner, explicit):
        known = self._history.units.get(self._history.identity(unit.write_ref, owner))
        refs = set()
        for ref in unit.related_refs:
            if ref.startswith("mem:"):
                self._resolve_reference(ref, owner)
            else:
                original = self._history.units.get(self._history.identity(ref, owner))
                if original is None:
                    raise ValueError(f"Unknown submission reference: {ref}")
                refs.add(original["episode_ref"])
        if explicit is not None and unit.episode_ref is not None and explicit != unit.episode_ref:
            raise ValueError("Conflicting episode references")
        ref = explicit or unit.episode_ref or (known["episode_ref"] if known else None)
        if ref is None and unit.type == "feedback" and refs:
            if len(refs) != 1:
                raise ValueError("Feedback for multiple tasks requires an explicit episode_ref")
            ref = next(iter(refs))
        ref = ref or self._history.active_episodes.get(scope_key(owner)) or uuid4().hex
        if not isinstance(ref, str) or not ref:
            raise ValueError("episode_ref must be nonempty text")
        if unit.type == "feedback" and refs - {ref}:
            raise ValueError("Feedback references belong to a different task")
        events = []
        for event in unit.events:
            self._check_input("event", event.extensions)
            if event.scope is not None and event.scope != owner:
                raise ValueError("Event belongs to a different scope")
            if event.episode_id is not None and event.episode_id not in {ref, unit.write_ref}:
                raise ValueError("Event belongs to a different task")
            event = replace(event, scope=owner, episode_id=ref)
            self._contract.event(event)
            events.append(event)

        def generated_timestamp(event_id):
            if known:
                for key in known["event_ids"]:
                    previous = self._history.events[key]
                    if previous["event_id"] == event_id:
                        return previous["timestamp"]
            return events[-1].timestamp if events else utc_now()

        if not events and (unit.output is not None or unit.task):
            content = unit.output if isinstance(unit.output, str) else json.dumps(unit.output, ensure_ascii=False) if unit.output is not None else unit.task
            events.append(MemoryEvent("message", content, role="assistant" if unit.output is not None else "user",
                event_id=unit.write_ref + ":content", episode_id=ref, scope=owner,
                timestamp=generated_timestamp(unit.write_ref + ":content"),
                extensions={"_unit_field": "output" if unit.output is not None else "task"}))
        accepts_feedback = not self.capabilities.accepted_events or any(
            kind.value == "feedback" for kind in self.capabilities.accepted_events)
        if accepts_feedback and unit.feedback and (not events or set(unit.feedback) - {"success", "reward", "text"}) and not any(event.kind.value == "feedback" for event in events):
            events.append(MemoryEvent("feedback", json.dumps(dict(unit.feedback), ensure_ascii=False), role="assistant",
                event_id=unit.write_ref + ":feedback", episode_id=ref, scope=owner,
                timestamp=generated_timestamp(unit.write_ref + ":feedback"), extensions={"_unit_field": "feedback"}))
        if len({event.event_id for event in events}) != len(events):
            raise ValueError("Duplicate event ID in memory unit")
        for event in events:
            self._contract.event(event)
        return replace(unit, events=events, episode_ref=ref, scope=owner)

    def _merge_unit(self, unit):
        key = self._history.identity(unit.episode_ref, unit.scope)
        previous = self._history.episode(key) if key in self._history.episodes else MemoryEpisode(
            unit.episode_ref, events=[], scope=unit.scope)
        if unit.task and previous.task and unit.task != previous.task:
            raise ValueError("Units in one episode must describe the same task")
        events = {event.event_id: event for event in previous.events}
        for event in unit.events:
            if event.event_id in events and events[event.event_id] != event:
                raise ValueError("An event ID cannot be reused for different event data")
            events[event.event_id] = event
        extensions = deepcopy(dict(previous.extensions))
        for name, value in unit.extensions.items():
            extensions[name] = {**extensions[name], **value} if isinstance(extensions.get(name), dict) and isinstance(value, Mapping) else deepcopy(value)
        contributions = {json_sha256(row.to_dict()): row for row in previous.applied_memory}
        contributions.update({json_sha256(row.to_dict()): row for row in unit.applied_memory})
        texts = list(dict.fromkeys(text for text in (previous.feedback, unit.feedback.get("text")) if text))
        return replace(previous, task=unit.task or previous.task, events=list(events.values()),
            output=unit.output if unit.output is not None else previous.output,
            success=unit.feedback.get("success", previous.success), reward=unit.feedback.get("reward", previous.reward),
            feedback="\n".join(texts) or None, applied_memory=list(contributions.values()), extensions=extensions)

    def _ready_episode(self, episode):
        required = self.capabilities.required_episode_fields
        available = episode_fields(episode)
        if any(not available[name] for name in required):
            return None
        validate_episode(self.capabilities, episode)
        self._check_input("add", episode.extensions)
        return episode

    def _learning_result(self, episode, *, submitted):
        feedback = "valid" if episode.success is not None or episode.reward is not None or episode.feedback else (
            "pending" if self.capabilities.required_episode_fields & {"success", "reward"} else "not_required")
        return EpisodeResult(self._history.identity(episode.episode_id, episode.scope),
            case_id=episode.episode_id, feedback=feedback, learning="submitted" if submitted else "pending",
            metadata={"scope": episode.scope})

    async def add(self, unit: MemoryUnit, *, episode_ref=None, scope=None):
        self._writable()
        self._contract.require(Operation.ADD)
        if not isinstance(unit, MemoryUnit):
            raise TypeError("add requires MemoryUnit")
        self._check_input("add", unit.extensions)
        async with self._lock:
            self._writable()
            owner = self._scope(scope, unit.scope)
            unit = self._normalize_unit(unit, owner, episode_ref)
            from mematlas.core.activity import append
            append("submission", method=self.method_id, unit=unit.to_dict())
            identity = self._history.identity(unit.write_ref, owner)
            task_key = self._history.identity(unit.episode_ref, owner)
            ended = task_key in self._history.ended
            before = self._history.snapshot()
            # Validate retries before the task-closed guard.
            if identity in self._history.units:
                if self._history.unit(identity) != unit:
                    raise ValueError("A write_ref cannot be reused for different submission data")
                if identity in self._history.processed_units:
                    append("accepted", method=self.method_id, unit=unit.to_dict())
                    return WriteReceipt(unit.write_ref, unit.episode_ref)
            if ended and unit.type not in {"feedback", "experience"}:
                raise ValueError("An ended task only accepts related experience or feedback units")
            previous = self._history.results.get(task_key)
            whole_task = bool(self.capabilities.required_episode_fields)
            if whole_task and previous and previous.learning == "submitted":
                raise ValueError("This task has already been learned; use a supported update for corrections")
            aggregate = self._merge_unit(unit)
            self._history.validate_events(unit.events)
            await self._setup()
            candidate = self._ready_episode(aggregate) if whole_task and ended else None
            if not whole_task:
                candidate = unit.as_episode()
            self._history.record_unit(unit)
            self._history.record_episode(aggregate)
            if not ended:
                active = self._history.active_episodes.get(scope_key(owner))
                if active is None:
                    self._history.active_episodes[scope_key(owner)] = unit.episode_ref
            await self._write_call("add", lambda: self._method.add(unit, learning_episode=candidate),
                episode_ref=unit.episode_ref, history_before=before)
            self._history.mark_processed(identity)
            self._history.mark_observed(self._history.event_identity(event) for event in unit.events)
            submitted = candidate is not None or bool(previous and previous.learning == "submitted")
            self._history.record_result(self._learning_result(aggregate, submitted=submitted))
            self._history._record("task_state", active_episodes=self._history.active_episodes, ended=self._history.ended)
            self._history.persist()
            self._scopes.add(scope_key(owner))
            append("accepted", method=self.method_id, unit=unit.to_dict())
            return WriteReceipt(unit.write_ref, unit.episode_ref)

    async def episode_end(self, episode_ref=None, *, status="completed", scope=None):
        self._writable()
        self._contract.require(Operation.EPISODE_END)
        if status not in {"completed", "aborted", "cancelled"}:
            raise ValueError("status must be completed, aborted or cancelled")
        async with self._lock:
            self._writable()
            owner = self._scope(scope)
            ref = episode_ref if episode_ref is not None else self._history.active_episodes.get(scope_key(owner))
            if not isinstance(ref, str) or not ref:
                raise ValueError("No active task; supply episode_ref")
            key = self._history.identity(ref, owner)
            if key in self._history.ended:
                if self._history.ended[key]["status"] != status:
                    raise ValueError("An ended task cannot change status")
                return None
            before = self._history.snapshot()
            episode = self._history.episode(key) if key in self._history.episodes else MemoryEpisode(ref, events=[], scope=owner)
            previous = self._history.results.get(key)
            candidate = self._ready_episode(episode) if self.capabilities.required_episode_fields and not (
                previous and previous.learning == "submitted") else None
            await self._setup()
            self._history.record_episode(episode)
            await self._write_call("episode_end", lambda: self._method.episode_end(ref, scope=owner,
                status=status, learning_episode=candidate), episode_ref=ref, history_before=before)
            self._history.ended[key] = {"status": status}
            self._scopes.add(scope_key(owner))
            if self._history.active_episodes.get(scope_key(owner)) == ref:
                self._history.active_episodes.pop(scope_key(owner))
            submitted = candidate is not None or bool(previous and previous.learning == "submitted")
            self._history.record_result(self._learning_result(episode, submitted=submitted))
            self._history._record("task_state", active_episodes=self._history.active_episodes, ended=self._history.ended)
            self._history.persist()

    def _complete_episode(self, episode, scope):
        owner = self._scope(scope, episode.scope)
        existing = self._history.episode_events(episode.episode_id, owner)
        events = existing if episode.events is None else episode.events
        normalized = []
        seen = set()
        for event in events:
            if event.episode_id is not None and event.episode_id != episode.episode_id:
                raise ValueError("Event belongs to a different episode")
            if event.scope is not None and event.scope != owner:
                raise ValueError("Event belongs to a different scope")
            event = replace(event, scope=owner, episode_id=episode.episode_id)
            if event.event_id in seen:
                raise ValueError("Duplicate event ID in episode")
            seen.add(event.event_id)
            normalized.append(event)
        if {row.event_id for row in existing} - seen:
            raise ValueError("Explicit episode.events omitted recorded trajectory events")
        return replace(episode, scope=owner, events=normalized)

    async def flush(self):
        self._writable()
        async with self._lock:
            self._writable()
            await self._setup()
            await self._flush()

    async def _flush(self):
        if Operation.FLUSH in self.capabilities.operations:
            try:
                await self._invoke("flush", self._method.flush)
            except BaseException:
                self._failed_write()
                raise

    def _signature(self):
        storage = self.plugin.descriptor.storage
        ignored = set(storage.get("path_parameters", {})) | set(storage.get("namespace_parameters", ()))
        return json_sha256({"method": self.method_id, "interface_version": 3,
            "params": {key: value for key, value in self._requested_config.method.params.items() if key not in ignored},
            "providers": self._requested_config.to_dict()["providers"], "scope": dict(self.config.scope)})

    async def _capture_state(self):
        self._contract.require(Operation.SAVE)
        async with self._paused_backend():
            from mematlas.core.state_files import pack_state
            state = deepcopy(await self._invoke("save", self._method.save))
            return state if self.config.runtime.executor == "service" else pack_state(state, self.services.artifact_dir)

    async def _restore_state(self, state):
        self._contract.require(Operation.LOAD)
        try:
            async with self._paused_backend():
                # Isolated executors resolve file references in their worker,
                # keeping large payloads off the request pipe as well.
                from mematlas.core.state_files import unpack_state
                payload = state if self.config.runtime.executor in {"subprocess", "service"} else unpack_state(state)
                await self._invoke("load", lambda: self._method.load(deepcopy(payload)))
        except BaseException:
            self._failed_write()
            raise

    async def _freeze_backend(self):
        if self._backend_pause_depth:
            self._backend_pause_depth += 1
            return
        try:
            await self._invoke("_freeze", self._method._freeze)
            self._backend_pause_depth = 1
        except BaseException:
            # Even a partial freeze can leave the scheduler in an unknown state.
            self._failed_write()
            try:
                await self._invoke("_thaw", self._method._thaw)
            except BaseException:
                self.services.logger.debug("Failed to release partial freeze", exc_info=True)
            raise

    async def _thaw_backend(self):
        if self._backend_pause_depth > 1:
            self._backend_pause_depth -= 1
            return
        self._backend_pause_depth = 0
        try:
            await self._invoke("_thaw", self._method._thaw)
        except BaseException:
            self._failed_write()
            raise

    @asynccontextmanager
    async def _paused_backend(self):
        # Caller holds _lock. Nested snapshots reuse the existing scheduler pause.
        await self._freeze_backend()
        try:
            yield
        finally:
            await self._thaw_backend()

    async def save(self, path):
        self._usable()
        self._contract.require(Operation.SAVE)
        async with self._lock:
            await self._setup()
            async with self._paused_backend():
                payload = {"schema_version": 4, "method_id": self.method_id, "config_signature": self._signature(),
                    "state": await self._capture_state(), "history": self._history.snapshot(), "scopes": sorted(self._scopes),
                    "usage_records": self.services.usage.to_dicts(), "pricing": self.services.usage.pricing_spec()}
            payload["payload_sha256"] = json_sha256(payload)
            atomic_write_json(path, payload)

    async def load(self, path):
        self._writable()
        self._contract.require(Operation.LOAD)
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        digest = payload.pop("payload_sha256", None)
        if payload.get("schema_version") != 4:
            raise ValueError("Memory save requires schema version 4; old checkpoints require migration")
        if digest != json_sha256(payload):
            raise ValueError("Memory save checksum mismatch")
        if payload["method_id"] != self.method_id or payload["config_signature"] != self._signature():
            raise ValueError("Memory save method/configuration does not match")
        history = EpisodeHistory()
        history.restore(payload["history"])
        scopes = set(payload["scopes"])
        for key in scopes:
            owner = json.loads(key)
            if scope_key(owner) != key:
                raise ValueError("Invalid saved scope identity")
            self._contract.scope(owner)
        if self.capabilities.max_scopes is not None and len(scopes) > self.capabilities.max_scopes:
            raise ValueError("Saved state exceeds method scope limit")
        if payload["pricing"] != self.services.usage.pricing_spec():
            raise ValueError("Memory save pricing/currency does not match the current ledger")
        from mematlas.core.telemetry import UsageRecord
        usage_records = [UsageRecord(**row) for row in payload["usage_records"]]
        async with self._lock:
            self._writable()
            await self._setup()
            try:
                async with self._paused_backend():
                    await self._restore_state(payload["state"])
            except BaseException:
                self._failed_write()
                raise
            self._history.restore(history.snapshot())
            self._history.persist()
            self._scopes = scopes
            for record in usage_records:
                self.services.usage.add(record, imported=True)

    async def reset(self, *, scope=None):
        self._writable()
        self._contract.require(Operation.RESET)
        async with self._lock:
            self._writable()
            owner = self._scope(scope)
            await self._setup()
            try:
                await self._invoke("reset", lambda: self._method.reset(owner))
            except BaseException:
                self._failed_write()
                raise
            self._history.reset(owner)
            self._scopes.discard(scope_key(owner))

    @asynccontextmanager
    async def read_only(self):
        self._usable()
        mode = self.capabilities.read_only_mode
        if mode == "unsupported":
            raise CapabilityMismatchError("Method does not support frozen evaluation")
        async with self._lock:
            await self._setup()
            if not self.is_read_only:
                await self._freeze_backend()
                try:
                    snapshot = None
                    if {Operation.SAVE, Operation.LOAD}.issubset(self.capabilities.operations):
                        snapshot = await self._capture_state()
                    self._frozen_state = snapshot
                    self._frozen_history = self._history.snapshot()
                except BaseException:
                    await self._thaw_backend()
                    raise
            self._read_only_depth += 1
        try:
            yield self
        finally:
            async with self._lock:
                restored_violation = False
                try:
                    if self._read_only_depth == 1 and self.state == MemoryState.STARTED:
                        changed_history = self._history.snapshot() != self._frozen_history
                        if changed_history:
                            self._history.restore(self._frozen_history)
                            self._history.persist()
                        changed_state = self._frozen_state is not None and await self._capture_state() != self._frozen_state
                        if changed_state:
                            await self._restore_state(self._frozen_state)
                        if changed_history or changed_state and mode == "native":
                            restored_violation = True
                            raise MemoryStateError("Frozen evaluation changed persistent memory or history; original state restored")
                except MemoryStateError:
                    # A detected violation whose original state was restored is
                    # reported to the evaluator; existing retry semantics apply.
                    if not restored_violation:
                        self._failed_write()
                    raise
                except BaseException:
                    # An unverified frozen run must not resume learning on an
                    # uncertain backend, even when releasing its scheduler works.
                    self._failed_write()
                    raise
                finally:
                    self._read_only_depth -= 1
                    if not self.is_read_only:
                        self._frozen_state = self._frozen_history = None
                        await self._thaw_backend()

    async def command(self, name, payload):
        self._usable()
        self._contract.require(Operation.COMMAND)
        commands = self.plugin.descriptor.commands
        if name not in commands:
            raise CapabilityMismatchError(f"Unknown method command {name!r}; see describe()")
        if self.is_read_only and name not in self.capabilities.read_only_commands:
            raise MemoryStateError("Command updates are forbidden during read_only()")
        from mematlas.core.validation import validate_parameters
        errors = validate_parameters(payload, commands[name].get("input_schema", {"type": "object"}),
                                     path=f"{self.method_id}.command.{name}")
        if errors:
            raise ValueError("; ".join(errors))
        async with self._lock:
            if self.is_read_only and name not in self.capabilities.read_only_commands:
                raise MemoryStateError("Command updates are forbidden during read_only()")
            await self._setup()
            try:
                return await self._invoke(f"command:{name}", lambda: self._method.command(name, deepcopy(dict(payload))))
            except BaseException:
                self._failed_write()
                raise

    def stats(self):
        return {"method_id": self.method_id, "run_id": self.run_id, "state": self.state.value,
            "read_only": self.is_read_only, "backend_status": dict(self.backend_status),
            "execution_guarantees": self.executors.get(self.config.runtime.executor).guarantees.to_dict(),
            "recorded_events": len(self._history.events), "submitted_units": len(self._history.units),
            "ended_episodes": len(self._history.ended), "episodes": self._history.counts(),
            "operations": self.traces.summary(), "usage": self.services.usage.summary()}

    async def _close(self):
        async with self._lock:
            await self._close_locked()

    async def _close_locked(self):
        if self.state == MemoryState.CLOSED:
            return
        try:
            if self.state == MemoryState.STARTED and not self.is_read_only:
                await self._flush()
        finally:
            try:
                if self._method is not None:
                    await self._invoke("close", self._method.close)
            finally:
                self.state = MemoryState.CLOSED

    async def __aenter__(self):
        await self._setup()
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        await self._close()
