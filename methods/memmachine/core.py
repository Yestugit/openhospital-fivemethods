"""Pinned native MemMachine: event LTM, STM summaries and semantic consolidation.

SQLite/USEARCH are upstream backends. Bridges below only supply public providers,
scope identity and complete transactional checkpoints, including live STM state.
"""
from __future__ import annotations

import asyncio
import base64
from collections import deque
from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
import sqlite3
from types import SimpleNamespace
from uuid import uuid4

from mematlas.core.models import content_text
from mematlas.core.scope import scope_key
from .utils import digest


class _SummaryGate:
    """Pause the native summarizer between batches without draining pending work."""
    def __init__(self, core, memory):
        self.core, self.memory = core, memory
        self.ready = asyncio.Event()
        self.ready.set()
        self.active_done = asyncio.Event()
        self.active_done.set()
        self.blocked = []
        self.active_task = None
        original = memory._consolidator._create_summary
        original_wait = memory._consolidator.wait_until_done

        async def create(summary, episodes):
            if asyncio.current_task() is self.active_task:
                return await original(summary, episodes)
            if core._frozen:
                self.blocked = list(episodes)
                self.ready.clear()
                await self.ready.wait()
                self.blocked = []
            self.active_done.clear()
            self.active_task = asyncio.current_task()
            try:
                from mematlas.core.activity import activity
                with activity("background", "stm_summary"):
                    return await original(summary, episodes)
            finally:
                self.active_task = None
                self.active_done.set()
        memory._consolidator._create_summary = create

        async def wait():
            if core._frozen:
                await self.pause()
            else:
                await original_wait()
        memory._consolidator.wait_until_done = wait

    async def pause(self):
        await self.active_done.wait()
        # Let the native loop reach the next batch boundary, where it parks.
        await asyncio.sleep(0)

    def resume(self):
        consolidator = self.memory._consolidator
        self.ready.set()
        if consolidator._pending_episodes and not consolidator.is_running:
            consolidator._worker_task = asyncio.create_task(consolidator._run_summary_loop())

    async def detach(self):
        consolidator = self.memory._consolidator
        if self.blocked:
            consolidator._pending_episodes = self.blocked + consolidator._pending_episodes
            consolidator._worker_task.cancel()
            await asyncio.gather(consolidator._worker_task, return_exceptions=True)
            consolidator._worker_task = None
            self.blocked = []


@dataclass(frozen=True)
class _Session:
    org_id: str
    project_id: str
    session_key: str


def _providers(services, dimensions):
    import openai
    from openai.types.chat import ChatCompletion
    from memmachine_server.common.language_model.openai_chat_completions_language_model import (
        OpenAIChatCompletionsLanguageModel, OpenAIChatCompletionsLanguageModelParams)
    from memmachine_server.common.embedder import Embedder
    from memmachine_server.common.data_types import SimilarityMetric

    client = openai.AsyncOpenAI(api_key="mematlas-service-bridge", base_url="http://127.0.0.1")

    async def create(**kwargs):
        if kwargs.get("tools") or kwargs.get("stream"):
            raise ValueError("This profile uses native direct retrieval; tool/stream LLM calls are unsupported")
        messages = kwargs.pop("messages")
        kwargs.pop("model", None)
        text = await services.llm.complete(messages, **kwargs)
        schema_name = kwargs.get("response_format", {}).get("json_schema", {}).get("name")
        if schema_name == "_SemanticFeatureUpdateRes":
            # The official feature prompt asks for a command/list, whereas its
            # OpenAI parser expects an object containing `commands`. Providers
            # without enforced JSON Schema can follow the prompt literally.
            # Wrap only validated commands; never invent or discard a feature.
            from memmachine_server.semantic_memory.semantic_llm import _SemanticFeatureUpdateRes
            from mematlas.core.activity import append
            original = text
            for repair in range(4):
                try:
                    raw = json.loads(text)
                    candidate = {"commands": raw} if isinstance(raw, list) else {"commands": [raw]} if isinstance(raw, dict) and "command" in raw else raw
                    _SemanticFeatureUpdateRes.model_validate(candidate)
                    if candidate != raw:
                        append("format_normalization", operation="semantic_commands_envelope", original=text, normalized=candidate)
                        text = json.dumps(candidate, ensure_ascii=False)
                    break
                except (ValueError, TypeError) as error:
                    append("json_repair", operation="semantic_commands_schema", attempt=repair, response=text, error=str(error))
                    if repair == 3:
                        raise
                    text = await services.llm.complete([*messages,
                        {"role":"assistant", "content":text},
                        {"role":"user", "content":"Repair only the JSON structure/types of this response to match the supplied JSON Schema. Preserve the intended commands and facts; do not invent patient data or silently drop commands. Validation error: " + str(error) + "\nRequired schema: " + json.dumps(kwargs['response_format']['json_schema']['schema'])}], **kwargs)
            if text != original:
                append("format_result", operation="semantic_commands_schema", original=original, normalized=text)
        # Keep the original upstream schema conversion, JSON repair, validation,
        # retries and response normalization in the native OpenAI model.
        return ChatCompletion(id="mematlas-" + uuid4().hex, created=0, model="public-service",
            object="chat.completion", choices=[{"index": 0, "finish_reason": "stop",
                "message": {"role": "assistant", "content": text, "refusal": None}}])

    client.chat.completions.create = create
    model = OpenAIChatCompletionsLanguageModel(OpenAIChatCompletionsLanguageModelParams(
        client=client, model="public-service"))

    class BridgeEmbedder(Embedder):
        @property
        def model_id(self):
            return str(getattr(services.embedder, "model", "public-service"))

        @property
        def dimensions(self):
            return dimensions

        @property
        def similarity_metric(self):
            return SimilarityMetric.COSINE

        async def _ingest_embed(self, inputs, max_attempts=1):
            result = await services.embedder.embed(inputs)
            if len(result) != len(inputs) or any(len(row) != dimensions for row in result):
                raise ValueError("Embedding response does not match embedding_dims or input count")
            return result

        async def _search_embed(self, queries, max_attempts=1):
            return await self._ingest_embed(queries, max_attempts)

    return model, BridgeEmbedder(), client


class MemMachineCore:
    def __init__(self, params, services, run_id):
        self.params, self.services = params, services
        self.path = services.register_resource(params.get("storage_path") or services.artifact_dir / "memmachine-native")
        self.path.mkdir(parents=True, exist_ok=True)
        self.live = self.path / "native"
        self.live.mkdir(exist_ok=True)
        self._frozen = False
        self._handles, self._scopes, self._memory, self._summary_gates = {}, {}, {}, {}
        self._producers = {}
        manifest = self.path / "native.json"
        if manifest.exists():
            data = json.loads(manifest.read_text(encoding="utf-8"))
            self._scopes, self._memory = data["scopes"], data.get("stm", {})
            self._producers = data.get("producers", {})

    async def setup(self):
        from memmachine_server.common.configuration import Configuration
        from memmachine_server.common.configuration.log_conf import LogConf
        from memmachine_server.common.resource_manager.resource_manager import ResourceManagerImpl
        from memmachine_server.main.memmachine import MemMachine
        class RuntimeLogging(LogConf):
            def apply(self):
                # MemAtlas already owns logging. Do not reset its root handlers
                # or interpret its unrelated LOG_FORMAT/LOG_PATH environment.
                return None

        config = Configuration.model_validate({"logging": {"path": str(self.path / "native.log"), "level": "WARNING"},
            "resources": {"databases": {
                "sql": {"provider": "sqlite", "config": {"path": (self.live / "relational.sqlite").as_posix()}},
                "vectors": {"provider": "sqlite_vector_store", "config": {
                    "path": (self.live / "vectors.sqlite").as_posix(), "vector_search_engine": "usearch",
                    "index_directory": (self.live / "indexes").as_posix()}}},
                "language_models": {"llm": {"provider": "openai-chat-completions", "config": {
                    "model": "public-service", "api_key": "mematlas-service-bridge"}}},
                "embedders": {"embedding": {"provider": "openai", "config": {
                    "model": "public-service", "dimensions": self.params["embedding_dims"],
                    "api_key": "mematlas-service-bridge"}}}, "rerankers": {}},
            "session_manager": {"database": "sql", "instance_cache_size": 0},
            "episode_store": {"database": "sql", "with_count_cache": False},
            "episodic_memory": {"enabled": True, "long_term_memory_enabled": True,
                "short_term_memory_enabled": True,
                "long_term_memory": {"backend": "event", "embedder": "embedding",
                    "vector_store": "vectors", "segment_store": "sql"},
                "short_term_memory": {"llm_model": "llm"}},
            "semantic_memory": {"enabled": True, "storage_backend": "vector_store",
                "feature_store": "sql", "vector_collection": "vectors", "config_database": "sql",
                "vector_dimensions": self.params["embedding_dims"], "embedding_model": "embedding",
                "llm_model": "llm", "with_config_cache": False},
            "prompt": {"default_project_categories": ["profile_prompt"],
                "default_user_categories": ["profile_prompt"]}})
        config.logging = RuntimeLogging.model_validate(config.logging.model_dump())
        self.resources = ResourceManagerImpl(config)
        model, embedder, self._client = _providers(self.services, self.params["embedding_dims"])
        self.resources.language_model_manager._resources["llm"] = model
        self.resources.embedder_manager._resources["embedding"] = embedder
        self.machine = MemMachine(config, self.resources)
        self.semantic = await self.resources.get_semantic_service()
        self.semantic._debug_fail_loudly = True
        # Instrument the actual batch, not the scheduler loop or idle sleeps.
        from memmachine_server.semantic_memory.semantic_ingestion import IngestionService
        if not getattr(IngestionService.process_set_ids, "_mematlas_activity", False):
            original_batch = IngestionService.process_set_ids
            async def timed_batch(service, set_ids):
                from mematlas.core.activity import activity
                with activity("background", "semantic_ingestion", task_ids=[str(x) for x in set_ids]):
                    return await original_batch(service, set_ids)
            timed_batch._mematlas_activity = True
            IngestionService.process_set_ids = timed_batch
        self.manager = await self.resources.get_episodic_memory_manager()
        for owner, scope in list(self._scopes.items()):
            await self._session(scope)
        if not self._frozen:
            await self.semantic.start()

    @staticmethod
    def _identity(scope):
        owner = digest(scope_key(scope))
        # All scope fields participate in every native set and session identity.
        return _Session("mematlas_" + owner, "memory_" + owner, "session_" + owner)

    async def _session(self, scope):
        owner = scope_key(scope)
        if owner not in self._handles:
            session = self._identity(scope)
            await self.machine.create_session(session.session_key)
            context = self.manager.open_episodic_memory(session.session_key)
            memory = await context.__aenter__()
            self._handles[owner] = (session, context, memory)
            self._summary_gates[owner] = _SummaryGate(self, memory.short_term_memory)
            self._scopes[owner] = dict(scope)
            if owner in self._memory:
                self._restore_stm(memory.short_term_memory, self._memory[owner])
            if self._frozen:
                await self._pause_purges()
        return self._handles[owner][0]

    async def add(self, events, scope, episode_ref, metadata):
        from memmachine_server.common.episode_store import EpisodeEntry
        session = await self._session(scope)
        entries = [EpisodeEntry(content=content_text(e.content), producer_id=str(scope.get("agent_id")
            if e.role in {"assistant", "agent", "ai"} and scope.get("agent_id") else scope.get("user_id", "user")),
            producer_role=e.role or "user", created_at=e.timestamp,
            metadata={**metadata, "episode_ref": episode_ref, "event_id": e.event_id, "event_kind": e.kind.value})
            for e in events]
        await self.machine.add_episodes(session, entries)
        owner = scope_key(scope)
        self._producers[owner] = sorted(set(self._producers.get(owner, [])) | {entry.producer_id for entry in entries})

    async def search(self, query, scope, view, top_k):
        from memmachine_common.api import MemoryType
        if scope_key(scope) not in self._handles:
            return []
        session = await self._session(scope)
        targets = [MemoryType.Episodic, MemoryType.Semantic] if view == "default" else [
            MemoryType.Episodic if view == "episodic" else MemoryType.Semantic]
        response = await self.machine.query_search(session, query=query, target_memories=targets,
            limit=top_k, agent_mode=False, set_metadata={"producer_id": str(scope.get("user_id", "user"))})
        items = []
        episodic = response.episodic_memory
        if episodic:
            for kind, rows in [("long_term_episode", episodic.long_term_memory.episodes),
                               ("short_term_episode", episodic.short_term_memory.episodes)]:
                for row in rows:
                    data = row.model_dump(mode="json")
                    items.append({"memory_id": str(row.uid), "content": row.content,
                        **({"score": row.score} if row.score is not None else {}),
                        "method_id": "memmachine", "memory_type": kind, "metadata": data})
            for summary in episodic.short_term_memory.episode_summary:
                if summary:
                    items.append({"memory_id": "summary:" + digest([scope_key(scope), summary]), "content": summary,
                        "method_id": "memmachine", "memory_type": "short_term_summary"})
        for row in response.semantic_memory or []:
            data = row.model_dump(mode="json")
            items.append({"memory_id": str(row.metadata.id or digest(data)),
                "content": f"{row.category} / {row.tag} / {row.feature_name}: {row.value}",
                "method_id": "memmachine", "memory_type": "semantic_feature", "metadata": data})
        # Native streams have independent ordering and budgets; no global re-rank/truncation.
        return items

    async def flush(self):
        if self._frozen:
            raise RuntimeError("Cannot drain native learning while frozen")
        # Explicit flush advances the upstream dirty-set eligibility only while
        # draining; normal add/search retain upstream 5-message/5-minute policy.
        old_count, old_age = self.semantic._feature_update_message_limit, self.semantic._feature_time_limit
        from datetime import timedelta
        self.semantic._feature_update_message_limit, self.semantic._feature_time_limit = 1, timedelta(0)
        try:
            async with asyncio.timeout(self.params["flush_timeout_s"]):
                while True:
                    dirty = [row async for row in self.semantic._semantic_storage.get_history_set_ids(
                        min_uningested_messages=1,
                        older_than=await self.semantic._semantic_storage.get_storage_time())]
                    if not dirty:
                        break
                    task = self.semantic._ingestion_task
                    if task is not None and task.done():
                        task.result()
                        raise RuntimeError("Native semantic ingestion stopped before draining")
                    await asyncio.sleep(0.1)
                for _, _, memory in self._handles.values():
                    await memory.short_term_memory._consolidator.wait_until_done()
        finally:
            self.semantic._feature_update_message_limit, self.semantic._feature_time_limit = old_count, old_age

    async def _pause_purges(self):
        tasks = list(self.resources._segment_store_purge_tasks)
        self.resources._segment_store_purge_tasks.clear()
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def freeze(self):
        self._frozen = True
        await self.semantic.stop()
        await self._pause_purges()
        for gate in self._summary_gates.values():
            await gate.pause()

    async def thaw(self):
        from memmachine_server.common.resource_manager.resource_manager import _purge_deleted_partitions_forever
        self._frozen = False
        for gate in self._summary_gates.values():
            gate.resume()
        if not self.resources._segment_store_purge_tasks:
            self.resources._segment_store_purge_tasks.extend(asyncio.create_task(_purge_deleted_partitions_forever(store))
                for store in self.resources._segment_stores.values())
        await self.semantic.start()

    @staticmethod
    def _stm(memory, blocked=()):
        return {"episodes": [row.model_dump(mode="json") for row in memory._memory],
            "message_len": memory._current_message_len, "episode_count": memory._current_episode_count,
            "summary": memory._consolidator._summary,
            "pending": [row.model_dump(mode="json") for row in [*blocked, *memory._consolidator._pending_episodes]]}

    @staticmethod
    def _restore_stm(memory, state):
        from memmachine_server.common.episode_store import Episode
        memory._memory = deque(Episode.model_validate(row) for row in state["episodes"])
        memory._current_message_len, memory._current_episode_count = state["message_len"], state["episode_count"]
        memory._consolidator._summary = state["summary"]
        memory._consolidator._pending_episodes = [Episode.model_validate(row) for row in state["pending"]]
        memory._consolidator._worker_task = None

    async def export_state(self):
        if not self._frozen:
            raise RuntimeError("Native checkpoint requires the runtime freeze boundary")
        from memmachine_server.common.vector_store.sqlite_vector_store import _save_collection_index
        vectors = await self.resources.get_vector_store("vectors")
        for (namespace, name), engine in vectors._search_engines.items():
            await _save_collection_index(create_session=vectors._create_session, namespace=namespace,
                name=name, search_engine=engine, path=vectors._index_path(namespace, name))
        files = {}
        for path in sorted(self.live.rglob("*")):
            if not path.is_file() or path.name.endswith(("-wal", "-shm", "-journal")):
                continue
            if path.suffix == ".sqlite":
                connection = sqlite3.connect(path)
                try:
                    raw = connection.serialize()
                finally:
                    connection.close()
            else:
                raw = path.read_bytes()
            files[path.relative_to(self.live).as_posix()] = base64.b64encode(raw).decode()
        self._memory = {owner: self._stm(row[2].short_term_memory, self._summary_gates[owner].blocked)
                        for owner, row in self._handles.items()}
        return {"version": 1, "files": files, "scopes": deepcopy(self._scopes), "stm": deepcopy(self._memory),
            "producers": deepcopy(self._producers)}

    async def import_state(self, state):
        if state.get("version") != 1:
            raise ValueError("Unsupported MemMachine native checkpoint")
        decoded = {}
        for name, raw in state["files"].items():
            target = (self.live / name).resolve()
            if not target.is_relative_to(self.live.resolve()):
                raise ValueError("Invalid native checkpoint path")
            decoded[target] = base64.b64decode(raw, validate=True)
        frozen = self._frozen
        await self._close_resources()
        # Every removal is confined to the owned native/ directory after resolution.
        for path in self.live.rglob("*"):
            if path.is_file():
                resolved = path.resolve()
                if not resolved.is_relative_to(self.live.resolve()):
                    raise ValueError("Native storage contains an external link")
                resolved.unlink()
        for path, raw in decoded.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
        self._handles, self._summary_gates = {}, {}
        self._scopes, self._memory = deepcopy(state["scopes"]), deepcopy(state["stm"])
        self._producers = deepcopy(state["producers"])
        self._frozen = frozen
        await self.setup()

    async def reset(self, scope):
        owner = scope_key(scope)
        if owner not in self._handles:
            return
        session, context, memory = self._handles.pop(owner)
        gate = self._summary_gates.pop(owner)
        await gate.detach()
        await context.__aexit__(None, None, None)
        # Native deletion methods remove only this scope's session + semantic sets.
        await self.machine._delete_session_episode_store(session.session_key)
        await self.machine._delete_session_episodic_memory(session.session_key)
        await self.machine._delete_session_semantic_memory(session)
        semantic_manager = await self.resources.get_semantic_session_manager()
        for producer in self._producers.pop(owner, []):
            await semantic_manager.delete_feature_set(session, set_metadata={"producer_id": producer})
        manager = await self.resources.get_session_data_manager()
        await manager.delete_session(session.session_key)
        self._scopes.pop(owner, None)
        self._memory.pop(owner, None)

    async def _close_resources(self):
        stopping_error = None
        try:
            await self.semantic.stop()
        except Exception as error:
            # A failed native ingestion task is already terminal. Release its
            # databases even when stop re-raises that failure.
            stopping_error = error
            self.semantic._ingestion_task = None
        for gate in self._summary_gates.values():
            await gate.detach()
        for session, context, memory in self._handles.values():
            await context.__aexit__(None, None, None)
        await self.manager.close()
        # Upstream database_manager.close gathers vector shutdown and engine
        # disposal concurrently; shutdown can reopen a just-disposed pool.
        # Order the native shutdown before disposing its engines on Windows.
        database_manager = self.resources.database_manager
        for name, store in list(database_manager.vector_stores.items()):
            await store.shutdown()
            if hasattr(store, "_sync_sqlalchemy_engine"):
                store._sync_sqlalchemy_engine.dispose()
            database_manager.vector_stores.pop(name)
        await self.resources.close()
        await self._client.close()
        if stopping_error is not None:
            raise stopping_error

    async def close(self):
        try:
            await self.freeze()
            state = await self.export_state()
            (self.path / "native.json").write_text(json.dumps(
                {"scopes": state["scopes"], "stm": state["stm"], "producers": state["producers"]},
                ensure_ascii=False, sort_keys=True), encoding="utf-8")
        finally:
            await self._close_resources()
