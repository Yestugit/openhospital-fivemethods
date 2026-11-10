"""Provider and persistence bridges for the pinned MemOS tree_text implementation.

No extraction, reconciliation, ranking or graph storage algorithm is reimplemented.
The explicit profile is upstream synchronous tree_text without the MOS scheduler.
"""
from __future__ import annotations

import asyncio
import base64
from copy import deepcopy
import json
import os
from pathlib import Path
import re
import threading
from uuid import UUID, uuid4, uuid5

from mematlas.core.models import content_text
from mematlas.core.scope import scope_key
from .utils import digest

_FACTORY_LOCK = threading.RLock()


class _LLM:
    def __init__(self, service, loop, config):
        self.service, self.loop, self.config = service, loop, config

    def generate(self, messages, **kwargs):
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        return asyncio.run_coroutine_threadsafe(self.service.complete(messages, **kwargs), self.loop).result()


class _Embedding:
    def __init__(self, service, loop, config, dimensions):
        self.service, self.loop, self.config, self.dimensions = service, loop, config, dimensions

    def embed(self, texts):
        vectors = asyncio.run_coroutine_threadsafe(self.service.embed(texts), self.loop).result()
        if len(vectors) != len(texts) or any(len(row) != self.dimensions for row in vectors):
            raise ValueError("Embedding response does not match embedding_dims or input count")
        return vectors


def _encode(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, bytes):
        return {"__neo4j_type__": "bytes", "value": base64.b64encode(value).decode()}
    # Duration is a tuple subclass; preserve its native temporal type first.
    if type(value).__module__.startswith("neo4j.time"):
        return {"__neo4j_type__": type(value).__name__, "value": value.iso_format()}
    if isinstance(value, (list, tuple)):
        return [_encode(row) for row in value]
    if isinstance(value, dict):
        return {k: _encode(v) for k, v in value.items()}
    raise TypeError(f"Unsupported Neo4j checkpoint value: {type(value).__name__}")


def _decode(value):
    if isinstance(value, list):
        return [_decode(row) for row in value]
    if isinstance(value, dict):
        if "__neo4j_type__" in value:
            kind = value["__neo4j_type__"]
            if kind == "bytes":
                return base64.b64decode(value["value"], validate=True)
            from neo4j import time
            if kind not in {"Date", "Time", "DateTime", "Duration"}:
                raise ValueError("Unsupported Neo4j temporal value")
            return getattr(time, kind).from_iso_format(value["value"])
        return {k: _decode(v) for k, v in value.items()}
    return value


def _remap(value, ids):
    """Remap native ID fields, including native JSON encoded history/source fields."""
    if isinstance(value, dict):
        return {k: _remap(v, ids) for k, v in value.items()}
    if isinstance(value, list):
        return [_remap(row, ids) for row in value]
    if isinstance(value, str):
        if value in ids:
            return ids[value]
        if value.startswith(("{", "[")):
            try:
                return json.dumps(_remap(json.loads(value), ids), ensure_ascii=False, sort_keys=True)
            except (ValueError, TypeError):
                pass
        for old, new in ids.items():
            value = value.replace("[working_binding:" + old + "]", "[working_binding:" + new + "]")
    return value


def _identifier(value):
    return "`" + value.replace("`", "``") + "`"


class MemOSCore:
    def __init__(self, params, services, run_id):
        self.params, self.services = params, services
        self.path = services.register_resource(params["storage_path"])
        self.path.mkdir(parents=True, exist_ok=True)
        self.entries = {}
        self._frozen = False
        self._provider_identity = uuid4().hex
        manifest = self.path / "native.json"
        if manifest.exists():
            data = json.loads(manifest.read_text(encoding="utf-8"))
            self.namespace, self._scopes = data["namespace"], data["scopes"]
        else:
            self.namespace, self._scopes = str(uuid4()), {}
            self._persist()

    def _persist(self):
        (self.path / "native.json").write_text(json.dumps(
            {"namespace": self.namespace, "scopes": self._scopes}, ensure_ascii=False, sort_keys=True), encoding="utf-8")

    async def setup(self):
        self.loop = asyncio.get_running_loop()
        # Eager imports surface missing extras before the first clinical write.
        await asyncio.to_thread(self._imports)
        await asyncio.to_thread(self._verify_service)
        for owner, scope in list(self._scopes.items()):
            await self._entry(scope)

    @staticmethod
    def _imports():
        from memos.memories.textual.tree import TreeTextMemory
        from memos.mem_reader.simple_struct import SimpleStructMemReader

    def _verify_service(self):
        from neo4j import GraphDatabase
        from mematlas.core.errors import ConfigurationError
        try:
            with GraphDatabase.driver(self.params["neo4j_uri"],
                    auth=(self.params["neo4j_user"], os.environ[self.params["neo4j_password_env"]]),
                    connection_timeout=5, connection_acquisition_timeout=5) as driver:
                agent = driver.get_server_info().agent
                version = re.search(r"Neo4j/(\d+)\.(\d+)", agent)
                if not version or tuple(map(int, version.groups())) < (5, 18):
                    raise ConfigurationError("MemOS requires Neo4j >=5.18 for native vector prefiltering")
                with driver.session(database=self.params["neo4j_database"]) as session:
                    session.run("RETURN 1 AS ready").consume()
                    index = session.run("SHOW INDEXES YIELD name, options WHERE name = 'memory_vector_index' "
                                        "RETURN options").single()
                    if index is not None:
                        dimensions = index["options"].get("indexConfig", {}).get("vector.dimensions")
                        if dimensions is not None and dimensions != self.params["embedding_dims"]:
                            raise ConfigurationError("MemOS native memory_vector_index has dimensions="
                                + str(dimensions) + ", but embedding_dims=" + str(self.params["embedding_dims"])
                                + ". Use a separate Neo4j instance/database with matching dimensions; "
                                "the adapter will not replace an existing shared index.")
        except ConfigurationError:
            raise
        except Exception as error:
            raise ConfigurationError("MemOS cannot reach its configured Neo4j database: "
                + type(error).__name__ + ". Check neo4j_uri/user/database and the password environment variable.") from None

    def _tenant(self, owner):
        return "mematlas_" + digest([self.namespace, owner])[:40]

    def _make(self, scope):
        from memos.configs.memory import TreeTextMemoryConfig
        from memos.configs.mem_reader import SimpleStructMemReaderConfig
        from memos.llms.factory import LLMFactory
        from memos.embedders.factory import EmbedderFactory
        from memos.memories.textual.tree import TreeTextMemory
        from memos.mem_reader.simple_struct import SimpleStructMemReader
        from memos.mem_reader.utils import count_tokens_text

        owner = scope_key(scope)
        tenant = self._tenant(owner)
        provider_identity = tenant + "_" + self._provider_identity
        # Unique config identities prevent upstream factory singletons sharing
        # providers across runs. Constructors are restored before leaving setup.
        llm = {"backend": "openai", "config": {"model_name_or_path": provider_identity,
            "api_key": "mematlas-service-bridge", "api_base": "http://127.0.0.1"}}
        embedding = {"backend": "universal_api", "config": {"provider": "openai",
            "model_name_or_path": provider_identity, "api_key": "mematlas-service-bridge",
            "embedding_dims": self.params["embedding_dims"], "max_tokens": None}}
        tree_config = TreeTextMemoryConfig.model_validate({"extractor_llm": llm,
            "dispatcher_llm": llm, "embedder": embedding, "mode": "sync", "reorganize": False,
            "graph_db": {"backend": "neo4j", "config": {"uri": self.params["neo4j_uri"],
                "user": self.params["neo4j_user"], "password": os.environ[self.params["neo4j_password_env"]],
                "db_name": self.params["neo4j_database"], "use_multi_db": False,
                "user_name": tenant, "embedding_dimension": self.params["embedding_dims"]}}})
        reader_config = SimpleStructMemReaderConfig.model_validate({"llm": llm, "embedder": embedding,
            "chunker": {"backend": "sentence", "config": {"save_rawfile": False}}})
        # Upstream declares a string although Chonkie supports a token-counting
        # callable. Use MemOS's own chat token counter; no tokenizer model download.
        reader_config.chunker.config.tokenizer_or_token_counter = count_tokens_text
        with _FACTORY_LOCK:
            old_llm = LLMFactory.backend_to_class["openai"]
            old_embedder = EmbedderFactory.backend_to_class["universal_api"]
            LLMFactory.backend_to_class["openai"] = lambda cfg: _LLM(self.services.llm, self.loop, cfg)
            EmbedderFactory.backend_to_class["universal_api"] = lambda cfg: _Embedding(
                self.services.embedder, self.loop, cfg, self.params["embedding_dims"])
            try:
                tree, reader = TreeTextMemory(tree_config), SimpleStructMemReader(reader_config)
            finally:
                LLMFactory.backend_to_class["openai"] = old_llm
                EmbedderFactory.backend_to_class["universal_api"] = old_embedder
        reader.set_graph_db(tree.graph_store)
        searcher = tree.get_searcher()
        update_usage = searcher._update_usage_history
        searcher._update_usage_history = lambda *args, **kwargs: (
            None if self._frozen else update_usage(*args, **kwargs))
        reader.set_searcher(searcher)
        return {"tree": tree, "reader": reader, "searcher": searcher,
            "scope": dict(scope), "tenant": tenant, "ids": {}}

    async def _entry(self, scope):
        owner = scope_key(scope)
        if owner not in self.entries:
            entry = await asyncio.to_thread(self._make, scope)
            self.entries[owner] = entry
            self._scopes[owner] = dict(scope)
            self._persist()
        return self.entries[owner]

    async def add(self, events, scope, episode_ref, metadata):
        entry = await self._entry(scope)
        messages = [{"role": "assistant" if e.role in {"assistant", "agent", "ai"} else
                     "system" if e.role == "system" else "user", "content": content_text(e.content),
                     "chat_time": e.timestamp.isoformat(), "message_id": e.event_id} for e in events]
        def write():
            groups = entry["reader"].get_memory([messages], type="chat", mode="fine",
                info={"user_id": entry["tenant"], "session_id": episode_ref,
                    "source_event_ids": [e.event_id for e in events], **metadata}, user_name=entry["tenant"])
            entry["tree"].add([row for group in groups for row in group], user_name=entry["tenant"])
        await asyncio.to_thread(write)

    async def search(self, query, scope, view, top_k):
        entry = self.entries.get(scope_key(scope))
        if entry is None:
            return []
        rows = await asyncio.to_thread(entry["searcher"].search, query, top_k=top_k,
            mode=self.params["search_mode"], memory_type="All" if view == "default" else view,
            user_name=entry["tenant"])
        items = []
        for row in rows:
            data = row.model_dump(mode="json") if hasattr(row, "model_dump") else dict(row)
            native_id = str(data["id"])
            logical_id = entry["ids"].get(native_id, native_id)
            details = _remap(data.get("metadata", {}), entry["ids"])
            items.append({"memory_id": logical_id, "content": data.get("memory", ""),
                **({"score": details["score"]} if details.get("score") is not None else {}), "method_id": "memos",
                "memory_type": details.get("memory_type", "tree_text"), "metadata": details})
        return items

    async def flush(self):
        # mode=sync completes extraction, merge/conflict handling and cleanup in add.
        return None

    async def freeze(self):
        self._frozen = True
        for entry in self.entries.values():
            await asyncio.to_thread(entry["searcher"]._usage_executor.shutdown, wait=True)

    async def thaw(self):
        from memos.context.context import ContextThreadPoolExecutor
        for entry in self.entries.values():
            entry["searcher"]._usage_executor = ContextThreadPoolExecutor(max_workers=4, thread_name_prefix="usage")
        self._frozen = False

    def _graph_state(self, entry):
        graph = entry["tree"].graph_store
        with graph.driver.session(database=graph.db_name) as session:
            def read(tx):
                nodes = list(tx.run("MATCH (n {user_name:$tenant}) RETURN n.id AS id, labels(n) AS labels, properties(n) AS props", tenant=entry["tenant"]))
                edges = list(tx.run("MATCH (a {user_name:$tenant})-[r]->(b {user_name:$tenant}) RETURN a.id AS source,b.id AS target,type(r) AS type,properties(r) AS props", tenant=entry["tenant"]))
                return nodes, edges
            nodes, edges = session.execute_read(read)
        ids = {str(n["id"]): entry["ids"].get(str(n["id"]), str(n["id"])) for n in nodes}
        if any(n["id"] is None for n in nodes):
            raise ValueError("MemOS checkpoint encountered a native node without an id")
        entry["ids"].update(ids)
        result_nodes = []
        for row in nodes:
            props = dict(row["props"])
            props.pop("user_name", None)
            result_nodes.append({"id": ids[str(row["id"])], "labels": sorted(row["labels"]), "props": _encode(_remap(props, ids))})
        result_edges = [{"source": ids[str(row["source"])], "target": ids[str(row["target"])],
                         "type": row["type"], "props": _encode(_remap(dict(row["props"]), ids))} for row in edges]
        return {"scope": entry["scope"], "nodes": sorted(result_nodes, key=lambda row: row["id"]),
            "edges": sorted(result_edges, key=digest), "counts": deepcopy(entry["tree"].memory_manager.current_memory_size)}

    async def export_state(self):
        return {"version": 1, "scopes": [await asyncio.to_thread(self._graph_state, self.entries[key])
                                          for key in sorted(self.entries)]}

    def _replace_graph(self, entry, state):
        graph = entry["tree"].graph_store
        logical_to_native = {row["id"]: str(uuid5(UUID(self.namespace), entry["tenant"] + row["id"])) for row in state["nodes"]}
        def write(tx):
            tx.run("MATCH (n {user_name:$tenant}) DETACH DELETE n", tenant=entry["tenant"]).consume()
            for row in state["nodes"]:
                props = _remap(_decode(row["props"]), logical_to_native)
                props["user_name"] = entry["tenant"]
                labels = ":".join(_identifier(label) for label in row["labels"])
                tx.run("CREATE (n" + (":" + labels if labels else "") + ") SET n=$props", props=props).consume()
            for row in state["edges"]:
                tx.run("MATCH (a {id:$source,user_name:$tenant}),(b {id:$target,user_name:$tenant}) CREATE (a)-[r:" +
                    _identifier(row["type"]) + "]->(b) SET r=$props", source=logical_to_native[row["source"]],
                    target=logical_to_native[row["target"]], tenant=entry["tenant"],
                    props=_remap(_decode(row["props"]), logical_to_native)).consume()
        with graph.driver.session(database=graph.db_name) as session:
            session.execute_write(write)
        entry["ids"] = {native: logical for logical, native in logical_to_native.items()}
        entry["tree"].memory_manager.current_memory_size = deepcopy(state["counts"])

    async def import_state(self, state):
        if state.get("version") != 1:
            raise ValueError("Unsupported MemOS native checkpoint")
        restored = {scope_key(row["scope"]): row for row in state["scopes"]}
        for owner, entry in list(self.entries.items()):
            if owner not in restored:
                await self.reset(entry["scope"])
                await asyncio.to_thread(entry["searcher"]._usage_executor.shutdown, wait=True)
                await asyncio.to_thread(entry["tree"].memory_manager.close)
                await asyncio.to_thread(entry["tree"].graph_store.driver.close)
                del self.entries[owner]
                self._scopes.pop(owner, None)
        for row in state["scopes"]:
            entry = await self._entry(row["scope"])
            await asyncio.to_thread(self._replace_graph, entry, row)
        self._persist()

    async def reset(self, scope):
        owner = scope_key(scope)
        entry = self.entries.get(owner)
        if entry is None:
            return
        await asyncio.to_thread(self._replace_graph, entry, {"nodes": [], "edges": [],
            "counts": {name: 0 for name in entry["tree"].memory_manager.current_memory_size}})

    async def close(self):
        for entry in self.entries.values():
            await asyncio.to_thread(entry["searcher"]._usage_executor.shutdown, wait=True)
            await asyncio.to_thread(entry["tree"].memory_manager.close)
            await asyncio.to_thread(entry["tree"].graph_store.driver.close)
