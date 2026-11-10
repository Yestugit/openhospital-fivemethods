"""Read-only native inventory, invoked only while the runtime is frozen.

Text is counted at its native object/field identity. Database/history/STM layers
are reported separately; snapshots and vectors are never treated as text.
"""
from collections import defaultdict
import asyncio
import json
from pathlib import Path
import sqlite3

_TOKENIZERS = {}


def tokenizer(directory):
    key = str(Path(directory).resolve())
    if key not in _TOKENIZERS:
        from tokenizers import Tokenizer
        _TOKENIZERS[key] = Tokenizer.from_file(str(Path(key) / "tokenizer.json"))
        _TOKENIZERS[key].no_truncation()
        _TOKENIZERS[key].no_padding()
    return _TOKENIZERS[key]


def count(text, directory):
    return len(tokenizer(directory).encode(str(text), add_special_tokens=False).ids)


def files_inventory(roots):
    files = {}
    for root in roots:
        root = Path(root)
        for p in ([root] if root.is_file() else root.rglob("*")):
            if not p.is_file() or any(part in {"snapshots", "state-transfer", "logs"} for part in p.relative_to(root.parent if root.is_file() else root).parts):
                continue
            if p.suffix in {".jsonl", ".log", ".lock"}:
                continue
            category = "wal_transactions" if p.name.endswith(("-wal", "-journal")) or "transactions" in p.parts else "database_index_cache_state"
            files[str(p.resolve())] = {"bytes": p.stat().st_size, "category": category}
    return files


async def inspect_native(method, options):
    result = await asyncio.to_thread(_inspect, method, options)
    if options["method"] == "memmachine":
        storage = method.backend.semantic._semantic_storage
        sets = [row async for row in storage.get_history_set_ids(
            min_uningested_messages=1, older_than=await storage.get_storage_time())]
        result["pending_semantic_sets"] = len(sets)
        result["background_pending"] = bool(sets or result.get("pending_stm_episodes"))
    else:
        result["background_pending"] = bool(result.get("pending_buffered_events"))
    return result


def _inspect(method, options):
    model_dir = options["tokenizer_dir"]
    method_id = options["method"]
    objects, memory_state = [], {}
    backend = getattr(method, "backend", None)
    roots = []
    def add(layer, identity, field, value):
        if isinstance(value, str) and value:
            objects.append({"layer": layer, "identity": str(identity), "field": field,
                            "tokens": count(value, model_dir)})
    def sqlite_text(path):
        # Readonly URI plus query_only forbids accidental writes/checkpoints.
        connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
        try:
            connection.execute("PRAGMA query_only=ON")
            tables = [r[0] for r in connection.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]
            for table in tables:
                quoted = '"' + table.replace('"', '""') + '"'
                cursor = connection.execute("SELECT * FROM " + quoted)
                columns = [d[0] for d in cursor.description]
                for index, row in enumerate(cursor):
                    for column, value in zip(columns, row):
                        if any(word in column.lower() for word in ("content", "text", "summary", "value", "message", "memory", "data")) and not any(word in column.lower() for word in ("vector", "embedding", "metadata", "hash", "timestamp")):
                            add(f"{path.name}:{table}", index, column, value)
        finally:
            connection.close()
    if method_id == "memos":
        roots = [backend.path, Path(options["neo4j_data_dir"]) / "databases", Path(options["neo4j_data_dir"]) / "transactions"]
        for entry in backend.entries.values():
            graph = entry["tree"].graph_store
            with graph.driver.session(database=graph.db_name) as session:
                for row in session.run("MATCH (n {user_name:$tenant}) RETURN n.id AS id, properties(n) AS props", tenant=entry["tenant"]):
                    for key in ("memory", "content", "summary"):
                        add("tree:" + str(row["props"].get("memory_type", "native")), row["id"], key, row["props"].get(key))
    elif method_id == "mem0":
        roots = [backend.root]
        for name, collection in backend.memory.vector_store.client._client.collections.items():
            for index, payload in enumerate(collection.payload):
                for key, value in payload.items():
                    if key in {"data", "memory", "text", "content"}:
                        add("collection:" + name, index, key, value)
        sqlite_text(backend.root / "history.db")
    elif method_id == "memmachine":
        roots = [backend.path]
        for path in backend.live.glob("*.sqlite"):
            sqlite_text(path)
        memory_state = {owner: backend._stm(row[2].short_term_memory, backend._summary_gates[owner].blocked)
                        for owner, row in backend._handles.items()}
        for owner, state in memory_state.items():
            add("stm_summary", owner, "summary", state.get("summary"))
            for index, episode in enumerate(state.get("episodes", [])):
                add("stm_episodes", owner + str(index), "content", episode.get("content"))
    elif method_id in {"emem", "memora"}:
        roots = [method.root]
        for scope, native in method.backends.items():
            if method_id == "memora":
                result = native.store.collection.get(include=["documents", "metadatas"])
                for identity, document, metadata in zip(result["ids"], result["documents"], result["metadatas"]):
                    layer = "chroma:" + str(metadata.get("type", metadata.get("memory_type", "native")))
                    # Upstream documents contain the natural-language index;
                    # metadata.value contains the retained memory body.
                    add(layer, scope + identity, "index", metadata.get("index", document))
                    add(layer, scope + identity, "value", metadata.get("value"))
            else:
                engine = native.engine
                from dataclasses import asdict, is_dataclass
                for name in ("edu_embedding_store", "argument_embedding_store", "session_content_store"):
                    store = getattr(engine, name)
                    for identity in store.get_all_ids():
                        content = store.get_row(identity)["content"]
                        if is_dataclass(content):
                            content = json.dumps(asdict(content), ensure_ascii=False, sort_keys=True)
                        elif not isinstance(content, str):
                            content = json.dumps(content, ensure_ascii=False, sort_keys=True, default=str)
                        add("emem:" + name, scope + str(identity), "content", content)
                # conversation_content_store is a duplicate aggregate of these
                # same sessions; report its physical bytes, not duplicate text.
                for key, value in getattr(engine, "session_id_to_summary", {}).items():
                    add("emem:session_summary", scope + str(key), "summary", value)
        memory_state = {"scopes": method.scopes}
        if method_id == "memora":
            memory_state["embedding_cache_entries"] = sum(len(native.store._embedding_cache) for native in method.backends.values())
            memory_state["embedding_cache"] = {scope:list(native.store._embedding_cache.items()) for scope,native in method.backends.items()}
    else:
        raise ValueError("No native inventory for " + method_id)
    files = files_inventory(roots)
    layers = defaultdict(int)
    identities = set()
    for row in objects:
        key = (row["layer"], row["identity"], row["field"])
        if key in identities:
            raise ValueError("Duplicate native text identity")
        identities.add(key)
        layers[row["layer"]] += row["tokens"]
    from importlib.metadata import version
    return {"tokenizer_runtime": version("tokenizers"), "memory_text_tokens": sum(layers.values()), "text_layers": dict(layers), "text_objects": objects,
            "storage_bytes": sum(row["bytes"] for row in files.values()), "storage_files": files,
            "memory_state_serialized_bytes": len(json.dumps(memory_state, default=str).encode("utf-8")),
            "pending_stm_episodes":sum(len(s.get("pending", ())) for s in memory_state.values()) if method_id == "memmachine" else 0,
            "pending_buffered_events":sum(len(e.get("events", ())) for s in method.scopes.values() for e in s.get("episodes",{}).values()) if method_id in {"emem","memora"} else 0,
            "storage_scope": "native_database_index_cache_state_including_wal_excluding_logs_and_snapshots"}
