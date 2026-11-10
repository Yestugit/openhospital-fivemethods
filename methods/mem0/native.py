"""Lifecycle and complete native database snapshots, without algorithm reimplementation."""
from __future__ import annotations

import asyncio
from collections import defaultdict
import base64
from contextvars import copy_context
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
from uuid import uuid4
import zlib

from mematlas.core.errors import ConfigurationError
from .provenance import IDENTITY, verify_source

SNAPSHOT_SCHEMA = "mematlas.mem0.native.snapshot-file.v1"


class NativeBackend:
    def __init__(self, services, params, dimensions, llm_options):
        self.services, self.params, self.dimensions = services, params, dimensions
        self.root = Path(params.get("storage_path") or services.artifact_dir / "mem0-native").resolve()
        self.snapshots = self.root.parent / "snapshots" / self.root.name
        self.llm_options = llm_options
        self.memory = None
        self.token = uuid4().hex

    async def setup(self):
        errors = verify_source()
        if errors:
            raise ConfigurationError("; ".join(errors))
        marker = self.root / "native-identity.json"
        if self.root.exists() and any(self.root.iterdir()) and not marker.is_file():
            raise ConfigurationError("Refusing unmarked/legacy Mem0 storage; choose a new storage_path. Historical databases are not migrated.")
        expected = {"source": IDENTITY, "snapshot_schema": SNAPSHOT_SCHEMA, "dimensions": self.dimensions}
        if marker.is_file() and json.loads(marker.read_text(encoding="utf-8")) != expected:
            raise ConfigurationError("Mem0 storage identity/dimensions mismatch; use a new storage_path")
        self.root.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps(expected, sort_keys=True), encoding="utf-8")
        self.services.register_resource(self.root)
        self.snapshots.mkdir(parents=True, exist_ok=True)
        self.services.register_resource(self.snapshots)
        # Configure before the first native import: upstream otherwise writes
        # ~/.mem0 and enables remote telemetry. Neither affects its algorithm.
        if "mem0" not in sys.modules:
            os.environ["MEM0_DIR"] = str(self.root / "native-config")
            os.environ["MEM0_TELEMETRY"] = "False"
        else:
            from mem0.memory.telemetry import MEM0_TELEMETRY
            if MEM0_TELEMETRY:
                raise ConfigurationError("Mem0 was imported with telemetry enabled; run this adapter in its isolated interpreter")
        from .providers import Binding, _BINDINGS
        self.binding = Binding(self.services, asyncio.get_running_loop())
        _BINDINGS[self.token] = self.binding
        try:
            await asyncio.to_thread(self._open)
        except Exception:
            await self.close()
            raise

    def _open(self):
        from mem0 import Memory
        from mem0.configs.base import MemoryConfig
        from mem0.utils.factory import EmbedderFactory, LlmFactory
        from mem0.utils.spacy_models import get_nlp_full, get_nlp_lemma
        from .providers import BridgeLlmConfig
        LlmFactory.register_provider("mematlas_native", "mematlas.methods.mem0.providers.NativeLLM", BridgeLlmConfig)
        EmbedderFactory.provider_to_class["mematlas_native"] = "mematlas.methods.mem0.providers.NativeEmbedder"
        config = MemoryConfig(
            vector_store={"provider": "qdrant", "config": {
                "collection_name": "mem0", "embedding_model_dims": self.dimensions,
                "path": str(self.root / "qdrant"), "on_disk": True}},
            history_db_path=str(self.root / "history.db"),
            llm={"provider": "openai", "config": {"binding": self.token, **self.llm_options,
                "enable_vision": bool(self.params.get("enable_vision", False))}},
            embedder={"provider": "openai", "config": {"model": self.token, "embedding_dims": self.dimensions}},
        )
        # Upstream factories support provider registration, while their config
        # validators have a fixed whitelist. Change only transport selection
        # after validation; Memory itself and its algorithm stay unmodified.
        config.llm.provider = config.embedder.provider = "mematlas_native"
        self.memory = Memory(config)
        if get_nlp_full() is None or get_nlp_lemma() is None:
            raise ConfigurationError("The official en_core_web_sm pipeline could not load; simplified NLP fallback is forbidden")
        # Prewarm both native encoders/collections at setup so frozen reads
        # cannot mutate the database through upstream lazy initialization.
        for store in (self.memory.vector_store, self.memory.entity_store):
            if store._get_bm25_encoder() is None or store._encode_bm25("native setup verification") is None:
                raise ConfigurationError("Native BM25 encoder failed to initialize; hybrid fallback is forbidden")

    async def call(self, name, *args, **kwargs):
        self.binding.context = copy_context()
        self.binding.failures.clear()
        result = await asyncio.to_thread(getattr(self.memory, name), *args, **kwargs)
        # Native catches some provider failures. Surface them rather than
        # reporting a successful empty result in an efficiency experiment.
        if self.binding.failures:
            raise self.binding.failures[0]
        return result

    def _files(self):
        # All operations are serialized by the Memory executor. Native writes
        # commit synchronously; these SQLite files have no pending WAL state.
        with self.memory.db._lock:
            self.memory.db.connection.commit()
            files = {"history.db": (self.root / "history.db").read_bytes()}
        qdrant = self.root / "qdrant"
        # Qdrant local persists original float64 provider vectors but searches
        # newly inserted vectors as float32. Its disk-only reload also compacts
        # tombstones and changes internal ordering. Preserve live arrays and
        # their dtypes/ordering so frozen rollback is bit-for-bit faithful.
        def array(value):
            return {"dtype": str(value.dtype), "shape": list(value.shape), "values": value.tolist()}
        live = {}
        for name, collection in self.memory.vector_store.client._client.collections.items():
            live[name] = {
                "vectors": {key: array(value) for key, value in collection.vectors.items()},
                "multivectors": {key: [array(value) for value in rows] for key, rows in collection.multivectors.items()},
                "sparse_vectors": {key: [value.model_dump(mode="json") for value in rows] for key, rows in collection.sparse_vectors.items()},
                "sparse_vectors_idf": {key: dict(value) for key, value in collection.sparse_vectors_idf.items()},
                "payload": collection.payload, "ids": list(collection.ids.items()), "ids_inv": collection.ids_inv,
                "deleted": array(collection.deleted),
                "deleted_per_vector": {key: array(value) for key, value in collection.deleted_per_vector.items()},
            }
        files["qdrant-live.json"] = json.dumps(live, sort_keys=True, separators=(",", ":")).encode("utf-8")
        for path in sorted(qdrant.rglob("*")):
            if path.is_file() and path.name != ".lock":
                if path.suffix in {"-wal", "-journal"} or path.name.endswith(("-wal", "-journal")):
                    raise RuntimeError("Unexpected uncheckpointed native SQLite journal")
                files[path.relative_to(self.root).as_posix()] = path.read_bytes()
        return files

    async def save(self):
        files = await asyncio.to_thread(self._files)
        payload = json.dumps({name: base64.b64encode(data).decode("ascii") for name, data in sorted(files.items())},
                             sort_keys=True, separators=(",", ":")).encode("ascii")
        digest = sha256(payload).hexdigest()
        packed = zlib.compress(payload, 9)
        path = self.snapshots / (digest + ".json.zlib")
        if path.exists():
            if path.read_bytes() != packed:
                raise ValueError("Content-addressed Mem0 snapshot file was changed")
        else:
            # File name and content are deterministic; create exclusively to
            # avoid overwriting another checkpoint or a concurrent writer.
            try:
                with path.open("xb") as stream:
                    stream.write(packed)
            except FileExistsError:
                if path.read_bytes() != packed:
                    raise ValueError("Conflicting native snapshot content")
        return {"schema": SNAPSHOT_SCHEMA, "source": IDENTITY, "dimensions": self.dimensions,
                "sha256": digest, "snapshot_file": str(path), "size_bytes": len(packed),
                "file_sha256": sha256(packed).hexdigest()}

    async def load(self, state):
        if state.get("schema") != SNAPSHOT_SCHEMA or state.get("source") != IDENTITY or state.get("dimensions") != self.dimensions:
            raise ValueError("Only this pinned native Mem0 checkpoint is supported; legacy/reimplementation snapshots are rejected")
        path = Path(state["snapshot_file"])
        packed = path.read_bytes()
        if len(packed) != state["size_bytes"] or sha256(packed).hexdigest() != state["file_sha256"]:
            raise ValueError("Mem0 snapshot file checksum/size mismatch")
        payload = zlib.decompress(packed)
        if sha256(payload).hexdigest() != state["sha256"]:
            raise ValueError("Mem0 checkpoint checksum mismatch")
        files = {name: base64.b64decode(value, validate=True) for name, value in json.loads(payload).items()}
        if not {"history.db", "qdrant/meta.json", "qdrant-live.json"}.issubset(files):
            raise ValueError("Incomplete native databases")
        for name in files:
            path = PurePosixPath(name)
            if path.is_absolute() or ".." in path.parts or "\\" in name or ":" in name or not (
                name in {"history.db", "qdrant-live.json"} or name.startswith("qdrant/") and path.name in {"meta.json", "storage.sqlite"}
            ):
                raise ValueError("Invalid native checkpoint path")
        await asyncio.to_thread(self._replace_files, files)

    def _close_memory(self):
        if self.memory is not None:
            memory, self.memory = self.memory, None
            memory.close()
            memory.vector_store.client.close()
            # The native base provider configs allocate transport clients even
            # though actual calls use the shared public services.
            for client in (memory.llm.config.http_client, memory.embedding_model.config.http_client):
                if client is not None:
                    client.close()

    def _replace_files(self, files):
        self._close_memory()
        # Only replace adapter-owned targets, after resolving their boundaries.
        for name in ("qdrant", "history.db"):
            target = (self.root / name).resolve()
            if target.parent != self.root or target.name != name:
                raise ValueError("Native storage target escaped its owned directory")
            if target.is_dir():
                shutil.rmtree(target)
            elif target.exists():
                target.unlink()
        for name, data in files.items():
            if name == "qdrant-live.json":
                continue
            target = self.root.joinpath(*PurePosixPath(name).parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        self._open()
        import numpy as np
        from qdrant_client.models import SparseVector
        def array(value):
            return np.array(value["values"], dtype=value["dtype"]).reshape(value["shape"])
        for name, row in json.loads(files["qdrant-live.json"]).items():
            collection = self.memory.vector_store.client._client.collections[name]
            collection.vectors = {key: array(value) for key, value in row["vectors"].items()}
            collection.multivectors = {key: [array(value) for value in rows] for key, rows in row["multivectors"].items()}
            collection.sparse_vectors = {key: [SparseVector(**value) for value in rows] for key, rows in row["sparse_vectors"].items()}
            collection.sparse_vectors_idf = {key: defaultdict(int, {int(index): value for index, value in values.items()})
                                             for key, values in row["sparse_vectors_idf"].items()}
            collection.payload, collection.ids_inv = row["payload"], row["ids_inv"]
            collection.ids = dict(row["ids"])
            collection.deleted = array(row["deleted"])
            collection.deleted_per_vector = {key: array(value) for key, value in row["deleted_per_vector"].items()}

    async def close(self):
        await asyncio.to_thread(self._close_memory)
        if "mematlas.methods.mem0.providers" in sys.modules:
            from .providers import _BINDINGS
            _BINDINGS.pop(self.token, None)
