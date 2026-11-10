"""Original Memora algorithms, with narrow public provider/storage bridges."""
from __future__ import annotations

from functools import partial
import ast
import importlib
import importlib.util
import json
import re
from pathlib import Path
from threading import RLock
from types import SimpleNamespace
import uuid

from mematlas.core.models import content_text
from mematlas.core.provider_config import resolve_providers
from .utils import (canonical, checked_source, digest, get_blob, namespace, put_blob,
                    run_service, storage_path, put_json_array_blob)

REVISION = "dec3f8f2444eace7004fc084abe1be9f3d88270e"
_IMPORT_LOCK = RLock()
_SERVICES = {}
_FAILURES = {}


class ServiceCompletions:
    """OpenAI create/parse response contract at the public provider boundary."""
    def __init__(self, cfg):
        services = _SERVICES[cfg._mematlas_service_id]
        self.service = services.llm
        self.io_loop = getattr(services, "_native_io_loop", None)
        self.failures = _FAILURES[cfg._mematlas_service_id]

    def create(self, **kwargs):
        messages = kwargs.pop("messages")
        if kwargs.get("response_format", {}).get("type") in {"json_object", "json_schema"}:
            # Qwen's compatible endpoint requires this explicit keyword for
            # JSON mode; preserve the official task and requested object shape.
            messages = [*messages, {"role":"user", "content":"Return the requested result as valid JSON matching the response schema."}]
        try:
            result = run_service(self.service.complete(messages, **kwargs), self.io_loop)
        except BaseException as exc:
            self.failures.append(exc)
            raise
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=result))])

    def parse(self, **kwargs):
        from openai.lib._parsing._completions import type_to_response_format_param
        model = kwargs.pop("response_format")
        kwargs["response_format"] = type_to_response_format_param(model)
        messages = list(kwargs.pop("messages"))
        schema = json.dumps(model.model_json_schema(), ensure_ascii=False)
        # Some compatible endpoints accept response_format but do not enforce
        # it. Include its unchanged schema and validate with the native model.
        messages.append({"role":"user", "content":"Return JSON matching this response schema: " + schema})
        for repair in range(4):
            result = self.create(messages=messages, **kwargs)
            message = result.choices[0].message
            try:
                raw = json.loads(message.content)
                candidate = raw
                if set(model.model_fields) == {"entries"} and not (isinstance(raw, dict) and "entries" in raw):
                    candidate = {"entries":raw if isinstance(raw,list) else [raw]}
                parsed = model.model_validate(candidate)
                # Native schema permits a null updated_index (no key change),
                # but its writer unconditionally hashes it. Preserve exactly
                # the chosen original candidate key in that nullable case.
                if model.__name__ == "MemoryUpdateDecision" and parsed.should_update:
                    prompt = messages[0].get("content", "") if messages else ""
                    keys = {int(index):key for index,key in re.findall(
                        r"(?m)^Candidate (\d+):\s*\n- Similarity Score:[^\n]*\n- Index: ([^\n]+)", prompt)}
                    if parsed.best_candidate_index not in keys:
                        raise ValueError("An update requires a valid best_candidate_index from the original candidates")
                    if not isinstance(parsed.updated_value, str) or not parsed.updated_value.strip():
                        raise ValueError("An update requires nonempty updated_value from the original facts")
                    if parsed.updated_index is None:
                        parsed = parsed.model_copy(update={"updated_index":keys[parsed.best_candidate_index]})
                        candidate = parsed.model_dump()
                message.parsed = parsed
                if candidate != raw:
                    from mematlas.core.activity import append
                    append("format_normalization", operation="memora_entries_envelope", original=message.content, normalized=candidate)
                return result
            except (ValueError, TypeError) as error:
                from mematlas.core.activity import append
                append("json_repair", operation="memora_native_schema", attempt=repair, response=message.content, error=str(error))
                if repair == 3:
                    self.failures.append(error)
                    raise
                messages.extend([{"role":"assistant", "content":message.content},
                    {"role":"user", "content":"Repair only JSON structure and types to match the schema above; preserve the facts, do not invent or silently discard entries. Validation error: " + str(error)}])


def _native_chat_class(root):
    """Execute official invoke/response handling; select public transport only.

    Removes precisely the unused torch, transformers, azure.identity top-level
    import statements. HF/managed-identity functions are never called. Every
    original method body, including prompt formatting and exception handling,
    remains unmodified.
    """
    path = root / "src" / "memora" / "utils" / "llm.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    tree.body = [node for node in tree.body if not (
        isinstance(node, ast.Import) and any(alias.name == "torch" for alias in node.names)
        or isinstance(node, ast.ImportFrom) and node.module in {"transformers", "azure.identity"})]
    module = {"__name__": "memora.utils._native_llm", "__file__": str(path)}
    exec(compile(tree, str(path), "exec"), module)

    class PublicChatCompletionModel(module["ChatCompletionModel"]):
        def __init__(self, cfg, token_usage_callback=None):
            self.cfg = cfg
            self.token_usage_callback = None
            self.model_type = "azure"
            completion = ServiceCompletions(cfg)
            self.client = SimpleNamespace(chat=SimpleNamespace(completions=completion),
                beta=SimpleNamespace(chat=SimpleNamespace(completions=completion)))
            self.hf_model = self.hf_tokenizer = None

    return PublicChatCompletionModel


class PublicEmbeddingModel:
    def __init__(self, cfg, client=None):
        services = _SERVICES[cfg._mematlas_service_id]
        self.service = services.embedder
        self.io_loop = getattr(services, "_native_io_loop", None)
        self.failures = _FAILURES[cfg._mematlas_service_id]

    def generate_embeddings(self, input):
        try:
            return run_service(self.service.embed(list(input)), self.io_loop)
        except BaseException as exc:
            self.failures.append(exc)
            raise


def _load(root):
    namespace("memora", root / "src" / "memora")
    namespace("memora.utils", root / "src" / "memora" / "utils")
    llm = namespace("memora.utils.llm", root / "src" / "memora" / "utils")
    embedding = namespace("memora.utils.embedding", root / "src" / "memora" / "utils")
    if not hasattr(llm, "ChatCompletionModel"):
        llm.ChatCompletionModel = _native_chat_class(root)
    embedding.BaseEmbeddingModel = PublicEmbeddingModel
    # The official retriever package imports its optional local GRPO policy;
    # namespace selection leaves the selected prompted-policy module untouched.
    namespace("memora.retriever", root / "src" / "memora" / "retriever")
    # Full LocalMemoraClient eagerly loads PDF/Office processors. OpenHospital
    # text ingestion instead composes its actual AgentMemory + ChatMemoryBuilder
    # calls, exactly as LocalMemoraClient.add does, after official segmentation.
    agent = importlib.import_module("memora.core.memory").AgentMemory
    builder = importlib.import_module("memora.builder.chat_memory_builder").ChatMemoryBuilder
    retriever = importlib.import_module("memora.retriever.prompted_policy_retriever").PromptedPolicyRetriever
    name = "memora._mematlas_official_segmentation"
    module = importlib.import_module("sys").modules.get(name)
    if module is None:
        spec = importlib.util.spec_from_file_location(name, root / "app" / "locomo" / "providers" / "memora" / "segmentation.py")
        module = importlib.util.module_from_spec(spec)
        importlib.import_module("sys").modules[name] = module
        spec.loader.exec_module(module)
    return agent, builder, retriever, module.ConversationSegmenter


class NativeMemora:
    def __init__(self, directory, scope, params, services, state=None):
        self.directory = storage_path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.params, self.services = params, services
        self.sources = {}
        root = checked_source(services, "memora", REVISION)
        from omegaconf import OmegaConf
        providers = resolve_providers(params, services.provider_configs)
        self.service_id = uuid.uuid4().hex
        _SERVICES[self.service_id] = services
        self.failures = _FAILURES[self.service_id] = []
        self.cfg = OmegaConf.create({
            "_mematlas_service_id": self.service_id,
            "llm": {"model": providers["llm"].model or "configured-llm", "seed": providers["llm"].options.get("seed", 42)},
            "openai": {"api_type": "openai", "embedding_model": providers["embedding"].model or "configured-embedding"},
            "memory": {"backend": "local", "db_type": "chromadb", "persist_path": str(self.directory),
                "collection_name": "memora", "distance": "cosine", "top_k": params.get("top_k", 30),
                "query_score_threshold": 0.4, "update_score_threshold": 0.8, "bm25_score_threshold": 0.4,
                "enable_cue_index": True, "cue_top_k": 20, "enable_hybrid_search": True, "hybrid_top_k": 10,
                "enable_segmentation": True, "enable_episodic_memory": True, "use_segments_as_episodic": True,
                "include_factual": True, "return_history": True, "enhance_query": False, "multimodal_support": False},
            "retrieval": {"enable_llm_filter": False, "enable_relaxed_frontier": False}, "eval": {"max_workers": 1}})
        with _IMPORT_LOCK:
            AgentMemory, ChatMemoryBuilder, Retriever, Segmenter = _load(root)
            self.agent = AgentMemory(self.cfg, user_id=digest(scope))
            self.model = importlib.import_module("memora.utils.llm").ChatCompletionModel(self.cfg)
            self.builder = ChatMemoryBuilder(self.cfg, self.agent, self.model)
            self.segmenter = Segmenter(self.cfg)
            self.retriever = Retriever(self.cfg, memory_client=self, model_client=self.model,
                                      max_steps=params.get("max_steps", 4))
        self.store = self.agent._store
        self._bind_converted_cue_deletion()
        if state is not None:
            self.validate_state(state)
            self.import_state(state)
        else:
            self.store.build_bm25_index()

    def _bind_converted_cue_deletion(self):
        original_delete = self.agent.delete

        def delete(key):
            target = self.agent.get(key)
            if target is not None and target.is_primary_index():
                # Upstream permits a cue phrase to become a primary index and
                # ignores that reference while it remains a primary. Deleting
                # that primary must also remove those now-obsolete references,
                # otherwise a later unrelated primary deletion asserts on it.
                # Update only relation metadata; retain facts and exact vectors.
                result = self.store.collection.get(include=["metadatas"])
                ids, updates = [], []
                for identity, metadata in zip(result["ids"], result["metadatas"]):
                    if metadata.get("index") == key:
                        continue
                    cues = [value.strip() for value in metadata.get("cue_indices", "").split("||") if value.strip()]
                    if key in cues:
                        ids.append(identity)
                        updates.append({"cue_indices": "||".join(value for value in cues if value != key)})
                if ids:
                    self.store.collection.update(ids=ids, metadatas=updates)
                    from mematlas.core.activity import append
                    append("native_relation_cleanup", method="memora", operation="delete_converted_cue_primary",
                           deleted_index=key, affected_primary_ids=ids)
            return original_delete(key)

        self.agent.delete = delete

    def query(self, context, **kwargs):
        return self.agent.query(context, enhance_query=False, return_history=True, **kwargs)

    def get(self, key):
        return self.agent.get(key)

    def get_user_id(self):
        return self.agent.get_user_id()

    def ingest(self, events, episode_ref, part):
        self.failures.clear()
        messages = [{"role": event.role or "user", "content": content_text(event.content)} for event in events]
        segments = self.segmenter.segment_conversation(messages)
        for i, segment in enumerate(segments):
            indices = segment["indices"]
            metadata = {"timestamp": events[indices[0]].timestamp.isoformat(),
                        "segment_topic": segment["topic"], "segment_index": i,
                        "episode_ref": episode_ref, "episode_part": part,
                        "source_event_ids": json.dumps([events[j].event_id for j in indices])}
            before = self.store.collection.get(include=["metadatas"])
            prior = {row["index"]: row for row in before["metadatas"]}
            self.builder.build([messages[j] for j in indices], metadata=metadata)
            after = self.store.collection.get(include=["metadatas"])
            for row in after["metadatas"]:
                index = row["index"]
                if row != prior.get(index):
                    original = self.sources.get(index, [])
                    self.sources[index] = list(dict.fromkeys(original + [events[j].event_id for j in indices]))
        self.store.build_bm25_index()
        self._check_failures()

    def _row(self, entry):
        metadata = entry.model_dump(mode="json")
        return {"memory_id": entry.index, "content": entry.value,
                "score": float(entry.score or 0), "metadata": metadata,
                "source_event_ids": self.sources.get(entry.index, [])}

    def search(self, query, top_k):
        self.failures.clear()
        if not self.store.collection.count():
            return [], []
        entries = self.retriever.retrieve(query, top_k=top_k)
        # Preserve upstream working-set expansion, which can exceed initial k.
        episodic = self.agent.get_episodic_memories_for_results(entries)
        known = {entry.index for entry in entries}
        entries += [entry for key, entry in episodic.items() if key not in known]
        self._check_failures()
        return [self._row(entry) for entry in entries], self.retriever.get_trace()

    def _check_failures(self):
        if self.failures:
            raise RuntimeError("Memora provider failed; native fallback is not a valid benchmark result") from self.failures[-1]

    def detail(self, memory_id):
        entry = self.agent.get(memory_id)
        return [self._row(entry)] if entry is not None else []

    def export_state(self, snapshots):
        # Collection-level logical snapshot includes ALL primary/cue/episodic
        # rows and exact stored vectors, independent of list_memories(limit=20).
        result = self.store.collection.get(include=["documents", "metadatas", "embeddings"])
        embeddings = result["embeddings"]
        rows = ({"id": key, "document": result["documents"][i], "metadata": result["metadatas"][i],
                        "embedding": embeddings[i].tolist() if hasattr(embeddings[i], "tolist") else list(embeddings[i])}
                       for i, key in enumerate(result["ids"]))
        # Preserve native insertion/enumeration order: upstream BM25 uses this
        # sequence to break ties. Sorting a checkpoint would alter retrieval.
        return {"collection": put_json_array_blob(snapshots, rows),
                "cache_version": 1,
                "embedding_cache": put_json_array_blob(snapshots, self.store._embedding_cache.items()),
                "metadata": dict(self.store.collection.metadata or {}),
                "configuration": self.store.collection.configuration, "sources": self.sources}

    @staticmethod
    def validate_state(state):
        if state.get("cache_version") != 1:
            raise ValueError("Memora checkpoint lacks the complete native LRU cache; old checkpoints are not migrated")
        cache = json.loads(get_blob(state["embedding_cache"]))
        if not isinstance(cache, list) or any(not isinstance(row, list) or len(row)!=2 or not isinstance(row[0],str)
                or not isinstance(row[1],list) or not row[1] for row in cache):
            raise ValueError("Invalid Memora embedding cache checkpoint")
        rows = json.loads(get_blob(state["collection"]))
        if not isinstance(rows, list) or len({row["id"] for row in rows}) != len(rows):
            raise ValueError("Invalid Memora collection checkpoint")
        for row in rows:
            if not row["embedding"] or not isinstance(row["metadata"], dict):
                raise ValueError("Memora checkpoint requires exact stored vectors and metadata")
        if not isinstance(state["metadata"], dict) or state["metadata"].get("hnsw:space") != "cosine":
            raise ValueError("Memora checkpoint has incompatible collection metric")

    def import_state(self, state):
        rows = json.loads(get_blob(state["collection"]))
        self.sources = state["sources"]
        collection = self.store.collection
        if collection.configuration != state["configuration"]:
            raise ValueError("Memora checkpoint collection configuration differs")
        if dict(collection.metadata or {}) != state["metadata"]:
            collection.modify(metadata=state["metadata"])
        existing = collection.get()["ids"]
        if existing:
            collection.delete(ids=existing)
        for offset in range(0, len(rows), 1000):
            batch = rows[offset:offset + 1000]
            collection.upsert(ids=[row["id"] for row in batch], documents=[row["document"] for row in batch],
                metadatas=[row["metadata"] for row in batch], embeddings=[row["embedding"] for row in batch])
        self.store._embedding_cache.clear()
        self.store._embedding_cache.update(json.loads(get_blob(state["embedding_cache"])))
        self.store.build_bm25_index()

    def reset(self):
        ids = self.store.collection.get()["ids"]
        if ids:
            self.store.collection.delete(ids=ids)
        self.store._embedding_cache.clear()
        self.store.build_bm25_index()
        self.close()

    def close(self):
        # Chroma owns the shared PersistentClient lifetime; killing a singleton
        # here would invalidate other scopes. No background model jobs exist.
        _SERVICES.pop(self.service_id, None)
        _FAILURES.pop(self.service_id, None)

