from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import math
from numbers import Real
from typing import Any, Mapping
from uuid import uuid4

from mematlas.core.plugin import MethodServices
from mematlas.core.provider_config import resolve_providers
from mematlas.methods.core_utils import tokens
from mematlas.methods.a_mem.prompts import ANALYSIS_PROMPT, ANALYSIS_FORMAT, EVOLUTION_PROMPT, EVOLUTION_FORMAT
from mematlas.methods.storage import method_data_dir


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d%H%M")


@dataclass
class MemoryNote:
    content: str
    id: str = field(default_factory=lambda: uuid4().hex)
    keywords: list[str] = field(default_factory=list)
    links: list[str] = field(default_factory=list)
    retrieval_count: int = 0
    timestamp: str = field(default_factory=_timestamp)
    last_accessed: str = field(default_factory=_timestamp)
    context: str = "General"
    evolution_history: list[dict[str, Any]] = field(default_factory=list)
    category: str = "Uncategorized"
    tags: list[str] = field(default_factory=list)
    embedding: list[float] | None = None
    source_event_ids: list[str] = field(default_factory=list)
    evolution_source_event_ids: list[str] = field(default_factory=list)


class AgenticMemoryCore:
    """A-MEM linked notes with semantic metadata and LLM-guided evolution."""

    def __init__(
        self,
        services: MethodServices | None = None,
        *,
        evo_threshold: int = 100,
        neighbor_k: int = 5,
        link_threshold: float = 0.0,
        storage_path: str | None = None,
        offline: bool = False,
        **_: Any,
    ):
        self.services = services or MethodServices()
        self.evo_threshold = evo_threshold
        self.neighbor_k = neighbor_k
        self.link_threshold = link_threshold
        self.offline = offline
        # Some OpenAI-compatible providers support JSON Object but not JSON Schema.
        # This is transport configuration; prompts and local field checks stay unchanged.
        self._response_format = resolve_providers({}, self.services.provider_configs)["llm"].options.get("response_format")
        self.evo_count = 0
        self.memories: dict[str, MemoryNote] = {}
        try:
            import chromadb
        except ImportError as exc:
            raise RuntimeError(
                "A-Mem requires chromadb; install chromadb"
            ) from exc
        directory = method_data_dir(self.services, "a-mem", storage_path)
        self._chroma_client = chromadb.PersistentClient(path=str(directory))
        self._collection = self._chroma_client.get_or_create_collection(
            name="memories",
            # The referenced upstream collection uses Chroma's default L2 space.
            metadata={"hnsw:space": "l2"},
            embedding_function=None,
        )
        self._load_collection()

    @staticmethod
    def _fallback_embedding(text: str, dimensions: int = 128) -> list[float]:
        values = [0.0] * dimensions
        for token in tokens(text):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % dimensions
            values[index] += 1.0 if digest[4] & 1 else -1.0
        norm = sum(value * value for value in values) ** 0.5
        return [value / norm for value in values] if norm else [0.0] * dimensions

    def _metadata(self, note: MemoryNote) -> dict[str, Any]:
        return {
            "keywords": json.dumps(note.keywords, ensure_ascii=False),
            "links": json.dumps(note.links, ensure_ascii=False),
            "retrieval_count": note.retrieval_count,
            "timestamp": note.timestamp,
            "last_accessed": note.last_accessed,
            "context": note.context,
            "evolution_history": json.dumps(note.evolution_history, ensure_ascii=False),
            "category": note.category,
            "tags": json.dumps(note.tags, ensure_ascii=False),
            "source_event_ids": json.dumps(note.source_event_ids, ensure_ascii=False),
            "evolution_source_event_ids": json.dumps(note.evolution_source_event_ids, ensure_ascii=False),
        }

    def _upsert_note(self, note: MemoryNote) -> None:
        if not note.embedding:
            raise ValueError("A-Mem note requires a nonempty Embedding vector")
        self._collection.upsert(
            ids=[note.id],
            documents=[note.content],
            embeddings=[note.embedding],
            metadatas=[self._metadata(note)],
        )

    def _load_collection(self) -> None:
        payload = self._collection.get(include=["documents", "metadatas", "embeddings"])
        documents = payload.get("documents") or []
        metadatas = payload.get("metadatas") or []
        embeddings = payload.get("embeddings")
        if embeddings is None:
            embeddings = []
        for note_id, content, metadata, embedding in zip(
            payload.get("ids", []),
            documents,
            metadatas,
            embeddings,
        ):
            metadata = dict(metadata or {})
            def decode(name: str, fallback: Any):
                try:
                    return json.loads(metadata.get(name, ""))
                except (TypeError, json.JSONDecodeError):
                    return fallback
            self.memories[str(note_id)] = MemoryNote(
                id=str(note_id),
                content=str(content or ""),
                keywords=list(decode("keywords", [])),
                links=list(decode("links", [])),
                retrieval_count=int(metadata.get("retrieval_count", 0)),
                timestamp=str(metadata.get("timestamp", _timestamp())),
                last_accessed=str(metadata.get("last_accessed", _timestamp())),
                context=str(metadata.get("context", "General")),
                evolution_history=list(decode("evolution_history", [])),
                category=str(metadata.get("category", "Uncategorized")),
                tags=list(decode("tags", [])),
                embedding=list(embedding) if embedding is not None else None,
                source_event_ids=list(decode("source_event_ids", [])),
                evolution_source_event_ids=list(decode("evolution_source_event_ids", [])),
            )

    def _persist(self) -> None:
        for note in self.memories.values():
            self._upsert_note(note)

    async def _analyze(self, content: str) -> dict[str, Any]:
        if self.offline:
            return {"keywords": list(dict.fromkeys(tokens(content)))[:12],
                    "context": content[:240] or "General", "tags": []}
        response = await self.services.llm.complete([
            {"role": "system", "content": "You must respond with a JSON object."},
            {"role": "user", "content": ANALYSIS_PROMPT + content},
        ], temperature=0.7, max_tokens=1000, response_format=self._response_format or ANALYSIS_FORMAT)
        try:
            parsed = json.loads(str(response))
            if (not isinstance(parsed, dict) or not isinstance(parsed.get("context"), str)
                    or any(not self._strings(parsed.get(key)) for key in ("keywords", "tags"))):
                raise ValueError("Invalid analysis fields")
            return {key: parsed[key] for key in ("keywords", "context", "tags")}
        except (ValueError, TypeError):
            # Upstream analysis failure retains an unannotated note, not invented facts.
            self.services.logger.warning("A-Mem analysis returned invalid JSON/fields; retaining General context")
            return {"keywords": [], "context": "General", "tags": []}

    @staticmethod
    def _strings(value: Any) -> bool:
        return isinstance(value, list) and all(isinstance(item, str) for item in value)

    async def _embedding(self, text: str) -> list[float] | None:
        if self.services.embedder is None:
            if self.offline:
                return self._fallback_embedding(text)
            raise RuntimeError("A-Mem standard mode requires an Embedding service")
        vectors = await self.services.embedder.embed([text])
        if len(vectors) != 1 or not vectors[0] or any(
            isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value)
            for value in vectors[0]
        ):
            raise ValueError("A-Mem Embedding service must return one nonempty finite vector")
        return [float(value) for value in vectors[0]]

    async def _rank(self, query: str, k: int) -> list[dict[str, Any]]:
        if not self.memories or k < 1:
            return []
        query_embedding = await self._embedding(query)
        result = self._collection.query(
            query_embeddings=[query_embedding], n_results=min(k, len(self.memories)), include=["distances"],
        )
        ranked = []
        for note_id, distance in zip(result["ids"][0], result["distances"][0]):
            note = self.memories[str(note_id)]
            ranked.append({
                **asdict(note), "distance": float(distance),
                "score": 1.0 / (1.0 + max(0.0, float(distance))),
                "memory": note.content,
            })
        return ranked

    async def _evolve(self, note: MemoryNote, neighbors: list[dict[str, Any]]) -> bool:
        eligible = [item for item in neighbors if float(item.get("score", 0.0)) >= self.link_threshold]
        if not eligible or self.offline:
            return False
        neighbors_text = "".join(
            f"memory id:{item['id']}\tmemory index:{index}\ttalk start time:{item['timestamp']}\t"
            f"memory content: {item['content']}\tmemory context: {item['context']}\t"
            f"memory keywords: {item['keywords']}\tmemory tags: {item['tags']}\n"
            for index, item in enumerate(eligible)
        )
        response = await self.services.llm.complete([
            {"role": "system", "content": "You must respond with a JSON object."},
            {"role": "user", "content": EVOLUTION_PROMPT.format(
                content=note.content, context=note.context, keywords=note.keywords,
                nearest_neighbors_memories=neighbors_text, neighbor_number=len(eligible))},
        ], temperature=0.7, max_tokens=1000, response_format=self._response_format or EVOLUTION_FORMAT)
        try:
            decision = json.loads(str(response))
            if not isinstance(decision, dict) or type(decision.get("should_evolve")) is not bool:
                raise ValueError("Invalid evolution decision")
            for key in ("actions", "suggested_connections", "tags_to_update", "new_context_neighborhood"):
                if not self._strings(decision.get(key)):
                    raise ValueError(f"Invalid {key}")
            tags = decision.get("new_tags_neighborhood")
            if not isinstance(tags, list) or any(not self._strings(row) for row in tags):
                raise ValueError("Invalid neighborhood tags")
            if "update_neighbor" in decision["actions"] and decision["should_evolve"]:
                if len(tags) != len(eligible) or len(decision["new_context_neighborhood"]) != len(eligible):
                    raise ValueError("Neighborhood updates must align with supplied neighbors")
        except (TypeError, ValueError):
            self.services.logger.warning("A-Mem evolution returned invalid JSON/fields; skipping evolution")
            return False
        if not decision["should_evolve"]:
            return False
        actions = decision["actions"]
        # Capture the decision's complete input lineage NOW. Looking up a mutable
        # neighbor graph during later retrieval would leak future dependencies.
        input_sources = list(dict.fromkeys([
            *note.source_event_ids, *note.evolution_source_event_ids,
            *(event_id for item in eligible for key in ("source_event_ids", "evolution_source_event_ids")
              for event_id in item.get(key, [])),
        ]))
        eligible_ids = {str(item["id"]) for item in eligible}
        connections = [str(value) for value in decision.get("suggested_connections", [])
                       if str(value) in eligible_ids]
        changed_at = _timestamp()
        if "strengthen" in actions:
            note.links = list(dict.fromkeys([*note.links, *connections]))
            tags = decision.get("tags_to_update", [])
            if isinstance(tags, list):
                note.tags = list(dict.fromkeys(str(value) for value in tags))
            note.evolution_history.append({"at": changed_at, "action": "strengthen", "connections": connections})
            note.evolution_source_event_ids = list(dict.fromkeys([*note.evolution_source_event_ids, *input_sources]))
        if "update_neighbor" in actions:
            for item, context, tags in zip(eligible, decision["new_context_neighborhood"], decision["new_tags_neighborhood"]):
                # Use the actual retrieved ID, not an index into insertion order.
                neighbor = self.memories[item["id"]]
                neighbor.context = context
                neighbor.tags = list(tags)
                neighbor.evolution_history.append({"at": changed_at, "action": "update_neighbor", "source_memory_id": note.id})
                neighbor.evolution_source_event_ids = list(dict.fromkeys([
                    *neighbor.evolution_source_event_ids, *input_sources,
                ]))
                self._upsert_note(neighbor)
        return True

    async def add_note(self, content: str, **kwargs: Any) -> str:
        analysis = await self._analyze(content)
        supplied_tags = [str(value) for value in kwargs.get("tags", [])]
        note = MemoryNote(
            content=content,
            timestamp=str(kwargs.get("timestamp") or kwargs.get("time") or _timestamp()),
            category=str(kwargs.get("category") or "Uncategorized"),
            keywords=analysis["keywords"],
            context=analysis["context"],
            tags=list(dict.fromkeys([*supplied_tags, *analysis["tags"]])),
            embedding=await self._embedding(content),
            source_event_ids=list(kwargs.get("source_event_ids", [])),
        )
        neighbors = await self._rank(content, min(self.neighbor_k, len(self.memories)))
        evolved = await self._evolve(note, neighbors) if neighbors else False
        self.memories[note.id] = note
        if evolved:
            self.evo_count += 1
            if self.evo_count % self.evo_threshold == 0:
                await self.consolidate_memories()
        self._upsert_note(note)
        return note.id

    async def search(self, query: str, k: int = 5, *, read_only: bool = False):
        # Upstream retrieval does not mutate retrieval_count or timestamps.
        return await self._rank(query, k)

    async def search_agentic(self, query: str, k: int = 5, *, read_only: bool = False):
        direct = await self.search(query, max(k, 1), read_only=read_only)
        output = list(direct)
        seen = {item["id"] for item in output}
        for item in direct:
            for linked_id in item.get("links", []):
                if linked_id not in seen and linked_id in self.memories:
                    linked = self.memories[linked_id]
                    output.append({**asdict(linked), "memory": linked.content, "is_neighbor": True})
                    seen.add(linked_id)
        return output[:k]

    async def consolidate_memories(self):
        for note in self.memories.values():
            note.links = list(dict.fromkeys(link for link in note.links if link in self.memories))
        self._persist()

    def read(self, memory_id: str):
        return asdict(self.memories[memory_id])

    async def update(self, memory_id: str, **values: Any):
        note = self.memories[memory_id]
        old = {"content": note.content, "context": note.context, "tags": list(note.tags)}
        for key, value in values.items():
            if hasattr(note, key) and key not in {"id", "embedding"}:
                setattr(note, key, value)
        if "content" in values:
            analysis = await self._analyze(note.content)
            note.keywords = analysis["keywords"]
            note.context = analysis["context"]
            note.embedding = await self._embedding(note.content)
        note.evolution_history.append({"at": _timestamp(), "action": "manual_update", "old": old})
        self._upsert_note(note)
        return asdict(note)

    async def delete(self, memory_id: str):
        removed = self.memories.pop(memory_id, None)
        if removed is not None:
            self._collection.delete(ids=[memory_id])
        await self.consolidate_memories()
        return asdict(removed) if removed else None

    def export_state(self) -> dict[str, Any]:
        return {
            "memories": {key: asdict(note) for key, note in self.memories.items()},
            "evo_count": self.evo_count,
        }

    def import_state(self, state: Mapping[str, Any]) -> None:
        records = state.get("memories", {}) if isinstance(state, Mapping) else {}
        restored = {}
        if isinstance(records, Mapping):
            for memory_id, payload in records.items():
                if isinstance(payload, Mapping):
                    values = dict(payload)
                    values["id"] = str(values.get("id") or memory_id)
                    if not values.get("embedding"):
                        raise ValueError("A-Mem snapshots must preserve Embedding vectors")
                    restored[values["id"]] = MemoryNote(**values)
        existing_ids = self._collection.get(include=[])["ids"]
        if existing_ids:
            self._collection.delete(ids=existing_ids)
        self.memories = restored
        self.evo_count = int(state.get("evo_count", 0)) if isinstance(state, Mapping) else 0
        self._persist()

    async def close(self) -> None:
        self._persist()
