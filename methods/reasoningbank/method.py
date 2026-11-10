"""Memory v2 adapter for the external-feedback ReasoningBank variant."""
from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from mematlas.core.config import MethodConfig
from mematlas.core.models import MemoryContribution, MemoryEpisode, MemoryEvent, Scope, content_text
from mematlas.core.plugin import BaseMemoryMethod, MethodContext, MethodPlugin, MethodServices
from mematlas.core.scope import normalize_scope
from mematlas.core.files import atomic_write_json, read_json
from mematlas.methods.core_utils import cosine_similarity, offline_embedding
from mematlas.methods.manifest import load_method_descriptor
from mematlas.methods.services import prepare_services
from mematlas.methods.storage import configured_file

from .config import _DESCRIPTOR
from .prompts import extraction_prompt, parse_notes, retrieval_query


DESCRIPTOR = _DESCRIPTOR


def _group_id(scope: Scope, episode_id: str) -> str:
    payload = json.dumps({"scope": dict(scope), "episode_id": episode_id}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _finite_vector(value: Any) -> list[float]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError("ReasoningBank embeddings must be nonempty numeric vectors")
    vector = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item):
            raise ValueError("ReasoningBank embeddings must be finite numeric vectors")
        vector.append(float(item))
    return vector


class ReasoningBankMemory(BaseMemoryMethod):
    """Single-trajectory extraction with query-level similarity retrieval.

    The native method stores all notes induced from one trajectory as one group,
    so retrieval never splits a successful or failed episode into independent
    memories.
    """

    def __init__(self, config: MethodConfig, services: MethodServices):
        self.config = config
        self.params = dict(config.params)
        self.services = services
        self.domain = str(self.params.get("domain", "web navigation"))
        self._groups: dict[str, dict[str, Any]] = {}
        self._path: Path | None = None

    async def setup(self, context: MethodContext) -> None:
        # Extraction always uses an LLM. Offline mode only replaces the vector
        # provider with the deterministic local embedding used by smoke tests.
        prepare_services(self.params, self.services, method_id="reasoningbank",
                         llm=True, embedding=self.params.get("implementation_mode") != "offline")
        if self.services.llm is None:
            raise ValueError("ReasoningBank requires an extraction LLM")
        self._path = configured_file(self.services, "reasoningbank", self.params.get("storage_path"),
                                     "reasoningbank.json")
        if self._path.exists():
            self._restore(read_json(self._path, {}))
        else:
            self._persist_native()

    async def _embed(self, text: str) -> list[float]:
        if self.params.get("implementation_mode") == "offline":
            return _finite_vector(offline_embedding(text))
        if self.services.embedder is None:
            raise ValueError("ReasoningBank standard mode requires an embedding service")
        vectors = await self.services.embedder.embed([text])
        if len(vectors) != 1:
            raise ValueError("ReasoningBank embedding provider must return one vector per query")
        return _finite_vector(vectors[0])

    async def _observe(self, event: MemoryEvent) -> None:
        # Native ReasoningBank learns only after an external success label.
        return None

    @staticmethod
    def _trajectory(episode: MemoryEpisode) -> str:
        events = episode.events or []
        parts = []
        for event in events:
            text = content_text(event.content).strip()
            if text:
                parts.append(f"{event.role or event.kind.value}: {text}")
        return "\n".join(parts)

    async def _learn(self, episode: MemoryEpisode, *, observed_event_ids=frozenset()) -> None:
        if not isinstance(episode.success, bool):
            raise ValueError("ReasoningBank requires a real boolean episode.success")
        if not isinstance(episode.task, str) or not episode.task.strip():
            raise ValueError("ReasoningBank requires a nonempty episode.task")
        trajectory = self._trajectory(episode)
        if not trajectory.strip():
            raise ValueError("ReasoningBank requires a nonempty trajectory in episode.events")
        scope = normalize_scope(episode.scope or {})
        identity = _group_id(scope, episode.episode_id)
        if identity in self._groups:
            return
        request = f"**Query:** {episode.task}\n\n**Trajectory:**\n{trajectory}"
        if self.params.get("include_feedback", False) and episode.feedback:
            label = "succeeded" if episode.success else "failed"
            request += f"\n\nThe task {label} because: {episode.feedback}"
        output = await self.services.llm.complete(
            [{"role": "system", "content": extraction_prompt(episode.success, domain=self.domain)},
             {"role": "user", "content": request}],
            temperature=float(self.params.get("extraction_temperature", 1.0)),
        )
        notes = parse_notes(output)
        vector = await self._embed(episode.task)
        if self._groups and any(len(row["vector"]) != len(vector) for row in self._groups.values()):
            raise ValueError("ReasoningBank task embedding dimension differs from the existing bank")
        self._groups[identity] = {
            "id": identity,
            "episode_id": episode.episode_id,
            "scope": dict(scope),
            "task": episode.task,
            "success": episode.success,
            "reward": episode.reward,
            "feedback": episode.feedback,
            "memory": output,
            "memory_items": output.split("\n\n"),
            "note_count": len(notes),
            "vector": vector,
            "source_event_ids": [event.event_id for event in episode.events or []],
        }
        self._persist_native()

    async def search(self, query, *, level_ref=None, top_k, scope, extensions) -> MemoryContribution:
        owner = normalize_scope(scope)
        candidates = [row for row in self._groups.values() if row["scope"] == owner]
        limit = top_k if top_k is not None else int(self.params.get("top_k", 1))
        if type(limit) is not int or not 1 <= limit <= 10:
            raise ValueError("ReasoningBank selects between 1 and 10 episode groups")
        if not candidates:
            return MemoryContribution(items=[], extensions={"reasoningbank": {
                "scope": dict(owner), "variant": "external_feedback_single_trajectory",
                "selection_unit": "episode_query", "domain": self.domain,
            }})
        task = content_text(query).strip()
        if not task:
            raise ValueError("ReasoningBank retrieval requires a nonempty text query")
        vector = await self._embed(retrieval_query(task, domain=self.domain))
        if any(len(row["vector"]) != len(vector) for row in candidates):
            raise ValueError("ReasoningBank query embedding dimension differs from stored task embeddings")
        scored = sorted(((cosine_similarity(vector, row["vector"]), row) for row in candidates),
                        key=lambda pair: (-pair[0], pair[1]["id"]))[:limit]
        items = []
        for score, row in scored:
            notes = row.get("memory_items") or row["memory"].split("\n\n")
            for index, note in enumerate(notes):
                items.append({
                    "content": note,
                    "memory_id": f"{row['id']}:{index}",
                    "score": float(score),
                    "source_event_ids": list(row["source_event_ids"]),
                    "memory_type": "reasoning_memory",
                    "metadata": {
                        "method_id": "reasoningbank",
                        "task": row["task"],
                        "source_episode_id": row["episode_id"],
                        "source_success": row["success"],
                        "group_id": row["id"],
                        "note_index": index,
                        "note_count": row["note_count"],
                        "scope": dict(row["scope"]),
                    },
                })
        return MemoryContribution(items=items, extensions={"reasoningbank": {
            "scope": dict(owner), "variant": "external_feedback_single_trajectory",
            "selection_unit": "episode_query", "domain": self.domain,
            "query": task,
        }})

    async def save(self) -> Mapping[str, Any]:
        return deepcopy({"schema_version": 3, "groups": self._groups})

    async def load(self, state: Mapping[str, Any]) -> None:
        self._restore(state)
        self._persist_native()

    async def reset(self, scope: Scope) -> None:
        owner = normalize_scope(scope)
        self._groups = {key: row for key, row in self._groups.items() if row["scope"] != owner}
        self._persist_native()

    def _persist_native(self) -> None:
        if self._path is not None:
            atomic_write_json(self._path, {"schema_version": 3, "groups": self._groups})

    def _restore(self, state: Mapping[str, Any]) -> None:
        if state.get("schema_version") not in {1, 2, 3} or not isinstance(state.get("groups"), Mapping):
            raise ValueError("Invalid ReasoningBank snapshot schema")
        restored = deepcopy(dict(state["groups"]))
        dimensions: set[int] = set()
        for key, row in restored.items():
            if not isinstance(row, Mapping) or row.get("id") != key:
                raise ValueError("ReasoningBank group identity does not match snapshot key")
            scope = normalize_scope(row.get("scope", {}))
            if key != _group_id(scope, str(row.get("episode_id", ""))):
                raise ValueError("ReasoningBank group identity is invalid")
            raw_notes = row.get("memory_items")
            if raw_notes is None:
                raw_notes = str(row.get("memory", "")).split("\n\n")
            if not isinstance(raw_notes, list) or any(not isinstance(note, str) for note in raw_notes):
                raise ValueError("ReasoningBank group memory_items must be a list of strings")
            notes = parse_notes("\n\n".join(raw_notes))
            vector = _finite_vector(row.get("vector"))
            if len(notes) != len(raw_notes) or len(raw_notes) != row.get("note_count"):
                raise ValueError("ReasoningBank note count does not match snapshot")
            source_ids = row.get("source_event_ids", [])
            if not isinstance(source_ids, list) or any(not isinstance(value, str) for value in source_ids):
                raise ValueError("ReasoningBank source_event_ids are invalid")
            dimensions.add(len(vector))
            row["scope"] = dict(scope)
            row["memory_items"] = list(raw_notes)
            row["vector"] = vector
        if len(dimensions) > 1:
            raise ValueError("ReasoningBank snapshot contains mixed embedding dimensions")
        self._groups = restored


class ReasoningBankPlugin(MethodPlugin):
    descriptor = DESCRIPTOR
    qa_factory = "mematlas.methods.reasoningbank.qa:ReasoningBankQA"

    def create_method(self, config: MethodConfig, services: MethodServices) -> BaseMemoryMethod:
        return ReasoningBankMemory(config, services)

    def healthcheck(self, config: MethodConfig, services: MethodServices) -> Mapping[str, Any]:
        result = dict(super().healthcheck(config, services))
        # Offline mode still needs an extraction LLM; only embeddings are local.
        if config.params.get("implementation_mode") == "offline":
            missing = [item for item in result.get("missing_services", []) if item != "embedding"]
            if services.llm is None and "llm" not in missing:
                missing.append("llm")
            result["missing_services"] = missing
            result["ready"] = not (result.get("config_errors") or result.get("missing_packages") or missing)
            result["status"] = "ready" if result["ready"] else "unavailable"
        return result


PLUGIN = ReasoningBankPlugin()
