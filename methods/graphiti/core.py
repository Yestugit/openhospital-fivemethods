from __future__ import annotations

from datetime import datetime
import json
import hashlib
import tempfile
import os
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

from mematlas.core.plugin import MethodServices
from mematlas.methods.core_utils import cosine_similarity, lexical_score, parse_json_object, tokens
from mematlas.methods.storage import SQLiteState, configured_file, load_mapping, ascii_database_path


class GraphitiCore:
    """Incremental temporal graph with episode provenance and hybrid retrieval."""

    def __init__(self, services: MethodServices, *, storage_path: str | None = None, **_: Any):
        self.services = services
        self.episodes: dict[str, dict[str, Any]] = {}
        self.entities: dict[str, dict[str, Any]] = {}
        self.facts: dict[str, dict[str, Any]] = {}
        self._store = SQLiteState(
            configured_file(services, "graphiti", storage_path, "graphiti.db")
        )
        try:
            import kuzu
        except ImportError as exc:
            raise RuntimeError(
                "Graphiti requires kuzu; install mematlas[native-methods]"
            ) from exc
        kuzu_root = self._store.path.parent
        kuzu_root.mkdir(parents=True, exist_ok=True)
        # Kuzu's Windows native backend cannot open paths containing non-ASCII
        # parent components. Keep the external DB in an ASCII temp root while
        # deriving its identity from the requested artifact path.
        kuzu_path = ascii_database_path(kuzu_root, "graphiti.kuzu", services=services)
        self._kuzu_db = kuzu.Database(str(kuzu_path))
        self._kuzu = kuzu.Connection(self._kuzu_db)
        self._kuzu.execute("CREATE NODE TABLE IF NOT EXISTS Episode(id STRING PRIMARY KEY, name STRING, content STRING, source_description STRING, reference_time STRING, group_id STRING)")
        self._kuzu.execute("CREATE NODE TABLE IF NOT EXISTS Fact(id STRING PRIMARY KEY, fact STRING, subject STRING, relation STRING, object STRING, group_id STRING, episode_id STRING, created_at STRING, valid_at STRING, invalid_at STRING)")
        self._kuzu.execute("CREATE NODE TABLE IF NOT EXISTS Entity(id STRING PRIMARY KEY, name STRING, group_id STRING)")
        self._kuzu.execute("CREATE NODE TABLE IF NOT EXISTS FactRecord(id STRING PRIMARY KEY, payload STRING, group_id STRING)")
        self._kuzu.execute("CREATE REL TABLE IF NOT EXISTS Relation(FROM Entity TO Entity, fact_id STRING)")
        self._indexed_records = {str(row[0]): str(row[1]) for row in self._kuzu.execute("MATCH (f:FactRecord) RETURN f.id, f.payload").get_all()}
        self._load_kuzu()
        state = load_mapping(self._store, "graph")
        if state:
            self.import_state(state, persist=False)
        self._persist_kuzu()

    def _persist(self) -> None:
        self._store.save("graph", self.export_state())
        self._persist_kuzu()

    def _load_kuzu(self) -> None:
        for row in self._kuzu.execute("MATCH (n:Episode) RETURN n.id, n.name, n.content, n.source_description, n.reference_time, n.group_id").get_all():
            episode_id, name, content, source_description, reference_time, group_id = row
            self.episodes[str(episode_id)] = {
                "uuid": str(episode_id), "name": name, "content": content,
                "source_description": source_description, "reference_time": reference_time,
                "group_id": group_id,
            }
        for row in self._kuzu.execute("MATCH (n:Fact) RETURN n.id, n.fact, n.subject, n.relation, n.object, n.group_id, n.episode_id, n.created_at, n.valid_at, n.invalid_at").get_all():
            fact_id, fact, subject, relation, obj, group_id, episode_id, created_at, valid_at, invalid_at = row
            self.facts[str(fact_id)] = {
                "uuid": str(fact_id), "fact": fact, "content": fact, "subject": subject,
                "relation": relation, "object": obj, "group_id": group_id,
                "episode_id": episode_id, "episode_ids": [episode_id],
                "source_event_ids": [], "created_at": created_at, "valid_at": valid_at,
                "invalid_at": invalid_at, "invalidated_by": None, "embedding": None,
            }

    def _persist_kuzu(self) -> None:
        # Store actual entity relations, with episode and vector provenance on facts.
        payloads = {key: json.dumps(value, ensure_ascii=False, sort_keys=True)
                    for key, value in self.facts.items()}
        self._kuzu.execute("BEGIN TRANSACTION")
        try:
            for key in self._indexed_records.keys() - payloads.keys():
                self._kuzu.execute("MATCH ()-[r:Relation]->() WHERE r.fact_id = $id DELETE r", {"id": key})
                self._kuzu.execute("MATCH (f:FactRecord) WHERE f.id = $id DELETE f", {"id": key})
            for key, payload in payloads.items():
                if self._indexed_records.get(key) == payload:
                    continue
                fact = self.facts[key]
                entities = []
                for name in (fact["subject"], fact["object"]):
                    entity_id = hashlib.sha256(json.dumps([fact["group_id"], name.casefold()]).encode()).hexdigest()
                    self._kuzu.execute(
                        "MERGE (e:Entity {id: $id}) SET e.name = $name, e.group_id = $group_id",
                        {"id": entity_id, "name": name, "group_id": fact["group_id"]},
                    )
                    entities.append(entity_id)
                self._kuzu.execute(
                    "MERGE (f:FactRecord {id: $id}) SET f.payload = $payload, f.group_id = $group_id",
                    {"id": key, "payload": payload, "group_id": fact["group_id"]},
                )
                self._kuzu.execute("MATCH ()-[r:Relation]->() WHERE r.fact_id = $id DELETE r", {"id": key})
                self._kuzu.execute(
                    "MATCH (s:Entity), (o:Entity) WHERE s.id = $s AND o.id = $o CREATE (s)-[:Relation {fact_id: $id}]->(o)",
                    {"s": entities[0], "o": entities[1], "id": key},
                )
            self._kuzu.execute("COMMIT")
        except BaseException:
            self._kuzu.execute("ROLLBACK")
            raise
        self._indexed_records = payloads

    async def build_indices_and_constraints(self, delete_existing: bool = False):
        if delete_existing:
            self.episodes.clear()
            self.entities.clear()
            self.facts.clear()
            self._persist()

    async def _extract(self, text: str, group_id: str) -> list[dict[str, Any]]:
        if self.services.llm is not None:
            active = [
                {
                    "id": fact["uuid"],
                    "subject": fact["subject"],
                    "relation": fact["relation"],
                    "object": fact["object"],
                    "fact": fact["fact"],
                }
                for fact in self.facts.values()
                if fact["group_id"] == group_id and fact.get("invalid_at") is None
            ]
            response = await self.services.llm.complete([
                {
                    "role": "user",
                    "content": (
                        "Extract temporal knowledge graph facts. Return JSON with facts; each fact has "
                        "subject, relation, object, fact, optional valid_at, and invalidates containing "
                        "IDs of existing facts directly contradicted by this episode. Resolve pronouns "
                        "using only this episode. Keep compatible multi-valued facts active.\n"
                        f"Existing active facts: {active}\nEpisode: {text}"
                    ),
                }
            ], temperature=0.0)
            parsed = parse_json_object(str(response))
            if parsed:
                facts = parsed.get("facts", [])
                normalized = [dict(item) for item in facts if isinstance(item, Mapping)]
                return normalized
            raise ValueError("Graphiti model response must contain a facts array")
        terms = list(dict.fromkeys(tokens(text)))
        subject = terms[0] if terms else "episode"
        return [{
            "subject": subject,
            "relation": "states",
            "object": text,
            "fact": text,
            "invalidates": [],
        }]

    async def _embedding(self, text: str) -> list[float] | None:
        if self.services.embedder is None:
            return None
        return list((await self.services.embedder.embed([text]))[0])

    async def add_episode(
        self,
        *,
        name: str,
        episode_body: str,
        source_description: str,
        reference_time: datetime,
        group_id: str,
        **_: Any,
    ):
        episode_id = uuid4().hex
        self.episodes[episode_id] = {
            "uuid": episode_id,
            "name": name,
            "content": episode_body,
            "source_description": source_description,
            "reference_time": reference_time.isoformat(),
            "group_id": group_id,
        }
        for extracted in await self._extract(episode_body, group_id):
            subject = str(extracted.get("subject", "unknown")).strip() or "unknown"
            relation = str(extracted.get("relation", "states")).strip() or "states"
            obj = str(extracted.get("object", "")).strip()
            fact_text = str(extracted.get("fact") or f"{subject} {relation} {obj}").strip()
            duplicate = next(
                (
                    fact
                    for fact in self.facts.values()
                    if fact["group_id"] == group_id
                    and fact["subject"].casefold() == subject.casefold()
                    and fact["relation"].casefold() == relation.casefold()
                    and fact["object"].casefold() == obj.casefold()
                    and fact.get("invalid_at") is None
                ),
                None,
            )
            if duplicate is not None:
                duplicate["episode_ids"].append(episode_id)
                duplicate["source_event_ids"].append(name)
                continue

            fact_id = uuid4().hex
            invalidates = extracted.get("invalidates", [])
            if isinstance(invalidates, str):
                invalidates = [invalidates]
            if not isinstance(invalidates, list):
                invalidates = []
            valid_at = str(extracted.get("valid_at") or reference_time.isoformat())
            self.facts[fact_id] = {
                "uuid": fact_id,
                "fact": fact_text,
                "content": fact_text,
                "subject": subject,
                "relation": relation,
                "object": obj,
                "group_id": group_id,
                "episode_id": episode_id,
                "episode_ids": [episode_id],
                "source_event_ids": [name],
                "created_at": reference_time.isoformat(),
                "valid_at": valid_at,
                "invalid_at": extracted.get("invalid_at"),
                "invalidated_by": None,
                "embedding": await self._embedding(fact_text),
            }
            for old_id in invalidates:
                old = self.facts.get(str(old_id))
                if old and old["group_id"] == group_id and old.get("invalid_at") is None:
                    old["invalid_at"] = valid_at
                    old["invalidated_by"] = fact_id
            for entity_name in (subject, obj):
                if not entity_name:
                    continue
                key = f"{group_id}:{entity_name.casefold()}"
                entity = self.entities.setdefault(key, {
                    "uuid": uuid4().hex,
                    "name": entity_name,
                    "group_id": group_id,
                    "episodes": [],
                })
                if episode_id not in entity["episodes"]:
                    entity["episodes"].append(episode_id)
        self._persist()
        return {"episode": self.episodes[episode_id]}

    async def search(self, query: str, *, group_ids=None, num_results: int = 10):
        allowed = set(group_ids or [])
        query_embedding = await self._embedding(query)
        ranked = []
        rows = self._kuzu.execute(
            "MATCH (s:Entity)-[r:Relation]->(o:Entity), (f:FactRecord) "
            "WHERE r.fact_id = f.id AND (size($groups) = 0 OR f.group_id IN $groups) "
            "RETURN DISTINCT f.payload", {"groups": list(allowed)},
        ).get_all()
        for row in rows:
            fact = json.loads(row[0])
            lexical = lexical_score(query, fact["fact"])
            entity_bonus = max(
                lexical_score(query, fact["subject"]),
                lexical_score(query, fact["object"]),
            )
            semantic = cosine_similarity(query_embedding or [], fact.get("embedding") or [])
            temporal_weight = 1.0 if fact.get("invalid_at") is None else 0.65
            score = (0.5 * lexical + 0.2 * entity_bonus + 0.3 * semantic) * temporal_weight
            ranked.append({**fact, "score": score})
        ranked.sort(key=lambda item: (item["score"], item["created_at"]), reverse=True)
        return ranked[:num_results]

    async def remove_episode(self, episode_id: str):
        episode = self.episodes.pop(episode_id, None)
        removed_fact_ids = set()
        for fact_id, fact in list(self.facts.items()):
            episode_ids = [value for value in fact.get("episode_ids", []) if value != episode_id]
            fact["episode_ids"] = episode_ids
            if episode is not None:
                fact["source_event_ids"] = [
                    value for value in fact.get("source_event_ids", [])
                    if value != episode.get("name")
                ]
            if not episode_ids:
                removed_fact_ids.add(fact_id)
                self.facts.pop(fact_id, None)
        for fact in self.facts.values():
            if fact.get("invalidated_by") in removed_fact_ids:
                fact["invalidated_by"] = None
                fact["invalid_at"] = None
        for key, entity in list(self.entities.items()):
            entity["episodes"] = [value for value in entity.get("episodes", []) if value != episode_id]
            if not entity["episodes"]:
                self.entities.pop(key, None)
        self._persist()

    def export_state(self) -> dict[str, Any]:
        return {
            "episodes": self.episodes,
            "entities": self.entities,
            "facts": self.facts,
        }

    def import_state(self, state: Mapping[str, Any], *, persist: bool = True) -> None:
        self.episodes = {
            str(key): dict(value)
            for key, value in state.get("episodes", {}).items()
            if isinstance(value, Mapping)
        }
        self.entities = {
            str(key): dict(value)
            for key, value in state.get("entities", {}).items()
            if isinstance(value, Mapping)
        }
        self.facts = {
            str(key): dict(value)
            for key, value in state.get("facts", {}).items()
            if isinstance(value, Mapping)
        }
        if persist:
            self._persist()

    async def close(self):
        self._persist()
        self._kuzu.close()
        self._kuzu_db.close()
        self._store.close()
