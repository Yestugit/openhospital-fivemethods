from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping
from uuid import uuid4

from mematlas.core.plugin import MethodServices
from mematlas.methods.core_utils import offline_embedding, parse_json_object
from mematlas.methods.storage import SQLiteState, configured_file, load_mapping


_DEFAULT_INSTRUCTIONS = """You maintain dense long-term memories for an AI agent.
Extract durable facts, preferences, relationships, procedures, and useful episode summaries.
Compare the new conversation with existing memories. Consolidate duplicates, update facts that
changed, and delete memories that became false or redundant. Do not store greetings or filler."""


@dataclass
class LangMemRecord:
    key: str
    value: dict[str, Any]
    score: float = 1.0
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def dict(self) -> dict[str, Any]:
        return asdict(self)


class LangMemCore:
    """LangMem-style extraction, consolidation, and namespace-scoped search."""

    def __init__(
        self,
        services: MethodServices,
        *,
        query_limit: int = 5,
        instructions: str | None = None,
        enable_inserts: bool = True,
        enable_updates: bool = True,
        enable_deletes: bool = False,
        storage_path: str | None = None,
        max_steps: int = 1,
        **_: Any,
    ):
        self.services = services
        self.query_limit = query_limit
        self.instructions = instructions or _DEFAULT_INSTRUCTIONS
        self.enable_inserts = enable_inserts
        self.enable_updates = enable_updates
        self.enable_deletes = enable_deletes
        self.max_steps = max_steps
        self.records: dict[str, dict[str, LangMemRecord]] = {}
        self._store = SQLiteState(
            configured_file(services, "langmem", storage_path, "langgraph_store.db")
        )
        try:
            from langgraph.store.memory import InMemoryStore
        except ImportError as exc:
            raise RuntimeError(
                "LangMem requires langgraph; install mematlas[native-methods]"
            ) from exc
        self._langgraph_store = InMemoryStore(index={"dims": 128, "embed": self._embed, "fields": ["content"]})
        self._synced: dict[str, dict[str, dict[str, Any]]] = {}
        self.import_state(load_mapping(self._store, "store"), persist=False)

    def _persist(self) -> None:
        self._store.save("store", self.export_state())

    @staticmethod
    def _namespace(config: Mapping[str, Any] | None) -> str:
        configurable = dict((config or {}).get("configurable", {}))
        return str(configurable.get("langgraph_user_id", "default"))

    @staticmethod
    def _normalize_messages(messages: list[Any]) -> list[dict[str, str]]:
        normalized = []
        for message in messages:
            if isinstance(message, Mapping):
                content = str(message.get("content", "")).strip()
                role = str(message.get("role", "user"))
            else:
                content = str(getattr(message, "content", "")).strip()
                role = str(getattr(message, "type", "user"))
            if content:
                normalized.append({"role": role, "content": content})
        return normalized

    async def _operations(
        self,
        messages: list[dict[str, str]],
        bucket: Mapping[str, LangMemRecord],
    ) -> list[dict[str, Any]] | None:
        if self.services.llm is None:
            return None
        existing = [{"id": record.key, **dict(record.value)} for record in bucket.values()]
        prompt = (
            f"{self.instructions}\n\n"
            "Return one JSON object with an operations array. Each operation uses action "
            "ADD, UPDATE, or DELETE. ADD requires content and may include memory_type and metadata. "
            "UPDATE requires id and the complete replacement content. DELETE requires id. "
            f"Allowed operations: inserts={self.enable_inserts}, updates={self.enable_updates}, "
            f"deletes={self.enable_deletes}.\n"
            f"Existing memories: {existing}\n"
            f"New conversation: {messages}"
        )
        response = await self.services.llm.complete(
            [
                {"role": "system", "content": "You are a memory subroutine for an AI."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )
        parsed = parse_json_object(str(response))
        operations = parsed.get("operations") if parsed else None
        if not isinstance(operations, list):
            raise ValueError("LangMem model response must contain an operations array")
        return [dict(item) for item in operations if isinstance(item, Mapping)] + ([{"action": "DONE"}] if parsed.get("done") else [])

    @staticmethod
    def _find_duplicate(bucket: Mapping[str, LangMemRecord], content: str) -> LangMemRecord | None:
        normalized = " ".join(content.lower().split())
        return next(
            (
                record
                for record in bucket.values()
                if " ".join(str(record.value.get("content", "")).lower().split()) == normalized
            ),
            None,
        )

    def _add(
        self,
        bucket: dict[str, LangMemRecord],
        content: str,
        *,
        role: str | None = None,
        memory_type: str = "semantic",
        metadata: Mapping[str, Any] | None = None,
    ) -> LangMemRecord | None:
        content = content.strip()
        if not content:
            return None
        duplicate = self._find_duplicate(bucket, content)
        now = datetime.now(timezone.utc).isoformat()
        if duplicate is not None:
            if role:
                duplicate.value["role"] = role
            duplicate.updated_at = now
            return duplicate
        if not self.enable_inserts:
            return None
        key = uuid4().hex
        record = LangMemRecord(
            key,
            {
                "content": content,
                "memory_type": memory_type,
                "metadata": dict(metadata or {}),
                **({"role": role} if role else {}),
            },
            created_at=now,
            updated_at=now,
        )
        bucket[key] = record
        return record

    async def ainvoke(self, data: Mapping[str, Any], config=None):
        namespace = self._namespace(config)
        bucket = self.records.setdefault(namespace, {})
        messages = self._normalize_messages(list(data.get("messages", [])))
        if not messages:
            return list(bucket.values())
        await self._sync_namespace(namespace)
        candidates = {}
        sizes = set()
        for step in range(self.query_limit // 4):
            size = min(len(messages), 1 << step)
            if size in sizes:
                break
            sizes.add(size)
            query = "\n\n".join(f"{message['role']}: {message['content']}" for message in messages[-size:])
            for record in await self.asearch(query=query, limit=10, config=config):
                candidates[record.key] = record
        relevant = {record.key: record for record in sorted(candidates.values(), key=lambda item: item.score,
                                                            reverse=True)[:self.query_limit]}
        operations = await self._operations(messages, relevant)
        if operations is None:
            for message in messages:
                self._add(bucket, message["content"], role=message["role"])
            self._persist()
            return list(bucket.values())

        max_steps = int(data.get("max_steps", self.max_steps))
        if max_steps < 1:
            raise ValueError("LangMem max_steps must be positive")
        for step in range(max_steps):
            before = set(bucket)
            self._apply_operations(bucket, operations, set(relevant))
            if not operations or any(str(item.get("action", "")).upper() == "DONE" for item in operations):
                break
            if step + 1 < max_steps:
                relevant = {key: bucket[key] for key in set(relevant) | (set(bucket) - before) if key in bucket}
                operations = await self._operations(messages + [{"role": "user", "content":
                    "Review the previous memory changes, correct errors or missing information, and return done=true when finished."}], relevant)
        self._persist()
        await self._sync_namespace(namespace)
        return list(bucket.values())

    def _apply_operations(self, bucket, operations, allowed_ids):
        now = datetime.now(timezone.utc).isoformat()
        for operation in operations:
            action = str(operation.get("action", "")).upper()
            memory_id = str(operation.get("id") or operation.get("memory_id") or "")
            if action in {"UPDATE", "DELETE", "REMOVE"} and memory_id not in allowed_ids:
                raise ValueError("LangMem operation references a memory outside the retrieved candidates")
            if action in {"ADD", "CREATE", "INSERT"}:
                self._add(
                    bucket,
                    str(operation.get("content", "")),
                    memory_type=str(operation.get("memory_type", "semantic")),
                    metadata=(
                        operation.get("metadata")
                        if isinstance(operation.get("metadata"), Mapping)
                        else {}
                    ),
                )
            elif action == "UPDATE" and self.enable_updates and memory_id in bucket:
                content = str(operation.get("content", "")).strip()
                if content:
                    bucket[memory_id].value.update({
                        "content": content,
                        "memory_type": str(
                            operation.get("memory_type")
                            or bucket[memory_id].value.get("memory_type", "semantic")
                        ),
                    })
                    if isinstance(operation.get("metadata"), Mapping):
                        bucket[memory_id].value["metadata"] = dict(operation["metadata"])
                    bucket[memory_id].updated_at = now
            elif action in {"DELETE", "REMOVE"} and self.enable_deletes:
                bucket.pop(memory_id, None)

    async def _embed(self, texts: list[str]) -> list[list[float]]:
        if self.services.embedder is not None:
            return await self.services.embedder.embed(texts)
        return [offline_embedding(text) for text in texts]

    async def _sync_namespace(self, namespace: str) -> None:
        from copy import deepcopy
        values = {key: record.value for key, record in self.records.get(namespace, {}).items()}
        previous = self._synced.get(namespace, {})
        for key in previous.keys() - values.keys():
            await self._langgraph_store.adelete(("memories", namespace), key)
        for key, value in values.items():
            if previous.get(key) != value:
                await self._langgraph_store.aput(("memories", namespace), key, value)
        self._synced[namespace] = deepcopy(values)

    async def asearch(self, *, query: str = "", limit: int | None = None, config=None, **_: Any):
        namespace = self._namespace(config)
        await self._sync_namespace(namespace)
        results = await self._langgraph_store.asearch(
            ("memories", namespace), query=query or None, limit=limit or self.query_limit,
        )
        bucket = self.records.get(namespace, {})
        return [LangMemRecord(
            result.key, dict(result.value), result.score if result.score is not None else 1.0,
            bucket[result.key].created_at, bucket[result.key].updated_at,
        ) for result in results]

    async def adelete(self, key: str, config=None):
        namespace = self._namespace(config)
        self.records.get(namespace, {}).pop(key, None)
        await self._sync_namespace(namespace)
        self._persist()

    def export_state(self) -> dict[str, Any]:
        return {
            namespace: {key: record.dict() for key, record in bucket.items()}
            for namespace, bucket in self.records.items()
        }

    def import_state(self, state: Mapping[str, Any], *, persist: bool = True) -> None:
        restored: dict[str, dict[str, LangMemRecord]] = {}
        for namespace, raw_bucket in state.items():
            if not isinstance(raw_bucket, Mapping):
                continue
            bucket = {}
            for key, payload in raw_bucket.items():
                if not isinstance(payload, Mapping):
                    continue
                values = dict(payload)
                values["key"] = str(values.get("key") or key)
                values["value"] = dict(values.get("value", {}))
                bucket[str(key)] = LangMemRecord(**values)
            restored[str(namespace)] = bucket
        self.records = restored
        if persist:
            self._persist()

    async def close(self) -> None:
        self._persist()
        self._store.close()
