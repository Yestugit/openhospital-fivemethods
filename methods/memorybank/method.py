from __future__ import annotations

import json
from copy import deepcopy
import hashlib
import re
from collections import defaultdict
from datetime import date
from pathlib import Path
from uuid import uuid4
from typing import Any, Mapping

from mematlas.core.config import MethodConfig
from mematlas.core.models import (
    ContentBlock,
    DecisionContext,
    EpisodeOutcome,
    MemoryContribution,
    MemoryEpisode,
    MemoryEvent,
    MemoryItem,
    MemoryScope,
)
from mematlas.core.plugin import BaseMemoryMethod, MethodPlugin, MethodContext, MethodServices, call_maybe_async
from mematlas.methods.manifest import load_method_descriptor
from mematlas.methods.services import prepare_services
from mematlas.methods.memorybank.core import MemoryBankCore
from mematlas.methods.storage import ascii_database_path


DESCRIPTOR = load_method_descriptor(__file__)


class MemoryBankMemory(BaseMemoryMethod):
    def __init__(self, config: MethodConfig, services: MethodServices):
        self.params = dict(config.params)
        self.services = services
        self.core = MemoryBankCore(services, seed=self.params.get("seed"))
        self._prepared_users: set[str] = set()
        self._pending: dict[str, tuple[str, str]] = {}
        self._memory: dict[str, Any] = {}
        self._path: Path | None = None
        self._retrievers: dict[str, Any] = {}
        self._current_dates: dict[str, str] = {}
        self._dirty = False
        self._faiss = None
        self._index = None
        self._index_ids: list[str] = []

    @staticmethod
    def _fallback_embedding(text: str, dimensions: int = 128) -> list[float]:
        values = [0.0] * dimensions
        for token in text.lower().split():
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % dimensions
            values[index] += 1.0 if digest[4] & 1 else -1.0
        norm = sum(value * value for value in values) ** 0.5
        return [value / norm for value in values] if norm else [0.0] * dimensions

    async def _embed(self, text: str) -> list[float]:
        if self.services.embedder is not None:
            return list((await self.services.embedder.embed([text]))[0])
        return self._fallback_embedding(text)

    def _index_path(self) -> Path:
        if self._path is None:
            raise RuntimeError("MemoryBank storage is not initialized")
        return self._path.with_name("memory.faiss")

    def _rebuild_index(self) -> None:
        if self._faiss is None:
            return
        import numpy as np
        if self._path is not None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
        vectors = []
        identifiers = []
        for user in self._memory.values():
            for summary in user.get("summary", {}).values():
                if isinstance(summary, Mapping) and summary.get("embedding"):
                    vectors.append(summary["embedding"])
                    identifiers.append(str(summary["memory_id"]))
            for records in user.get("history", {}).values():
                for record in records:
                    if record.get("embedding"):
                        vectors.append(record["embedding"])
                        identifiers.append(str(record.get("memory_id")))
        self._index_ids = identifiers
        if not vectors:
            self._index = None
            target = ascii_database_path(self._index_path(), "memorybank.faiss", services=self.services)
            target.unlink(missing_ok=True)
            target.with_suffix(".ids.json").unlink(missing_ok=True)
            return
        values = np.asarray(vectors, dtype="float32")
        self._index = self._faiss.IndexFlatL2(values.shape[1])
        self._index.add(values)
        target = ascii_database_path(self._index_path(), "memorybank.faiss", services=self.services)
        target.parent.mkdir(parents=True, exist_ok=True)
        self._faiss.write_index(self._index, str(target))
        self.services.register_resource(target.with_suffix(".ids.json"))
        target.with_suffix(".ids.json").write_text(
            json.dumps(self._index_ids, ensure_ascii=False), encoding="utf-8"
        )

    def _vector_scores(self, query_vector: list[float], top_k: int) -> dict[str, float]:
        if self._index is None:
            return {}
        import numpy as np
        query = np.asarray([query_vector], dtype="float32")
        scores, indices = self._index.search(query, top_k)
        return {
            self._index_ids[index]: -float(score)
            for score, index in zip(scores[0], indices[0])
            if 0 <= index < len(self._index_ids)
        }

    async def setup(self, context: MethodContext) -> None:
        prepare_services(self.params, self.services, method_id="memorybank", llm=True, embedding=True)
        try:
            import faiss
        except ImportError as exc:
            raise RuntimeError(
                "MemoryBank requires faiss-cpu; install mematlas[native-methods]"
            ) from exc
        self._faiss = faiss
        self._path = Path(
            self.params.get("storage_path", self.services.artifact_dir / "memorybank" / "memory.json")
        ).expanduser().resolve()
        self.services.register_resource(self._path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if self._path.exists():
            self._memory = json.loads(self._path.read_text(encoding="utf-8"))
        else:
            self._save()
        self._rebuild_index()

    @staticmethod
    def _user_id(scope: MemoryScope) -> str:
        return scope.user_id or scope.agent_id or scope.session_id or scope.key()

    def _save(self) -> None:
        if self._path is None:
            return
        self._path.write_text(
            json.dumps(self._memory, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self._dirty = False
        self._retrievers.clear()
        if self._faiss is not None:
            self._rebuild_index()

    async def observe(self, event: MemoryEvent) -> None:
        if not event.text:
            return
        user_id = self._user_id(event.scope)
        observed_day = event.timestamp.date().isoformat()
        if observed_day > self._current_dates.get(user_id, ""):
            self._current_dates[user_id] = observed_day
        role = (event.role or "user").lower()
        if role in {"user", "human", "patient"}:
            previous = self._pending.get(user_id)
            text = previous[0] + "\n" + event.text if previous else event.text
            self._pending[user_id] = (text, observed_day)
            return
        pending = self._pending.pop(user_id, None)
        if pending is None:
            return
        await self._store_pair(user_id, pending[0], event.text, pending[1])

    async def _store_pair(self, user_id: str, query: str, response: str, day: str) -> None:
        user = self._memory.setdefault(user_id, {
            "name": user_id,
            "summary": {},
            "personality": {},
            "overall_history": "",
            "overall_personality": "",
            "history": {},
        })
        records = user.setdefault("history", {}).setdefault(day, [])
        user.setdefault("summary", {}).setdefault(day, "")
        user.setdefault("personality", {}).setdefault(day, "")
        safe_user_id = re.sub(r"[^A-Za-z0-9-]", "-", user_id)
        records.append({
            "query": query,
            "response": response,
            "memory_id": f"{safe_user_id}_{day}_{uuid4().hex}",
            "memory_strength": 1,
            "last_recall_date": day,
        })
        records[-1]["embedding"] = await self._embed(f"{query} {response}")
        self._dirty = True
        self._save()
        self._rebuild_index()

    async def _official_search(self, query: str, user_id: str, top_k: int, *, read_only: bool = False) -> list[MemoryItem]:
        injected = self.services.object("memorybank.retriever")
        if injected is not None:
            raw = await call_maybe_async(
                injected,
                query,
                user_id,
                top_k,
                run_sync_in_thread=True,
            )
            return [
                MemoryItem(content=(ContentBlock.text(str(value)),), memory_type="retrieved")
                for value in raw
            ]
        current_date = self.params.get("current_date") or self._current_dates.get(user_id, date.today().isoformat())
        if not read_only and user_id not in self._prepared_users:
            if self.params.get("enable_forgetting", True):
                self.core.forget(self._memory.get(user_id, {}), current_date,
                                 self.params.get("forgetting_formula", "source"))
            self._prepared_users.add(user_id)
            self._save()
        semantic_scores = self._vector_scores(await self._embed(query), len(self._index_ids))
        documents = self.core.search(
            self._memory,
            user_id=user_id,
            query=query,
            top_k=top_k,
            current_date=current_date,
            enable_forgetting=bool(self.params.get("enable_forgetting", True)),
            semantic_scores=semantic_scores,
            read_only=read_only,
        )
        if not read_only:
            self._save()
        return [MemoryItem(
            content=(ContentBlock.text(str(value["text"])),),
            memory_id=str(value.get("memory_id") or "") or None,
            score=float(value.get("score", 0.0)),
            memory_type="retrieved",
            metadata={"method_id": "memorybank", "date": value.get("date")},
        ) for value in documents]

    async def retrieve(self, context: DecisionContext) -> MemoryContribution:
        user_id = self._user_id(context.scope)
        read_only = bool(context.metadata.get("read_only", False))
        frozen = deepcopy(self._memory) if read_only else None
        try:
            items = await self._official_search(context.text, user_id, context.top_k or 6, read_only=read_only)
        finally:
            if frozen is not None and self._memory != frozen:
                self._memory = frozen
                self._save()
        user_memory = self._memory.get(user_id, {})
        context_blocks = []
        overall_history = user_memory.get("overall_history")
        if overall_history:
            label = (
                "你和用户过去的回忆总结是"
                if self.params.get("language", "en") == "cn"
                else "The summary of your past memories with the user is"
            )
            text = f"{label}: {overall_history}"
            context_blocks.append(ContentBlock.text(text, kind="overall_history", source_item_indices=[len(items)]))
            items.append(MemoryItem(
                content=(ContentBlock.text(str(overall_history)),),
                memory_id=f"overall-history-{user_id}",
                memory_type="overall_history",
                metadata={"method_id": "memorybank"},
            ))
        retrieved_text = "\n".join(
            item.text for item in items if item.memory_type == "retrieved"
        )
        if retrieved_text:
            context_blocks.append(ContentBlock.text(retrieved_text, kind="related_memory",
                source_item_indices=[i for i, item in enumerate(items) if item.memory_type == "retrieved"]))
        personality = user_memory.get("overall_personality")
        if personality:
            context_blocks.append(ContentBlock.text(str(personality), kind="personality", source_item_indices=[len(items)]))
            items.append(MemoryItem(
                content=(ContentBlock.text(str(personality)),),
                memory_id=f"personality-{user_id}",
                memory_type="personality",
                metadata={"method_id": "memorybank"},
            ))
        return MemoryContribution(
            items=tuple(items),
            context=tuple(context_blocks),
            metadata={
                "method_id": "memorybank",
                "retriever": "integrated",
                "fidelity": DESCRIPTOR.fidelity,
                "read_only": read_only,
            },
        )

    async def learn(self, episode: MemoryEpisode) -> None:
        if not self.params.get("summarize_on_episode_end", False):
            return
        await self.core.summarize(
            self._memory,
            self._user_id(episode.result.scope),
            language=self.params.get("language", "en"),
        )
        await self._embed_summaries(self._user_id(episode.result.scope))
        self._save()

    async def _embed_summaries(self, user_id: str) -> None:
        summaries = self._memory.get(user_id, {}).get("summary", {})
        for day, value in list(summaries.items()):
            summary = value if isinstance(value, dict) else {"content": str(value)}
            summaries[day] = summary
            if summary.get("content") and not summary.get("embedding"):
                summary["memory_id"] = f"{user_id}_{day}_summary"
                summary["embedding"] = await self._embed(
                    f"The summary of the conversation on {day} is: {summary['content']}")

    async def command(self, name: str, payload: Mapping[str, Any]) -> Any:
        if name != "summarize":
            return await super().command(name, payload)
        scope = MemoryScope.from_dict(payload.get("scope"))
        await self.core.summarize(
            self._memory,
            self._user_id(scope),
            language=self.params.get("language", "en"),
        )
        await self._embed_summaries(self._user_id(scope))
        self._save()
        return self._memory.get(self._user_id(scope), {})

    async def reset(self, scope: MemoryScope | None = None) -> None:
        if scope is None:
            self._prepared_users.clear()
            self._memory.clear()
            self._pending.clear()
            self._current_dates.clear()
        else:
            user_id = self._user_id(scope)
            self._prepared_users.discard(user_id)
            self._memory.pop(user_id, None)
            self._pending.pop(user_id, None)
            self._current_dates.pop(user_id, None)
        self._save()

    async def commit(self) -> None:
        for user_id, (query, day) in list(self._pending.items()):
            await self._store_pair(user_id, query, "", day)
            self._pending.pop(user_id, None)
        self._save()

    async def save_state(self) -> Mapping[str, Any]:
        return {
            "version": 1,
            "memory": json.loads(json.dumps(self._memory, ensure_ascii=False)),
            "pending": {key: list(value) for key, value in self._pending.items()},
            "current_dates": dict(self._current_dates),
            "prepared_users": sorted(self._prepared_users),
            "random_state": self.core.random.getstate(),
        }

    async def load_state(self, state: Mapping[str, Any]) -> None:
        raw_memory = state.get("memory") if state.get("version") == 1 else state
        if not isinstance(raw_memory, Mapping):
            raise ValueError("MemoryBank snapshot memory must be a mapping")
        self._memory = json.loads(json.dumps(raw_memory, ensure_ascii=False))
        self._prepared_users = set(state.get("prepared_users", []))
        if "random_state" in state:
            version, internal, gaussian = state["random_state"]
            self.core.random.setstate((version, tuple(internal), gaussian))
        self._pending = {
            str(key): (str(value[0]), str(value[1]))
            for key, value in state.get("pending", {}).items()
            if isinstance(value, (list, tuple)) and len(value) == 2
        }
        self._current_dates = {
            str(key): str(value)
            for key, value in state.get("current_dates", {}).items()
        }
        if not self._current_dates:
            for user_id, user in self._memory.items():
                if isinstance(user, Mapping) and user.get("history"):
                    self._current_dates[str(user_id)] = max(user["history"])
        self._save()
        self._rebuild_index()


class MemoryBankPlugin(MethodPlugin):
    descriptor = DESCRIPTOR

    def create_method(self, config: MethodConfig, services: MethodServices) -> BaseMemoryMethod:
        return MemoryBankMemory(config, services)


PLUGIN = MemoryBankPlugin()
