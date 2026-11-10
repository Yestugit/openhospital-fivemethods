"""A-Mem 公共接口适配：组织事件写入、经历去重、检索贡献及状态保存恢复。

笔记分析、关联与自动演化由 core.py 处理；本模块不提供外部演化入口。
"""
from __future__ import annotations

from copy import deepcopy
import json
from typing import Any, Mapping

from mematlas.core.config import MethodConfig
from mematlas.core.models import Content, MemoryContribution, MemoryEpisode, MemoryEvent, Scope, content_text
from mematlas.core.plugin import BaseMemoryMethod, MethodContext, MethodPlugin, MethodServices
from mematlas.core.scope import scope_key
from mematlas.methods.a_mem.core import AgenticMemoryCore
from mematlas.methods.common import normalize_items
from mematlas.methods.manifest import load_method_descriptor
from mematlas.methods.services import prepare_services


DESCRIPTOR = load_method_descriptor(__file__)


class AMemMemory(BaseMemoryMethod):
    def __init__(self, config: MethodConfig, services: MethodServices):
        self.params = dict(config.params)
        self.services = services
        self.memory: AgenticMemoryCore | None = None
        self._seen: set[str] = set()
        self._sources: dict[str, list[str]] = {}
        self._active_scope: str | None = None

    def _ensure_scope(self, scope: Scope, *, read_only: bool = False) -> None:
        key = scope_key(scope)
        if self._active_scope is None:
            if not read_only:
                self._active_scope = key
        elif key != self._active_scope:
            raise ValueError("A-Mem uses one Memory instance per scope")

    async def setup(self, context: MethodContext) -> None:
        prepare_services(self.params, self.services, method_id="a-mem", llm=True, embedding=True)
        self.memory = AgenticMemoryCore(
            self.services,
            evo_threshold=int(self.params.get("evo_threshold", 100)),
            neighbor_k=int(self.params.get("neighbor_k", 5)),
            link_threshold=float(self.params.get("link_threshold", 0.0)),
            storage_path=self.params.get("storage_path"),
            offline=self.params.get("implementation_mode") == "offline",
        )

    async def _write(self, events: list[MemoryEvent], *, episode: bool = False) -> None:
        if not events:
            return
        if episode:
            content = "\n\n".join(
                f"[{event.role or event.kind.value}]\n{content_text(event.content)}" for event in events
            )
        else:
            content = content_text(events[0].content)
        identity = await self.memory.add_note(
            content, timestamp=events[-1].timestamp.strftime("%Y%m%d%H%M"),
            category="episode" if episode else events[0].kind.value,
            source_event_ids=[event.event_id for event in events],
        )
        self._sources[identity] = [event.event_id for event in events]
        self._seen.update(self._sources[identity])

    async def _observe(self, event: MemoryEvent) -> None:
        self._ensure_scope(event.scope)
        if self.params.get("write_policy", "event") == "event" and event.event_id not in self._seen:
            if content_text(event.content).strip():
                await self._write([event])

    async def _learn(self, episode: MemoryEpisode, *, observed_event_ids: frozenset[str]) -> None:
        # observed_event_ids records delivery, not successful note creation.
        # Our own written IDs also cover direct learn() and restored snapshots.
        self._ensure_scope(episode.scope)
        events = [event for event in episode.events
                  if event.event_id not in self._seen and content_text(event.content).strip()]
        if self.params.get("write_policy", "event") == "episode":
            await self._write(events, episode=True)
        else:
            for event in events:
                await self._write([event])

    async def search(self, query: Content, *, level_ref=None, top_k: int | None, scope: Scope,
                       extensions: Mapping[str, Any]) -> MemoryContribution:
        self._ensure_scope(scope, read_only=bool(extensions.get("read_only")))
        search = self.memory.search_agentic if self.params.get("agentic_search", True) else self.memory.search
        raw = await search(content_text(query), k=top_k or int(self.params.get("top_k", 5)))
        for item in raw:
            sources = list(self._sources.get(item["id"], []))
            # These are conservative dependencies of the model inputs that made
            # context/tags, not a claim that every input fact survives in the text.
            for event_id in item.get("evolution_source_event_ids", []):
                if event_id not in sources:
                    sources.append(event_id)
            item["source_event_ids"] = sources
        # A-Mem's formatted context includes temporal and semantic metadata.
        # DefaultQA already consumes this public field, so no method QA is needed.
        context = [{"content": (
            f"talk start time:{item['timestamp']}\tmemory content: {item['content']}\t"
            f"memory context: {item['context']}\tmemory keywords: {item['keywords']}\t"
            f"memory tags: {item['tags']}"
        ), "source_item_indices": [index]} for index, item in enumerate(raw)]
        return MemoryContribution(items=normalize_items(raw, method_id="a-mem", memory_type="linked_note"),
                                  context=context)

    async def detail(self, memory_id, *, scope):
        self._ensure_scope(scope, read_only=True)
        item = self.memory.read(memory_id)
        item["source_event_ids"] = list(self._sources.get(memory_id, ()))
        return MemoryContribution(items=normalize_items([item], method_id="a-mem", memory_type="linked_note"))

    async def update(self, memory_id, content, *, scope):
        self._ensure_scope(scope)
        if not isinstance(content, str):
            raise ValueError("A-MEM explicit updates currently accept text")
        await self.memory.update(memory_id, content=content)

    async def delete(self, memory_id, *, scope):
        self._ensure_scope(scope)
        await self.memory.delete(memory_id)
        self._sources.pop(memory_id, None)

    async def reset(self, scope: Scope) -> None:
        self._ensure_scope(scope)
        self.memory.import_state({"memories": {}, "evo_count": 0})
        self._seen.clear()
        self._sources.clear()
        self._active_scope = None

    async def save(self) -> Mapping[str, Any]:
        return deepcopy({"version": 3, "core": self.memory.export_state(),
                         "active_scope": self._active_scope, "seen_events": sorted(self._seen),
                         "sources": self._sources})

    async def load(self, state: Mapping[str, Any]) -> None:
        if state.get("version") != 3:
            raise ValueError("A-Mem state requires version 3 with persisted evolution input lineage")
        core = deepcopy(state["core"])
        seen = state["seen_events"]
        sources = state["sources"]
        owner = state["active_scope"]
        if owner is not None:
            if scope_key(json.loads(owner)) != owner:
                raise ValueError("Invalid A-Mem saved scope")
        if not isinstance(seen, list) or any(not isinstance(value, str) or not value for value in seen):
            raise ValueError("Invalid A-Mem written event IDs")
        if set(sources) != set(core["memories"]) or any(
            not isinstance(ids, list) or not ids or any(event_id not in seen for event_id in ids)
            for ids in sources.values()
        ):
            raise ValueError("Invalid A-Mem note provenance")
        for identity, note in core["memories"].items():
            dependencies = note.get("evolution_source_event_ids")
            if note.get("source_event_ids") != sources[identity] or not isinstance(dependencies, list) or any(
                event_id not in seen for event_id in dependencies
            ):
                raise ValueError("Invalid A-Mem evolution input provenance")
        self.memory.import_state(core)
        self._active_scope, self._seen, self._sources = owner, set(seen), deepcopy(sources)

    async def close(self) -> None:
        if self.memory is not None:
            await self.memory.close()


class AMemPlugin(MethodPlugin):
    descriptor = DESCRIPTOR

    def create_method(self, config: MethodConfig, services: MethodServices) -> BaseMemoryMethod:
        return AMemMemory(config, services)


PLUGIN = AMemPlugin()
