from __future__ import annotations

import json
from collections import defaultdict
from typing import Any, Mapping

from mematlas.core.config import MethodConfig
from mematlas.core.models import (
    ContentBlock,
    DecisionContext,
    EventKind,
    MemoryContribution,
    MemoryEpisode,
    MemoryEvent,
    MemoryItem,
    MemoryScope,
)
from mematlas.core.plugin import BaseMemoryMethod, MethodPlugin, MethodContext, MethodServices, call_maybe_async
from mematlas.methods.common import first_value
from mematlas.methods.langmem.core import LangMemCore
from mematlas.methods.manifest import load_method_descriptor
from mematlas.methods.services import prepare_services


DESCRIPTOR = load_method_descriptor(__file__)


class LangMemMemory(BaseMemoryMethod):
    def __init__(self, config: MethodConfig, services: MethodServices):
        self.params = dict(config.params)
        self.services = services
        self.manager = services.object("langmem.manager")
        self._owns_manager = self.manager is None
        self._buffers: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._scopes: dict[str, MemoryScope] = {}

    async def setup(self, context: MethodContext) -> None:
        if self.manager is None:
            prepare_services(self.params, self.services, method_id="langmem", llm=True, embedding=True)
        if self.manager is None:
            self.manager = LangMemCore(
                self.services,
                query_limit=int(self.params.get("query_limit", 5)),
                instructions=self.params.get("instructions"),
                max_steps=int(self.params.get("max_steps", 1)),
                enable_inserts=bool(self.params.get("enable_inserts", True)),
                enable_updates=bool(self.params.get("enable_updates", True)),
                enable_deletes=bool(self.params.get("enable_deletes", False)),
                storage_path=self.params.get("storage_path")
                or self.params.get("data_storage_path"),
            )
            self._owns_manager = True

    @staticmethod
    def _user_id(scope: MemoryScope) -> str:
        return scope.user_id or scope.agent_id or scope.session_id or scope.key()

    def _runnable_config(self, scope: MemoryScope) -> dict[str, Any]:
        return {"configurable": {"langgraph_user_id": self._user_id(scope)}}

    async def _flush(self, scope: MemoryScope) -> None:
        key = scope.key()
        messages = self._buffers.get(key, [])
        if not messages:
            return
        await call_maybe_async(
            self.manager.ainvoke,
            {"messages": list(messages)},
            config=self._runnable_config(scope),
        )
        self._buffers[key].clear()

    async def observe(self, event: MemoryEvent) -> None:
        if not event.text:
            return
        key = event.scope.key()
        role = event.role or (
            "user" if event.kind in {EventKind.MESSAGE, EventKind.OBSERVATION} else "assistant"
        )
        self._buffers[key].append({"role": role, "content": event.text})
        self._scopes[key] = event.scope
        flush_roles = set(self.params.get("flush_on_roles", ["assistant"]))
        if role in flush_roles or len(self._buffers[key]) >= int(self.params.get("max_buffer_messages", 12)):
            await self._flush(event.scope)

    async def retrieve(self, context: DecisionContext) -> MemoryContribution:
        if self.params.get("flush_before_search", False):
            await self._flush(context.scope)
        results = await call_maybe_async(
            self.manager.asearch,
            query=context.text,
            limit=context.top_k or int(self.params.get("query_limit", 5)),
            config=self._runnable_config(context.scope),
        )
        items = []
        for result in results:
            value = first_value(result, ("value", "content"), "")
            if hasattr(value, "model_dump"):
                value = value.model_dump(mode="json")
            if isinstance(value, Mapping):
                content = value.get("content") or json.dumps(value, ensure_ascii=False, default=str)
            else:
                content = str(value)
            items.append(MemoryItem(
                content=(ContentBlock.text(content),),
                memory_id=str(first_value(result, ("key", "id"), "")) or None,
                score=first_value(result, ("score",), None),
                memory_type="semantic",
                metadata={"method_id": "langmem"},
            ))
        return MemoryContribution.from_items(items, metadata={"method_id": "langmem"})

    async def learn(self, episode: MemoryEpisode) -> None:
        await self._flush(episode.result.scope)

    async def commit(self) -> None:
        for scope in list(self._scopes.values()):
            await self._flush(scope)

    async def reset(self, scope: MemoryScope | None = None) -> None:
        scopes = [scope] if scope else list(self._scopes.values())
        for target in scopes:
            if target is None:
                continue
            results = await call_maybe_async(
                self.manager.asearch,
                limit=int(self.params.get("reset_limit", 10000)),
                config=self._runnable_config(target),
            )
            for result in results:
                key = first_value(result, ("key", "id"))
                if key:
                    await call_maybe_async(
                        self.manager.adelete,
                        str(key),
                        config=self._runnable_config(target),
                    )
            self._buffers.pop(target.key(), None)

    async def save_state(self) -> Mapping[str, Any]:
        if hasattr(self.manager, "export_state"):
            manager_state = await call_maybe_async(self.manager.export_state)
            return {
                "version": 1,
                "manager": manager_state,
                "buffers": {key: list(value) for key, value in self._buffers.items()},
                "scopes": {key: scope.to_dict() for key, scope in self._scopes.items()},
            }
        state = {}
        for key, scope in self._scopes.items():
            records = await call_maybe_async(
                self.manager.asearch,
                limit=int(self.params.get("snapshot_limit", 10000)),
                config=self._runnable_config(scope),
            )
            state[key] = [
                item.dict() if hasattr(item, "dict") else str(item)
                for item in records
            ]
        return {"version": 1, "records": state}

    async def load_state(self, state: Mapping[str, Any]) -> None:
        self._buffers = defaultdict(list, {
            str(key): [dict(item) for item in value if isinstance(item, Mapping)]
            for key, value in state.get("buffers", {}).items()
            if isinstance(value, list)
        })
        self._scopes = {
            str(key): MemoryScope.from_dict(value)
            for key, value in state.get("scopes", {}).items()
            if isinstance(value, Mapping)
        }
        manager_state = state.get("manager")
        if hasattr(self.manager, "import_state") and isinstance(manager_state, Mapping):
            await call_maybe_async(self.manager.import_state, manager_state)
            return
        records = state.get("records", {})
        if not isinstance(records, Mapping):
            return
        for scope_key, values in records.items():
            scope = self._scopes.get(str(scope_key))
            if scope is None or not isinstance(values, list):
                continue
            messages = []
            for value in values:
                raw = first_value(value, ("value", "content"), "")
                content = raw.get("content", "") if isinstance(raw, Mapping) else str(raw)
                if content:
                    messages.append({"role": "user", "content": content})
            if messages:
                await call_maybe_async(
                    self.manager.ainvoke,
                    {"messages": messages},
                    config=self._runnable_config(scope),
                )

    async def close(self) -> None:
        if self._owns_manager and hasattr(self.manager, "close"):
            await call_maybe_async(self.manager.close)


class LangMemPlugin(MethodPlugin):
    descriptor = DESCRIPTOR

    def create_method(self, config: MethodConfig, services: MethodServices) -> BaseMemoryMethod:
        return LangMemMemory(config, services)


PLUGIN = LangMemPlugin()
