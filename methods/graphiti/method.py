from __future__ import annotations

from typing import Any, Mapping

from mematlas.core.config import MethodConfig
from mematlas.core.models import DecisionContext, EventKind, MemoryContribution, MemoryEvent, MemoryScope
from mematlas.core.plugin import BaseMemoryMethod, MethodPlugin, MethodContext, MethodServices, call_maybe_async
from mematlas.core.telemetry import UsageLedger, UsageRecord
from mematlas.methods.common import first_value, normalize_items
from mematlas.methods.graphiti.core import GraphitiCore
from mematlas.methods.manifest import load_method_descriptor
from mematlas.methods.services import prepare_services


DESCRIPTOR = load_method_descriptor(__file__)


class _TrackedAsyncEndpoint:
    def __init__(
        self,
        delegate: Any,
        usage: UsageLedger,
        *,
        provider: str,
        extra_body: Mapping[str, Any] | None = None,
        ensure_json_instruction: bool = False,
    ) -> None:
        self._delegate = delegate
        self._usage = usage
        self._provider = provider
        self._extra_body = dict(extra_body or {})
        self._ensure_json_instruction = ensure_json_instruction

    async def create(self, *args: Any, **kwargs: Any) -> Any:
        if self._extra_body:
            extra_body = dict(kwargs.get("extra_body") or {})
            extra_body.update({key: value for key, value in self._extra_body.items() if key not in extra_body})
            kwargs["extra_body"] = extra_body
        response_format = kwargs.get("response_format") or {}
        if self._ensure_json_instruction and response_format.get("type") in {"json_object", "json_schema"}:
            messages = [dict(message) for message in kwargs.get("messages", [])]
            if messages:
                messages[0]["content"] = (
                    str(messages[0].get("content", ""))
                    + "\nReturn the response as a valid JSON object."
                )
                kwargs["messages"] = messages
        response = await self._delegate.create(*args, **kwargs)
        raw_usage = getattr(response, "usage", None)
        if raw_usage is not None:
            prompt_tokens = int(
                getattr(raw_usage, "prompt_tokens", None)
                or getattr(raw_usage, "input_tokens", None)
                or getattr(raw_usage, "total_tokens", 0)
                or 0
            )
            completion_tokens = int(
                getattr(raw_usage, "completion_tokens", None)
                or getattr(raw_usage, "output_tokens", 0)
                or 0
            )
            self._usage.add(UsageRecord(
                provider=self._provider,
                model=str(kwargs.get("model", "unknown")),
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            ))
        return response


class _TrackedChat:
    def __init__(self, delegate: Any, usage: UsageLedger, extra_body: Mapping[str, Any] | None):
        self._delegate = delegate
        self.completions = _TrackedAsyncEndpoint(
            delegate.completions,
            usage,
            provider="openai_compatible",
            extra_body=extra_body,
            ensure_json_instruction=True,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)


class _TrackedAsyncOpenAI:
    def __init__(
        self,
        delegate: Any,
        usage: UsageLedger,
        *,
        chat_extra_body: Mapping[str, Any] | None = None,
    ) -> None:
        self._delegate = delegate
        self.chat = _TrackedChat(delegate.chat, usage, chat_extra_body)
        self.embeddings = _TrackedAsyncEndpoint(
            delegate.embeddings,
            usage,
            provider="openai_compatible_embedding",
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)


class GraphitiMemory(BaseMemoryMethod):
    def __init__(self, config: MethodConfig, services: MethodServices):
        self.params = dict(config.params)
        self.services = services
        self.graph = services.object("graphiti.instance")
        self._owns_graph = self.graph is None
        self.episode_type = None
        self._episode_ids: dict[str, tuple[str, MemoryScope]] = {}

    async def setup(self, context: MethodContext) -> None:
        if self.graph is None:
            prepare_services(self.params, self.services, method_id="graphiti", llm=True, embedding=True)
        if self.graph is None:
            self.graph = GraphitiCore(
                self.services,
                storage_path=self.params.get("storage_path")
                or self.params.get("data_storage_path"),
                **{key: value for key, value in self.params.items()
                   if key not in {"storage_path", "data_storage_path"}},
            )
            self._owns_graph = True
        if self.params.get("build_indices", True) and hasattr(self.graph, "build_indices_and_constraints"):
            await call_maybe_async(
                self.graph.build_indices_and_constraints,
                delete_existing=False,
            )

    @staticmethod
    def _group_id(scope: MemoryScope) -> str:
        return scope.team_id or scope.user_id or scope.agent_id or scope.run_id or scope.key()

    async def observe(self, event: MemoryEvent) -> None:
        if not event.text or event.event_id in self._episode_ids:
            return
        source = None
        if self.episode_type is not None:
            source = self.episode_type.message if event.kind == EventKind.MESSAGE else self.episode_type.text
        kwargs = {
            "name": str(event.metadata.get("name", event.event_id)),
            "episode_body": event.text,
            "source_description": str(event.metadata.get("source_description", event.kind.value)),
            "reference_time": event.timestamp,
            "group_id": self._group_id(event.scope),
            "update_communities": bool(self.params.get("update_communities", False)),
        }
        if source is not None:
            kwargs["source"] = source
        result = await call_maybe_async(self.graph.add_episode, **kwargs)
        episode = first_value(result, ("episode",), result)
        episode_id = str(first_value(episode, ("uuid", "id"), event.event_id))
        self._episode_ids[event.event_id] = (episode_id, event.scope)

    async def retrieve(self, context: DecisionContext) -> MemoryContribution:
        kwargs: dict[str, Any] = {"group_ids": [self._group_id(context.scope)]}
        if context.top_k is not None:
            kwargs["num_results"] = context.top_k
        raw = await call_maybe_async(self.graph.search, context.text, **kwargs)
        items = normalize_items(raw, method_id="graphiti", memory_type="temporal_fact")
        return MemoryContribution.from_items(items, metadata={"method_id": "graphiti"})

    async def reset(self, scope: MemoryScope | None = None) -> None:
        targets = [
            (event_id, episode_id)
            for event_id, (episode_id, event_scope) in self._episode_ids.items()
            if scope is None or event_scope.key() == scope.key()
        ]
        for event_id, episode_id in targets:
            await call_maybe_async(self.graph.remove_episode, episode_id)
            self._episode_ids.pop(event_id, None)

    async def save_state(self) -> Mapping[str, Any]:
        graph_state = None
        if hasattr(self.graph, "export_state"):
            graph_state = await call_maybe_async(self.graph.export_state)
        return {
            "version": 1,
            "event_episodes": {
                event_id: {
                    "episode_id": episode_id,
                    "scope": scope.to_dict(),
                }
                for event_id, (episode_id, scope) in self._episode_ids.items()
            },
            "graph": graph_state,
        }

    async def load_state(self, state: Mapping[str, Any]) -> None:
        graph_state = state.get("graph")
        if hasattr(self.graph, "import_state") and isinstance(graph_state, Mapping):
            await call_maybe_async(self.graph.import_state, graph_state)
        event_episodes = state.get("event_episodes", {})
        if not isinstance(event_episodes, Mapping):
            raise ValueError("Graphiti snapshot event_episodes must be a mapping")
        restored = {}
        for event_id, payload in event_episodes.items():
            if not isinstance(payload, Mapping):
                continue
            restored[str(event_id)] = (
                str(payload.get("episode_id", event_id)),
                MemoryScope.from_dict(payload.get("scope")),
            )
        self._episode_ids = restored

    async def close(self) -> None:
        if self._owns_graph and self.graph is not None and hasattr(self.graph, "close"):
            await call_maybe_async(self.graph.close)


class GraphitiPlugin(MethodPlugin):
    descriptor = DESCRIPTOR

    def create_method(self, config: MethodConfig, services: MethodServices) -> BaseMemoryMethod:
        return GraphitiMemory(config, services)


PLUGIN = GraphitiPlugin()
