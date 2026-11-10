"""Deterministic v2 method for evaluation adapter tests."""
from copy import deepcopy
from mematlas import (BaseMemoryMethod, MethodPlugin, MethodDescriptor, MethodCapabilities,
    MemoryForm, Modality, IntegrationSlot, Operation, EventKind, MemoryContribution)

class FixtureMethod(BaseMemoryMethod):
    def __init__(self):
        self.events, self.episodes = {}, []
    async def _observe(self, event):
        self.events[(str(event.scope), event.event_id)] = event.to_dict()
    async def _learn(self, episode, *, observed_event_ids):
        for event in episode.events:
            await self._observe(event)
        self.episodes.append(episode.to_dict())
    async def search(self, query, *, level_ref=None, top_k, scope, extensions):
        items = [{"memory_id": event["event_id"], "content": event["content"],
                  "source_event_ids": [event["event_id"]]}
                 for event in self.events.values() if event["scope"] == scope]
        return MemoryContribution(items=items[:top_k] if top_k else items,
            extensions={"fixture": {"query": query, "read_only": extensions.get("read_only", False)}})
    async def flush(self):
        pass
    async def save(self):
        return deepcopy({"events": list(self.events.values()), "episodes": self.episodes})
    async def load(self, state):
        self.events = {(str(event["scope"]), event["event_id"]): event for event in state["events"]}
        self.episodes = deepcopy(state["episodes"])
    async def reset(self, scope):
        self.events = {key: value for key, value in self.events.items() if value["scope"] != scope}

class FixturePlugin(MethodPlugin):
    def __init__(self, required_feedback=False):
        self.instances = []
        self.descriptor = MethodDescriptor("fixture", "Fixture", "Evaluation contract fixture",
            MethodCapabilities(modalities=frozenset({Modality.TEXT, Modality.IMAGE}),
                forms=frozenset({MemoryForm.RECORD}),
                operations=frozenset({Operation.INDEX,Operation.ADD,Operation.SEARCH,Operation.EPISODE_END,
                    Operation.FLUSH,Operation.SAVE,Operation.LOAD,Operation.RESET}),
                integration_slots=frozenset({IntegrationSlot.CONTEXT}),
                accepted_events=frozenset(EventKind), read_only_mode="native",
                required_episode_fields=frozenset({"success", "task", "trajectory"}) if required_feedback else frozenset()),
            interface_version=3)
    def create_method(self, config, services):
        method=FixtureMethod()
        self.instances.append(method)
        return method

def registry_for(plugin=None):
    from mematlas import PluginRegistry
    registry=PluginRegistry()
    registry.register(plugin or FixturePlugin())
    return registry

def config_for(path):
    return {"method": {"method_id": "fixture"}, "scope": {"user_id": "alice"},
            "runtime": {"artifact_dir": str(path)}}


async def add_event(memory, event, *, episode_ref=None):
    """Submit a standalone event unit in regressions that exercise that granularity."""
    from mematlas import MemoryUnit
    return await memory.add(MemoryUnit(events=[event], write_ref=event.event_id,
        episode_ref=event.episode_id or episode_ref), scope=event.scope)
