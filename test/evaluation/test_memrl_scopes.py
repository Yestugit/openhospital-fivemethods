"""Finite training completes each MemRL scope without changing native batches."""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest

from mematlas.core.config import MethodConfig
from mematlas.core.episodes import EpisodeResult
from mematlas.core.evaluation import BenchmarkCase
from mematlas.core.models import MemoryEpisode, MemoryEvent, WriteReceipt
from mematlas.core.plugin import MethodServices
from mematlas.core.scope import scope_key
from mematlas.evaluation.execution import _EvaluationRun
from mematlas.evaluation.experience_memory import EvaluationHistory
from mematlas.methods.memrl.method import MemRLMemory
from mematlas.methods.memrl.qa import MemRLQA


class _NativeService:
    """Record native service requests; the real adapter handles batching."""

    def __init__(self):
        self.tasks = []
        self.calls = []

    async def add_memories(self, **kwargs):
        tasks = kwargs["task_descriptions"]
        self.tasks.extend(tasks)
        self.calls.append(("add", len(tasks)))
        return [(task, "memory:" + task) for task in tasks]

    async def update_values(self, successes, retrieved_ids):
        self.calls.append(("update", len(successes)))
        return {}


class _LearningMemory:
    method_id = "memrl"

    def __init__(self, adapter):
        from mematlas.methods.memrl.method import DESCRIPTOR
        self.capabilities = DESCRIPTOR.capabilities
        self.adapter = adapter
        self.episodes = []

    async def add(self, unit, *, scope):
        from mematlas.runtime.history import EpisodeHistory
        if not hasattr(self, "_history"):
            self._history = EpisodeHistory()
        episode = replace(unit.as_episode(), scope=scope,
            events=[replace(event, scope=scope, episode_id=unit.episode_ref) for event in unit.events])
        self._history.record_episode(episode)
        return WriteReceipt(unit.write_ref, unit.episode_ref)

    async def episode_end(self, episode_ref, *, scope):
        key = self._history.identity(episode_ref, scope)
        episode = self._history.episode(key)
        await self.adapter._learn(episode)
        self.episodes.append(episode)
        self._history.record_result(EpisodeResult(key, feedback="valid", learning="submitted"))

    def _complete_episode(self, episode, scope):
        return replace(episode, scope=scope)

    def _scope(self, scope):
        return scope


class _Engine:
    def __init__(self, batches, position=0):
        self.plan = SimpleNamespace(batches=lambda: batches)
        self._position = position

    async def record_case_progress(self, position):
        self._position = position


def _cases():
    # Equivalent dictionaries with different insertion order share one scope.
    identities = [
        ("a1", {"user_id": "a", "agent_id": "qa"}),
        ("b1", {"user_id": "b", "agent_id": "qa"}),
        ("a2", {"agent_id": "qa", "user_id": "a"}),
        ("b2", {"agent_id": "qa", "user_id": "b"}),
    ]
    return tuple(BenchmarkCase(identity, identity, scope=scope,
        episode=MemoryEpisode(identity, task=identity, success=True, scope=scope,
            events=[MemoryEvent("message", "Execute " + identity)]))
        for identity, scope in identities)


def _execution(tmp_path, train, batches):
    services = {}

    async def factory(**kwargs):
        service = _NativeService()
        services[scope_key(kwargs["scope"])] = service
        return service

    adapter = MemRLMemory(MethodConfig("memrl", {"batch_size": 32}),
        MethodServices(objects={"memrl.service_factory": factory}, artifact_dir=tmp_path))
    memory = _LearningMemory(adapter)
    experience = SimpleNamespace(core_memory=memory, metadata={}, history=EvaluationHistory())
    execution = _EvaluationRun(None, None, train, (), None, None, None)
    execution.qa = MemRLQA(None, MethodServices())
    execution.engine = _Engine(batches)
    return execution, experience, adapter, services


@pytest.mark.asyncio
async def test_interleaved_scopes_finish_their_own_tail_after_intermediate_checkpoint(tmp_path):
    train = _cases()
    batches = [("a1", "b1"), ("a2", "b2")]
    execution, experience, adapter, services = _execution(tmp_path, train, batches)

    await execution._train_batch(batches[0], experience, 1)
    assert all(not service.calls for service in services.values())
    assert all(not episode.extensions.get("memrl", {}).get("end_of_training", False)
               for episode in experience.core_memory.episodes)

    await execution._train_batch(batches[1], experience, 2)
    assert not adapter._pending_batch
    assert services[scope_key(train[0].scope)].tasks == ["a1", "a2"]
    assert services[scope_key(train[1].scope)].tasks == ["b1", "b2"]
    assert all(service.calls == [("add", 2), ("update", 2)] for service in services.values())


@pytest.mark.asyncio
async def test_last_cases_are_computed_from_selected_training_subset(tmp_path):
    # ExperimentRunner applies train_limit before constructing _EvaluationRun.
    train = _cases()[:3]
    execution, experience, adapter, services = _execution(tmp_path, train, [("a1", "b1", "a2")])
    await execution._train_batch(("a1", "b1", "a2"), experience, 1)

    assert not adapter._pending_batch
    assert services[scope_key(train[0].scope)].tasks == ["a1", "a2"]
    assert services[scope_key(train[1].scope)].tasks == ["b1"]
    assert [episode.episode_id for episode in experience.core_memory.episodes
            if episode.extensions.get("memrl", {}).get("end_of_training")] == ["b1", "a2"]


@pytest.mark.asyncio
async def test_restored_execution_keeps_scope_ends_and_cached_episode_data(tmp_path):
    train = _cases()
    batches = [("a1", "b1"), ("a2", "b2")]
    execution, experience, adapter, services = _execution(tmp_path, train, batches)
    await execution._train_batch(batches[0], experience, 1)

    # A failed submission may already have saved its executed episode for retry.
    cached = replace(train[2].episode, task="cached actual trajectory",
                     extensions={"memrl": {"task_id": 23, "end_of_training": True}})
    experience.metadata["training_episodes"]["a2"] = cached.to_dict()
    experience.metadata = deepcopy(experience.metadata)
    restored = _EvaluationRun(None, None, train, (), None, None, None)
    restored.qa = MemRLQA(None, MethodServices())
    restored.engine = _Engine(batches, position=2)
    await restored._train_batch(batches[1], experience, 2)

    assert not adapter._pending_batch
    assert services[scope_key(train[0].scope)].tasks == ["a1", "cached actual trajectory"]
    assert services[scope_key(train[1].scope)].tasks == ["b1", "b2"]
    saved = experience.metadata["training_episodes"]
    assert saved["a2"]["extensions"]["memrl"] == {"task_id": 23, "end_of_training": True}
    assert saved["b2"]["extensions"]["memrl"]["end_of_training"] is True
    assert restored.engine._position == 4
