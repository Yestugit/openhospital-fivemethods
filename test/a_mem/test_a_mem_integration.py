"""A-Mem v2 regressions with actual Chroma and explicit controlled providers."""
from mematlas.evaluation.validation import submit_episode
from mematlas.test.evaluation.support import add_event
from copy import deepcopy
from datetime import datetime, timezone
import json

import pytest

from mematlas import Memory, MemoryEpisode, MemoryEvent
from mematlas.core.errors import CapabilityMismatchError, ConfigurationError, MemoryStateError
from mematlas.test.a_mem.fixtures import config, services


def test_preflight_checks_real_conditions_without_allocating_storage(tmp_path):
    report = Memory.check(config(tmp_path))
    assert report["valid"], report
    assert set(report["health"]["missing_services"]) == {"llm", "embedding"}
    assert not report["ready"]
    assert Memory.check(config(tmp_path), services=services())["ready"]
    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("policy", ["event", "episode"])
@pytest.mark.parametrize("observe_first", [False, True])
async def test_complete_episode_writes_once_and_retains_sources(tmp_path, policy, observe_first):
    events = [MemoryEvent("message", "Helios port is 8443", role="user", event_id="e1"),
              MemoryEvent("message", "Port recorded", role="assistant", event_id="e2")]
    async with Memory(config(tmp_path, write_policy=policy), services=services()) as memory:
        if observe_first:
            await add_event(memory, events[0], episode_ref='one')
        episode = MemoryEpisode("one", events=events)
        await submit_episode(memory, episode)
        await submit_episode(memory, episode)
        result = await memory.search("Helios")
        assert len(result.items) == (2 if policy == "event" or observe_first else 1)
        assert {v for item in result.items for v in item["source_event_ids"]} == {"e1", "e2"}
        assert memory.stats()["episodes"]["learned_exposures"] == 1


@pytest.mark.asyncio
async def test_automatic_evolution_updates_the_selected_neighbor_in_chroma(tmp_path):
    async with Memory(config(tmp_path, neighbor_k=1), services=services(evolve=True)) as memory:
        await add_event(memory, MemoryEvent("observation", "unrelated weather", event_id="weather"))
        await add_event(memory, MemoryEvent("observation", "Helios port is 8443", event_id="port"))
        await add_event(memory, MemoryEvent("observation", "Helios port changed to 9443", event_id="update"))
        state = await memory._method.save()
        port_id = next(key for key, ids in state["sources"].items() if ids == ["port"])
        core = memory._method.memory
        assert core.memories[port_id].context == "Updated service configuration"
        metadata = core._collection.get(ids=[port_id], include=["metadatas"])["metadatas"][0]
        assert metadata["context"] == "Updated service configuration"
        assert json.loads(metadata["tags"]) == ["updated"]
        result = await memory.search("Helios")
        update = next(item for item in result.items if item["content"] == "Helios port changed to 9443")
        assert port_id in update["metadata"]["links"]


@pytest.mark.asyncio
async def test_retrieval_is_side_effect_free_including_frozen_empty_memory(tmp_path):
    async with Memory(config(tmp_path), services=services()) as memory:
        async with memory.read_only():
            assert not (await memory.search("nothing")).items
        await add_event(memory, MemoryEvent("observation", "Helios port is 8443"))
        before = deepcopy(await memory._method.save())
        await memory.search("Helios")
        async with memory.read_only():
            assert (await memory.search("Helios")).items
            with pytest.raises(MemoryStateError):
                await add_event(memory, MemoryEvent("observation", "forbidden"))
        assert await memory._method.save() == before


@pytest.mark.asyncio
async def test_save_load_preserves_vectors_links_pending_history_and_dedup(tmp_path):
    snapshot = tmp_path / "state.json"
    event = MemoryEvent("message", "Helios port is 8443", role="user", event_id="e1", episode_id="pending")
    async with Memory(config(tmp_path / "one"), services=services(evolve=True)) as memory:
        await add_event(memory, event, episode_ref='pending')
        await add_event(memory, MemoryEvent("observation", "Helios port is 9443", event_id="e2"), episode_ref='pending')
        expected = await memory.search("Helios")
        await memory.save(snapshot)
    async with Memory(config(tmp_path / "two"), services=services()) as restored:
        await restored.load(snapshot)
        assert (await restored.search("Helios")).to_dict() == expected.to_dict()
        await submit_episode(restored, MemoryEpisode("pending"))
        assert len((await restored.search("Helios")).items) == 2
        with pytest.raises(CapabilityMismatchError):
            await restored.search("Helios", scope={"user_id": "bob"})
        await restored.reset()
        assert not (await restored.search("Helios")).items


@pytest.mark.asyncio
async def test_malformed_model_output_does_not_fabricate_links_or_keywords(tmp_path):
    async with Memory(config(tmp_path), services=services(malformed=True)) as memory:
        await submit_episode(memory, MemoryEpisode(events=[MemoryEvent("observation", "Helios 8443"),
                                                MemoryEvent("observation", "Helios 9443")]))
        result = await memory.search("Helios")
        assert len(result.items) == 2
        assert all(item["metadata"]["links"] == [] for item in result.items)
        assert all(item["metadata"]["keywords"] == [] for item in result.items)
        assert all(item["metadata"]["context"] == "General" for item in result.items)


def test_typed_config_and_invalid_settings_fail_before_allocating(tmp_path):
    from mematlas import MemAtlasConfig
    from mematlas.methods.a_mem import AMemConfig, SUPPORTED_OPERATIONS
    settings = AMemConfig(top_k=2, write_policy="episode")
    assert MemAtlasConfig(method=settings).method.params["top_k"] == 2
    assert "evolve" not in SUPPORTED_OPERATIONS and "command" not in SUPPORTED_OPERATIONS
    with pytest.raises(ConfigurationError):
        Memory(MemAtlasConfig(method=AMemConfig(top_k=0), runtime={"artifact_dir": str(tmp_path)}))
    assert not list(tmp_path.iterdir())


@pytest.mark.asyncio
async def test_invalid_embedding_never_silently_uses_hash_vectors(tmp_path):
    from mematlas.core.errors import MethodOperationError
    class EmptyVectors:
        async def embed(self, texts):
            return [[] for _ in texts]
    supplied = services()
    supplied.embedder = EmptyVectors()
    async with Memory(config(tmp_path), services=supplied) as memory:
        with pytest.raises(MethodOperationError, match="Embedding"):
            await add_event(memory, MemoryEvent("observation", "Helios"))


@pytest.mark.asyncio
async def test_default_qa_receives_note_time_and_context_with_provenance(tmp_path):
    from mematlas.core.contributions import contribution_text, applied_sources
    timestamp = datetime(2026, 11, 1, 13, 0, tzinfo=timezone.utc)
    async with Memory(config(tmp_path), services=services()) as memory:
        await add_event(memory, MemoryEvent("message", "Helios port is 8443", role="user",
                                        timestamp=timestamp, event_id="dated-event"))
        contribution = await memory.search("Helios")
        text = contribution_text(contribution)
        assert "202611011300" in text
        assert "Service configuration" in text
        assert "Helios port is 8443" in text
        assert [item["source_event_ids"] for item in applied_sources([contribution])] == [["dated-event"]]


@pytest.mark.asyncio
async def test_provider_json_object_mode_can_run_without_schema_api_support(tmp_path):
    from mematlas.test.a_mem.fixtures import FixedDecisions
    class ObjectOnly(FixedDecisions):
        async def complete(self, messages, **options):
            if options.get("response_format") != {"type": "json_object"}:
                raise ValueError("This provider only supports JSON Object responses")
            return await super().complete(messages, **options)
    supplied = services()
    supplied.llm = ObjectOnly(evolve=True)
    settings = config(tmp_path)
    settings["providers"] = {"llm": {"options": {"response_format": {"type": "json_object"}}}}
    async with Memory(settings, services=supplied) as memory:
        await submit_episode(memory, MemoryEpisode(events=[MemoryEvent("observation", "Helios 8443"),
                                                MemoryEvent("observation", "Helios 9443")]))
        result = await memory.search("Helios")
        assert len(result.items) == 2
        assert any(item["metadata"]["links"] for item in result.items)


@pytest.mark.asyncio
async def test_evolved_context_retains_the_updating_events_as_sources(tmp_path):
    async with Memory(config(tmp_path), services=services(evolve=True)) as memory:
        await submit_episode(memory, MemoryEpisode(events=[
            MemoryEvent("observation", "Helios port is 8443", event_id="old-port"),
            MemoryEvent("observation", "Helios port changed to 9443", event_id="new-port"),
        ]))
        result = await memory.search("Helios")
        old_note = next(item for item in result.items if item["content"] == "Helios port is 8443")
        assert old_note["source_event_ids"] == ["old-port", "new-port"]


@pytest.mark.asyncio
async def test_cross_neighbor_and_inherited_evolution_sources_survive_restore(tmp_path):
    import re
    from mematlas.test.a_mem.fixtures import FixedDecisions
    class CrossNeighbor(FixedDecisions):
        def __init__(self):
            super().__init__()
            self.evolution_calls = 0
        async def complete(self, messages, **options):
            if "should_evolve" not in messages[-1]["content"]:
                return await super().complete(messages, **options)
            self.evolution_calls += 1
            if self.evolution_calls == 1:
                return await super().complete(messages, **options)
            ids = re.findall(r"memory id:([^\s]+)", messages[-1]["content"])
            return json.dumps({"should_evolve": True, "actions": ["strengthen", "update_neighbor"],
                "suggested_connections": [], "tags_to_update": ["Helios 8443"],
                "new_context_neighborhood": ["Helios port is 8443"] * len(ids),
                "new_tags_neighborhood": [["copied-fact"] for _ in ids]})
    supplied = services()
    supplied.llm = CrossNeighbor()
    snapshot = tmp_path / "state.json"
    async with Memory(config(tmp_path / "one"), services=supplied) as memory:
        for identity, text in (("a", "Helios port is 8443"), ("b", "Task status is pending"),
                               ("c", "Combine the available facts"), ("d", "Summarize current status")):
            await add_event(memory, MemoryEvent("observation", text, event_id=identity))
        result = await memory.search("status")
        b = next(item for item in result.items if item["content"] == "Task status is pending")
        d = next(item for item in result.items if item["content"] == "Summarize current status")
        assert set(b["source_event_ids"]) == {"a", "b", "c", "d"}
        assert set(d["source_event_ids"]) == {"a", "b", "c", "d"}
        # No graph connections were selected; dependencies must not be inferred from links.
        assert not any(item["metadata"]["links"] for item in result.items)
        await memory.save(snapshot)
    async with Memory(config(tmp_path / "two"), services=services()) as restored:
        await restored.load(snapshot)
        assert (await restored.search("status")).to_dict() == result.to_dict()
