"""MemoryOS public workflows with real JSON/FAISS stores and controlled models."""
from mematlas.evaluation.validation import submit_episode
from mematlas.test.evaluation.support import add_event
import asyncio
from copy import deepcopy
from dataclasses import replace
import json

import pytest

from mematlas import MemoryUnit, MemAtlasConfig, Memory, MemoryEpisode, MemoryEvent, RuntimeConfig
from mematlas.core.contributions import applied_sources, contribution_text
from mematlas.core.errors import CapabilityMismatchError, MemoryStateError, MethodOperationError
from mematlas.core.plugin import MethodServices
from mematlas.core.scope import scope_key
from mematlas.methods.memoryos import MemoryOSConfig
from mematlas.methods.memoryos.core import MemoryOSCore
from mematlas.methods.memoryos.utils import format_contribution
from mematlas.providers import CallableLLM


class Models:
    def __init__(self):
        self.calls = []
        self.fail_profile = False
        self.fail_knowledge = False
        self.profile = "Food Interest (High): The user enjoys black coffee without sugar."

    async def complete(self, messages, **options):
        prompt = messages[-1]["content"]
        self.calls.append(prompt)
        if prompt.startswith("Update"):
            return '{"meta_info":"User discussed black coffee without sugar"}'
        if prompt.startswith("Summarize"):
            return '{"summaries":[{"content":"coffee preferences","keywords":["coffee"]}]}'
        if prompt.startswith("Check"):
            return '{"continuous":false}'
        if prompt.startswith("Please analyze"):
            if self.fail_profile:
                raise RuntimeError("controlled profile failure")
            return self.profile
        if prompt.startswith("Extract"):
            if self.fail_knowledge:
                raise RuntimeError("controlled knowledge failure")
            return '{"user_knowledge":["Coffee without sugar"],"assistant_knowledge":["Explain coffee recipes"]}'
        raise AssertionError(prompt)

    @property
    def profile_calls(self):
        return sum(prompt.startswith("Please analyze") for prompt in self.calls)


class Vectors:
    async def embed(self, texts):
        return [[1.0, 0.25, 0.5] for _ in texts]


async def test_reference_chain_summaries_and_topic_summaries_have_separate_inputs(tmp_path):
    calls = []
    meta_summaries = iter(("first page overview", "first and second page overview", "new chain overview"))
    continuations = iter((True, False))

    async def complete(messages, **options):
        calls.append(deepcopy(messages))
        prompt = messages[-1]["content"]
        if prompt.startswith("Check"):
            # A continuity response cannot replace the separate chain summary.
            return json.dumps({"continuous": next(continuations), "meta_info": "unused continuity text"})
        if prompt.startswith("Update"):
            return json.dumps({"meta_info": next(meta_summaries)})
        if prompt.startswith("Summarize"):
            return json.dumps({"summaries": [
                {"content": "topic A", "keywords": ["alpha"]},
                {"content": "topic B", "keywords": ["beta"]},
                {"content": "excess topic", "keywords": ["discarded"]},
            ]})
        raise AssertionError(prompt)

    core = MemoryOSCore(user_id="reference", short_term_capacity=1,
        mid_term_heat_threshold=100, mid_term_similarity_threshold=3,
        services=MethodServices(artifact_dir=tmp_path, llm=CallableLLM(complete), embedder=Vectors()))
    try:
        for text in ("first", "second", "third", "fourth"):
            await core.add_memory(text, "noted", timestamp="2026-10-31T12:00:00+00:00")
        assert len(core.mid_term_sessions) == 6  # Two topics per page, even if the model returns three.
        pages = {page["user_input"]: page for page in core.mid_term_pages}
        assert pages["first"]["meta_info"] == pages["second"]["meta_info"] == "first and second page overview"
        assert pages["first"]["next_page"] == pages["second"]["page_id"]
        assert pages["second"]["pre_page"] == pages["first"]["page_id"]
        assert pages["third"]["meta_info"] == "new chain overview"
        assert pages["third"]["pre_page"] is None
        assert all(page["meta_info"] != "unused continuity text" for page in core.mid_term_pages)
        updates = [messages[-1]["content"] for messages in calls if messages[-1]["content"].startswith("Update")]
        assert "Previous Meta-summary: None" in updates[0]
        assert "Previous Meta-summary: first page overview" in updates[1]
        assert "Previous Meta-summary: None" in updates[2]
        assert all("topic A" not in prompt and "topic B" not in prompt for prompt in updates)
        # All copies of the same page have the same chain overview and links.
        assert all(page["next_page"] == pages["second"]["page_id"]
                   for page in core.mid_term_pages if page["user_input"] == "first")
    finally:
        await core.close()


async def test_reference_generation_uses_dialogue_evidence_without_vectors_or_storage_metadata(tmp_path):
    models = Models()
    calls = []
    profile_started, knowledge_started = asyncio.Event(), asyncio.Event()

    async def complete(messages, **options):
        calls.append(deepcopy(messages))
        prompt = messages[-1]["content"]
        if prompt.startswith("Please analyze"):
            assert options == {"temperature": 0.7, "max_tokens": 2000}
            profile_started.set()
            await asyncio.wait_for(knowledge_started.wait(), timeout=2)
        elif prompt.startswith("Extract"):
            knowledge_started.set()
            await asyncio.wait_for(profile_started.wait(), timeout=2)
        return await models.complete(messages, **options)

    core = MemoryOSCore(user_id="reference", short_term_capacity=1, mid_term_heat_threshold=2.5,
        services=MethodServices(artifact_dir=tmp_path, llm=CallableLLM(complete), embedder=Vectors()))
    try:
        core.user_profile = "Existing supported coffee preference"
        for text in ("coffee first", "coffee second", "coffee third"):
            await core.add_memory(text, "I provided a coffee recipe", timestamp="2026-10-31T12:00:00+00:00",
                meta_data={"internal_marker": "DO_NOT_SEND_STORAGE_METADATA", "embedding": [12345.6789]})
        assert models.profile_calls == 1
        prompt_text = "\n".join(message["content"] for messages in calls for message in messages)
        for private in ("embedding", "page_id", "analyzed", "DO_NOT_SEND_STORAGE_METADATA", "12345.6789"):
            assert private not in prompt_text
        profile_messages = next(messages for messages in calls if messages[-1]["content"].startswith("Please analyze"))
        assert "Existing supported coffee preference" in profile_messages[-1]["content"]
        assert "Extraversion" in profile_messages[-1]["content"] and "Practicality" in profile_messages[-1]["content"]
        assert "If the dimension is not reflected, do not extract or list it." in profile_messages[0]["content"]
        assert "Output only the user profile section." in profile_messages[0]["content"]
        assert profile_messages[-1]["content"].count("(This reflects user's expectation of AI)") == 6
        assert "JSON" not in "\n".join(message["content"] for message in profile_messages)
        assert "User: coffee first (Timestamp: 2026-10-31T12:00:00+00:00)\nAssistant: I provided a coffee recipe (Timestamp: 2026-10-31T12:00:00+00:00)\nUser: coffee second" in profile_messages[-1]["content"]
        knowledge_messages = next(messages for messages in calls if messages[-1]["content"].startswith("Extract"))
        assert "actually did" in knowledge_messages[0]["content"]
        assert "Do not invent abilities" in knowledge_messages[0]["content"]
        assert "coffee first" in knowledge_messages[-1]["content"]
        assert "(Timestamp: 2026-10-31T12:00:00+00:00)" in knowledge_messages[-1]["content"]
    finally:
        await core.close()


async def test_existing_checkpoint_pages_load_without_regenerating_summaries(tmp_path):
    models = Models()
    core = MemoryOSCore(user_id="reference", services=MethodServices(
        artifact_dir=tmp_path, llm=CallableLLM(models.complete), embedder=Vectors()))
    legacy = {"user_id": "reference", "assistant_id": "default_assistant_profile", "short_term": [],
        "mid_term_sessions": {"session-existing": {
            "id": "session-existing", "summary": "existing coffee topic",
            "summary_embedding": [1.0, 0.25, 0.5], "summary_keywords": ["coffee"],
            "details": [{"page_id": "page-existing", "memory_id": "memory-existing",
                "user_input": "existing coffee fact", "agent_response": "noted",
                "timestamp": "2026-10-30T12:00:00+00:00", "meta_data": {},
                "embedding": [1.0, 0.25, 0.5], "page_keywords": ["coffee"],
                "meta_info": "previously saved chain overview", "pre_page": None,
                "next_page": None, "preloaded": False, "analyzed": True}],
            "N_visit": 2, "L_interaction": 0, "H_segment": 3.0,
            "last_visit_time": "2026-10-30T12:00:00+00:00", "access_count_lfu": 2}},
        "user_profile": "Existing profile", "user_knowledge": [], "assistant_knowledge": [],
        "last_evicted_page_id": "page-existing"}
    try:
        core.import_state(deepcopy(legacy))
        result = await core.retrieve_context(user_query="coffee", user_id="reference", read_only=True)
        assert result["retrieved_pages"][0]["meta_info"] == "previously saved chain overview"
        assert result["retrieved_pages"][0]["user_input"] == "existing coffee fact"
        assert core.export_state() == legacy
        assert models.calls == []
    finally:
        await core.close()


def services(models):
    return MethodServices(llm=CallableLLM(models.complete), embedder=Vectors())


def config_for(directory, **params):
    return {"method": {"method_id": "memoryos", "params": params},
            "scope": {"user_id": "alice"}, "runtime": {"artifact_dir": str(directory)}}


def unit(identity, text="coffee", reply="noted"):
    return MemoryEpisode(identity, events=[MemoryEvent("message", text, role="user"),
                                           MemoryEvent("message", reply, role="assistant")])


def core_for(memory, scope=None):
    return memory._method.instances[scope_key(scope or memory.config.scope)]


async def test_add_writes_real_messages_once_without_feedback_or_flush(tmp_path):
    models = Models()
    async with Memory(config_for(tmp_path), services=services(models)) as memory:
        user = MemoryEvent("message", "black coffee", role="user")
        answer = MemoryEvent("message", "preference noted", role="assistant")
        unit = MemoryUnit(events=[user, answer])
        receipt = await memory.add(unit)
        await memory.episode_end(receipt.episode_ref)
        assert await memory.add(unit) == receipt
        assert len(core_for(memory).short_term_memory.items) == 1
        used = await memory.search("coffee")
        assert used.items[0]["source_event_ids"] == [user.event_id, answer.event_id]
        assert "black coffee" in contribution_text(used)
        assert applied_sources([used]) == tuple(used.items)
        assert models.calls == []


async def test_complete_history_and_single_message_are_valid_learning_units(tmp_path):
    async with Memory(config_for(tmp_path), services=services(Models())) as memory:
        await submit_episode(memory, unit("history"))
        await submit_episode(memory, MemoryEpisode("single", events=[MemoryEvent("message", "plain fact", role="user")]))
        pairs = core_for(memory).short_term_memory.items
        assert [(pair["user_input"], pair["agent_response"]) for pair in pairs] == [
            ("coffee", "noted"), ("plain fact", "")]
        assert all(pair["meta_data"]["mematlas_event_ids"] for pair in pairs)


async def test_interleaved_tasks_and_single_messages_keep_unit_pairing(tmp_path):
    async with Memory(config_for(tmp_path), services=services(Models())) as memory:
        await memory.add(MemoryUnit(episode_ref="A", events=[
            MemoryEvent("message", "A", role="user"), MemoryEvent("message", "A reply", role="assistant")]))
        await memory.add(MemoryUnit(episode_ref="B", events=[MemoryEvent("message", "B", role="user")]))
        await memory.episode_end("A")
        await memory.episode_end("B")
        await submit_episode(memory, MemoryEpisode("attached", events=[MemoryEvent("message", "unattached", role="user")]))
        assert [(row["user_input"], row["agent_response"]) for row in core_for(memory).short_term_memory.items] == [
            ("A", "A reply"), ("B", ""), ("unattached", "")]
        assert not memory._method._buffers


async def test_full_scope_isolates_storage_retrieval_and_reset(tmp_path):
    owners = [{"user_id": user, "agent_id": agent, "session_id": session}
              for user, agent, session in [("alice", "a", "1"), ("alice", "b", "1"),
                                          ("alice", "a", "2"), ("bob", "a", "1")]]
    async with Memory(config_for(tmp_path), services=services(Models())) as memory:
        for index, owner in enumerate(owners):
            await submit_episode(memory, unit(f"unit-{index}", text=f"fact-{index}"), scope=owner)
        assert len({core_for(memory, owner)._storage_dir for owner in owners}) == 4
        for index, owner in enumerate(owners):
            used = await memory.search("fact", scope=owner)
            assert len(used.items) == 1 and f"fact-{index}" in used.items[0]["content"]
        await memory.reset(scope=owners[0])
        assert not (await memory.search("fact", scope=owners[0])).items
        assert (await memory.search("fact", scope=owners[1])).items


async def test_accepted_unit_and_open_task_survive_freeze_flush_save_and_load(tmp_path):
    models = Models()
    path = tmp_path / "pending.json"
    config = config_for(tmp_path / "runtime")
    async with Memory(config, services=services(models)) as memory:
        await add_event(memory, MemoryEvent("message", "pending input", role="user", episode_id="pending"), episode_ref='pending')
        state = await memory._method.save()
        await memory.flush()
        await memory.save(path)
        async with memory.read_only():
            assert (await memory.search("pending")).items
        assert await memory._method.save() == state
        assert len(memory._method.instances) == 1
    assert await memory._method.save() == state
    assert models.calls == []
    async with Memory(config, services=services(models)) as restored:
        await restored.load(path)
        assert await restored._method.save() == state
        await submit_episode(restored, MemoryEpisode("pending"))
        assert core_for(restored).short_term_memory.items[0]["user_input"] == "pending input"
        assert not restored._method._buffers


@pytest.mark.parametrize("online", [False, True])
async def test_native_automatic_evolution_does_not_repeat_processed_pages(tmp_path, online):
    models = Models()
    async with Memory(config_for(tmp_path, short_term_capacity=1, mid_term_heat_threshold=2.5),
                      services=services(models)) as memory:
        for index in range(3):
            episode = unit(f"unit-{index}", text=f"coffee {index}")
            if online:
                receipt = await memory.add(MemoryUnit.from_episode(episode))
                await memory.episode_end(receipt.episode_ref)
            else:
                await submit_episode(memory, episode)
        core = core_for(memory)
        assert len(core.short_term_memory.items) == 1 and len(core.mid_term_pages) == 2
        assert models.profile_calls == 1
        assert core.user_profile == models.profile
        assert core.user_knowledge[0]["knowledge"] == "Coffee without sugar"
        assert core.assistant_knowledge[0]["knowledge"] == "Explain coffee recipes"
        assert all(page["analyzed"] for page in core.mid_term_pages)
        await submit_episode(memory, unit("next", text="coffee next"))
        assert models.profile_calls == 1
        await submit_episode(memory, unit("following", text="coffee following"))
        assert models.profile_calls == 2
        assert all(page["analyzed"] for page in core.mid_term_pages)
        assert len(core.user_knowledge) == len(core.assistant_knowledge) == 1


async def test_public_evolve_is_rejected_without_changing_memory(tmp_path):
    models = Models()
    async with Memory(config_for(tmp_path), services=services(models)) as memory:
        await submit_episode(memory, unit("first"))
        state = await memory._method.save()
        assert not hasattr(memory, "evolve")
        assert await memory._method.save() == state
        assert models.calls == []


async def test_automatic_evolution_preserves_heat_gate_and_uses_accumulated_retrieval_visits(tmp_path):
    models = Models()
    async with Memory(config_for(tmp_path, short_term_capacity=1, mid_term_heat_threshold=3.5),
                      services=services(models)) as memory:
        assert not memory._method.instances
        await submit_episode(memory, unit("first"))
        await submit_episode(memory, unit("second"))
        assert models.profile_calls == 0
        await memory.search("coffee")
        await memory.search("coffee")
        assert models.profile_calls == 0
        state = await memory._method.save()
        await memory.flush()
        await memory.save(tmp_path / "hot-state.json")
        async with memory.read_only():
            await memory.search("coffee")
        assert await memory._method.save() == state and models.profile_calls == 0
        await submit_episode(memory, unit("third"))
        assert models.profile_calls == 1
        assert all(page["analyzed"] for page in core_for(memory).mid_term_pages)


async def test_frozen_retrieval_preserves_layers_visits_buffers_and_history(tmp_path):
    models = Models()
    async with Memory(config_for(tmp_path, short_term_capacity=1, mid_term_heat_threshold=2.5),
                      services=services(models)) as memory:
        for index in range(3):
            await submit_episode(memory, unit(f"unit-{index}"))
        await add_event(memory, MemoryEvent("message", "unfinished", role="user", episode_id="pending"))
        state = await memory._method.save()
        history = memory._history.snapshot()
        calls = list(models.calls)
        async with memory.read_only():
            first = await memory.search("coffee")
            second = await memory.search("coffee")
            assert first == second and len(first.items) >= 5
            for action in (lambda: memory.episode_end(), lambda: submit_episode(memory, unit("blocked")), lambda: memory.flush()):
                with pytest.raises(MemoryStateError):
                    await action()
            assert not (await memory.search("coffee", scope={"user_id": "unseen"})).items
        assert await memory._method.save() == state
        assert memory._history.snapshot() == history and models.calls == calls


async def test_save_restore_replaces_all_layers_scopes_and_duplicate_tracking(tmp_path):
    path = tmp_path / "snapshot.json"
    models = Models()
    config = config_for(tmp_path / "runtime", short_term_capacity=1, mid_term_heat_threshold=2.5)
    async with Memory(config, services=services(models)) as memory:
        for index in range(3):
            await submit_episode(memory, unit(f"unit-{index}"))
        await memory.save(path)
        snapshot = await memory._method.save()
        await submit_episode(memory, unit("extra"), scope={"user_id": "other"})
        await submit_episode(memory, unit("later", text="discard this"))
        await memory.load(path)
        assert await memory._method.save() == snapshot
        assert not (await memory.search("extra", scope={"user_id": "other"})).items
    async with Memory(config, services=services(models)) as restored:
        await restored.load(path)
        assert await restored._method.save() == snapshot
        async with restored.read_only():
            assert "black coffee" in contribution_text(await restored.search("coffee"))


async def test_loading_empty_snapshot_removes_previous_stores(tmp_path):
    async with Memory(config_for(tmp_path), services=services(Models())) as memory:
        empty = await memory._method.save()
        await submit_episode(memory, unit("discarded"))
        await memory._method.load(empty)
        assert await memory._method.save() == empty
        assert not (await memory.search("discarded")).items


@pytest.mark.parametrize("failure", ["profile", "knowledge"])
async def test_failed_long_term_extraction_rolls_back_complete_learning_unit(tmp_path, failure):
    models = Models()
    async with Memory(config_for(tmp_path, short_term_capacity=1, mid_term_heat_threshold=2.5),
                      services=services(models)) as memory:
        await submit_episode(memory, unit("first"))
        await submit_episode(memory, unit("second"))
        state = await memory._method.save()
        setattr(models, f"fail_{failure}", True)
        failed = unit("third")
        with pytest.raises(MethodOperationError, match=f"controlled {failure} failure"):
            await submit_episode(memory, failed)
        assert await memory._method.save() == state
        assert memory.stats()["state"] == "started"
        setattr(models, f"fail_{failure}", False)
        await submit_episode(memory, failed)
        assert core_for(memory).user_profile == models.profile


@pytest.mark.parametrize("response", ["", "   ", "None", "  nOnE  ", "x" * 29, "x" * 30,
                                     "  Conciseness (High): The user prefers brief answers.  "])
async def test_native_profile_save_gate_preserves_or_replaces_complete_profile(tmp_path, response):
    models = Models()
    models.profile = response
    async with Memory(config_for(tmp_path, short_term_capacity=1, mid_term_heat_threshold=2.5),
                      services=services(models)) as memory:
        await submit_episode(memory, unit("first"))
        await submit_episode(memory, unit("second"))
        core = core_for(memory)
        core.user_profile = "Existing profile: User enjoys black coffee."
        previous = core.user_profile
        await submit_episode(memory, unit("third"))
        expected = response.strip() if len(response.strip()) >= 30 else previous
        assert core.user_profile == expected
        persisted = json.loads(core._file("long_term_user.json").read_text(encoding="utf-8"))
        assert persisted["profile"] == expected
        assert core.user_knowledge[0]["knowledge"] == "Coffee without sugar"
        assert core.assistant_knowledge[0]["knowledge"] == "Explain coffee recipes"
        assert all(page["analyzed"] for page in core.mid_term_pages)


@pytest.mark.parametrize("existing_profile", ["", "None", "nOnE"])
async def test_native_profile_placeholder_when_no_profile_exists(tmp_path, existing_profile):
    models = Models()
    async with Memory(config_for(tmp_path, short_term_capacity=1, mid_term_heat_threshold=2.5),
                      services=services(models)) as memory:
        await submit_episode(memory, unit("first"))
        core_for(memory).user_profile = existing_profile
        await submit_episode(memory, unit("second"))
        await submit_episode(memory, unit("third"))
        profile_prompt = next(prompt for prompt in models.calls if prompt.startswith("Please analyze"))
        assert "**Existing User Profile:**\nNo existing profile data." in profile_prompt


@pytest.mark.parametrize("top_k", [1, 2, 4, 10])
def test_top_k_limits_all_layers_and_keeps_context_sources_consistent(top_k):
    original = {"retrieved_pages": [{"page_id": "p", "user_input": "old", "agent_response": "reply",
                                      "meta_data": {"mematlas_event_ids": ["e1", "e2"]}, "score": 0.8}],
                "retrieved_user_knowledge": [{"id": "uk", "knowledge": "user fact", "score": 0.7}],
                "retrieved_assistant_knowledge": [{"id": "ak", "knowledge": "assistant fact", "score": 0.6}]}
    before = deepcopy(original)
    result = format_contribution([{"user_input": "new", "agent_response": "new reply"}],
                                 original, "profile text", "alice", top_k)
    assert len(result.items) == min(top_k, 5)
    assert sum(len(row["source_item_indices"]) for row in result.context) == len(result.items)
    for row in result.context:
        assert all(result.items[index]["content"] in row["content"] for index in row["source_item_indices"])
    for text in ("old", "user fact", "assistant fact", "profile text"):
        assert (text in contribution_text(result)) == any(text in row["content"] for row in result.items)
    assert original == before


async def test_explicit_top_k_bounds_actual_retrieval_and_context(tmp_path):
    async with Memory(config_for(tmp_path, short_term_capacity=1, mid_term_heat_threshold=2.5),
                      services=services(Models())) as memory:
        for index in range(3):
            await submit_episode(memory, unit(f"unit-{index}", text=f"coffee {index}"))
        async with memory.read_only():
            assert len((await memory.search("coffee")).items) == 6
            limited = await memory.search("coffee", top_k=1)
            assert len(limited.items) == 1 and len(limited.context) == 1
            assert "coffee 2" in contribution_text(limited) and "coffee 0" not in contribution_text(limited)


async def test_invalid_message_role_fails_before_writing_any_episode_messages(tmp_path):
    async with Memory(config_for(tmp_path), services=services(Models())) as memory:
        with pytest.raises(MethodOperationError, match="role"):
            await submit_episode(memory, MemoryEpisode("bad", events=[MemoryEvent("message", "valid", role="user"),
                                                            MemoryEvent("message", "tool", role="tool")]))
        assert not memory._method.instances and not memory._method._buffers
        with pytest.raises(CapabilityMismatchError, match="Unsupported event"):
            await add_event(memory, MemoryEvent("action", "unsupported"), episode_ref='bad')


def test_missing_services_are_reported_before_starting_standard_method():
    report = Memory.check(MemAtlasConfig(method=MemoryOSConfig()), services=MethodServices())
    assert not report["ready"]
    assert set(report["health"]["missing_services"]) == {"llm", "embedding"}


def test_json_layers_can_be_saved_near_windows_path_limit(tmp_path):
    import os
    from mematlas.core.files import atomic_write_json, read_json
    if os.name != "nt":
        pytest.skip("Windows path-length regression")
    # The final filename fits the limit; atomic writing must fit it as well.
    filename = "long_term_assistant.json"
    directory = tmp_path / ("x" * (256 - len(filename) - len(str(tmp_path)) - 2))
    target = directory / filename
    assert len(str(target)) == 256
    atomic_write_json(target, {"knowledge": ["durable fact"]})
    assert read_json(target, {}) == {"knowledge": ["durable fact"]}


async def test_public_example_runs_same_agent_flow_without_extra_operations(tmp_path, monkeypatch):
    from mematlas.methods.memoryos import example
    inputs, memories = [], []
    models = Models()

    def create(config):
        config = replace(config, runtime=RuntimeConfig(artifact_dir=str(tmp_path)))
        memory = Memory(config, services=services(models))
        memories.append(memory)
        return memory

    async def agent(messages):
        inputs.append(messages)
        return "Helios port is 8443"

    monkeypatch.setattr(example, "Memory", create)
    assert await example.run(["Remember Helios", "What is the port?"], agent, providers={}) == [
        "Helios port is 8443", "Helios port is 8443"]
    assert "8443" in inputs[1][0]["content"]
    memory = memories[0]
    assert memory.config.method.params == {"short_term_capacity": 10, "retrieval_queue_capacity": 7,
                                            "mid_term_heat_threshold": 5}
    assert memory.stats()["episodes"]["learned_exposures"] == 2
    assert len(core_for(memory).short_term_memory.items) == 2
    for key in memory._history.episodes:
        episode = memory._history.episode(key)
        assert [event.role for event in episode.events] == ["user", "assistant"]
        assert all(event.episode_id == episode.episode_id for event in episode.events)
    assert {trace.operation for trace in memory.traces.all()} >= {"add", "search", "episode_end"}
    assert "evolve" not in {trace.operation for trace in memory.traces.all()}


async def test_generic_evaluation_consumes_complete_dialogue_history(tmp_path):
    from mematlas import ExperimentConfig, ExperimentRunner
    from mematlas.core.evaluation import BenchmarkCase, Prediction

    class Benchmark:
        benchmark_id = "memoryos-dialogue"

        def cases(self, split="test"):
            return [BenchmarkCase("case", "coffee", scope={"user_id": "alice"},
                                  reference="coffee", episode=unit("history"))]

        def describe(self):
            return {"benchmark_id": self.benchmark_id}

        async def native_score_async(self, predictions):
            return {"accuracy": float(predictions[0].output == predictions[0].reference)}

    async def agent(case, memory):
        contribution = await memory.search(case.query, scope=case.scope)
        output = "coffee" if "coffee" in contribution_text(contribution) else "missing"
        return MemoryEpisode(output=output, applied_memory=[contribution])

    config = MemAtlasConfig.from_dict(config_for(tmp_path))
    results = await ExperimentRunner().evaluate(ExperimentConfig((config,), Benchmark(), output_dir=str(tmp_path / "report")),
        agent=agent, services=services(Models()))
    assert not results[0].failures and results[0].metrics["native.accuracy"] == 1


async def test_continuous_evaluation_freezes_each_checkpoint_without_extra_evolution(tmp_path):
    from mematlas import ExperimentConfig, ExperimentRunner
    from mematlas.benchmarks.datasets import InMemoryBenchmark
    from mematlas.core.evaluation import BenchmarkCase
    models = Models()
    scope = {"user_id": "alice"}
    benchmark = InMemoryBenchmark("memoryos-continuous", [
        *[BenchmarkCase(identity, "coffee", scope=scope, episode=unit(identity), metadata={"split": "train"})
          for identity in ("train-1", "train-2")],
        BenchmarkCase("test", "coffee", scope=scope),
    ])
    supplied = services(models)
    supplied.objects["answer_llm"] = CallableLLM(lambda messages, **kwargs: "answer")
    config = ExperimentConfig.from_dict({"methods": [config_for(tmp_path / "memory")], "benchmark": benchmark,
        "protocol": "experience_learning_curve", "checkpoints": (0, 1, 2),
        "output_dir": str(tmp_path / "experiment")})
    result = (await ExperimentRunner().evaluate(config, services=supplied))[0]
    assert not result.failures and result.run_metadata["progress"]["learned_exposures"] == 2
    assert [len(row["predictions"][0]["applied_memory"][0]["items"]) for row in result.learning_curve] == [0, 1, 2]
    assert models.profile_calls == 0
