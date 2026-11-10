"""Native differential/lifecycle checks: actual Qdrant, SQLite, spaCy and BM25."""
import asyncio
from copy import deepcopy
from contextvars import copy_context
import json
from pathlib import Path
import re

import pytest

from mematlas import Memory, MemoryEvent, MemoryEpisode, MemoryUnit, MethodConfig
from mematlas.core.errors import ConfigurationError
from mematlas.methods.mem0.method import Mem0Plugin, native_identity
from mematlas.methods.mem0.native import NativeBackend
from mematlas.methods.mem0.provenance import IDENTITY, verify_source
from mematlas.test.mem0.native_support import config_for, require_native, services


@pytest.fixture(autouse=True)
def native_required():
    require_native()


def canonical(value):
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    text = re.sub(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", "UUID", text)
    return re.sub(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d[^\" ]*", "TIMESTAMP", text)


def collection_content(memory):
    result = {}
    for collection in memory.vector_store.client.get_collections().collections:
        rows, _ = memory.vector_store.client.scroll(collection.name, limit=10000, with_vectors=True)
        result[collection.name] = sorted([
            {"payload": row.payload, "vector": row.vector} for row in rows
        ], key=lambda row: row["payload"].get("data", ""))
    return result


async def test_adapter_matches_direct_official_native_batches_prompts_entities_and_ranking(tmp_path):
    wrapped_services, direct_services = services(), services()
    direct_services.artifact_dir = tmp_path / "direct-artifacts"
    direct = NativeBackend(direct_services, {"storage_path": str(tmp_path / "direct")}, 16,
                           {"model": direct_services.llm.model})
    async with Memory(config_for(tmp_path / "wrapped"), services=wrapped_services) as wrapped:
        await direct.setup()
        try:
            assert type(wrapped._method.memory).__module__ == "mem0.memory.main"
            assert verify_source() == []
            event = MemoryEvent("message", "Helios port is 8443 and Alice prefers tea", role="user", event_id="source")
            await wrapped.add(MemoryUnit(events=[event], episode_ref="unit"))
            direct.binding.context = copy_context()
            await asyncio.to_thread(direct.memory.add, [{"role": "user", "content": event.content}],
                metadata={"mematlas_event_ids": ["source"], "mematlas_event_kinds": ["message"],
                          "mematlas_scope": {"user_id": "alice"}},
                **native_identity({"user_id": "alice"}))
            actual = await wrapped.search("What is the Helios port?", top_k=3)
            expected = await asyncio.to_thread(direct.memory.search, "What is the Helios port?", top_k=3,
                filters=native_identity({"user_id": "alice"}), threshold=0.1, rerank=False)
            assert [row["content"] for row in actual.items] == [row["memory"] for row in expected["results"]]
            assert [row["score"] for row in actual.items] == pytest.approx([row["score"] for row in expected["results"]])
            assert canonical(collection_content(wrapped._method.memory)) == canonical(collection_content(direct.memory))
            assert wrapped_services.embedder.calls == direct_services.embedder.calls
            assert any(len(batch) > 1 for batch in wrapped_services.embedder.calls)
            assert canonical(wrapped_services.llm.calls) == canonical(direct_services.llm.calls)
            tracked = wrapped.services.usage.all()
            assert len(tracked) == len(wrapped_services.embedder.calls) + len(wrapped_services.llm.calls)
            assert all(row.operation in {"add", "search"} and row.operation_id and row.phase == "memory" for row in tracked)
            assert all(not row.usage_known for row in tracked)  # controlled providers report no token usage
            # Native recent-message context must also reach the second prompt.
            next_event = MemoryEvent("message", "Next turn", role="user")
            await wrapped.add(MemoryUnit(events=[next_event], episode_ref="second"))
            await asyncio.to_thread(direct.memory.add, [{"role": "user", "content": "Next turn"}],
                **native_identity({"user_id": "alice"}))
            assert canonical(wrapped_services.llm.calls[-1]) == canonical(direct_services.llm.calls[-1])
        finally:
            await direct.close()


async def test_native_file_checkpoint_is_deterministic_complete_and_restores_without_models(tmp_path):
    supplied = services()
    async with Memory(config_for(tmp_path), services=supplied) as memory:
        await memory.add(MemoryUnit(episode_ref="A", events=[MemoryEvent("message", "Helios port 8443", role="user")]))
        result = await memory.search("Helios")
        state = await memory._method.save()
        assert state == await memory._method.save()
        assert len(json.dumps(state)) < 3000
        manifest = state["native"]
        path = Path(manifest["snapshot_file"])
        assert path.is_file() and path.parent != memory._method.backend.root
        before_sql = list(memory._method.memory.db.connection.execute("SELECT * FROM messages"))
        before_history = list(memory._method.memory.db.connection.execute("SELECT * FROM history"))
        before_collections = collection_content(memory._method.memory)
        await memory.update(result.items[0]["memory_ref"], "replacement")
        calls = len(supplied.llm.calls), len(supplied.embedder.calls)
        await memory._method.load(state)
        assert calls == (len(supplied.llm.calls), len(supplied.embedder.calls))
        assert list(memory._method.memory.db.connection.execute("SELECT * FROM messages")) == before_sql
        assert list(memory._method.memory.db.connection.execute("SELECT * FROM history")) == before_history
        assert collection_content(memory._method.memory) == before_collections
        assert state == await memory._method.save()
        async with memory.read_only():
            assert await memory.search("Helios") == await memory.search("Helios")
        assert state == await memory._method.save()
        corrupted = deepcopy(state)
        corrupted["native"]["file_sha256"] = "bad"
        with pytest.raises(ValueError, match="checksum"):
            await memory._method.load(corrupted)
        assert state == await memory._method.save()


async def test_complete_owner_scopes_are_isolated_in_search_detail_and_reset(tmp_path):
    owners = [{"user_id": "alice"}, {"user_id": "alice", "agent_id": "doctor"},
              {"user_id": "alice", "agent_id": "doctor", "session_id": "s1"},
              {"user_id": "alice", "agent_id": "doctor", "session_id": "s1", "run_id": "r1"}]
    supplied = services()
    async with Memory(config_for(tmp_path), services=supplied) as memory:
        for number, owner in enumerate(owners):
            supplied.llm.facts = [f"Helios port is {8443 + number}"]
            await memory.add(MemoryUnit(events=[MemoryEvent("message", f"port {number}", role="user")],
                                       scope=owner, episode_ref=f"unit-{number}"))
        for number, owner in enumerate(owners):
            result = await memory.search("Helios port", scope=owner)
            assert [row["content"] for row in result.items] == [f"Helios port is {8443 + number}"]
        row = (await memory.search("Helios port", scope=owners[1])).items[0]
        with pytest.raises(ValueError, match="different scope"):
            await memory._method.detail(row["memory_id"], scope=owners[0])
        await memory.reset(scope=owners[0])
        assert not (await memory.search("Helios", scope=owners[0])).items
        assert (await memory.search("Helios", scope=owners[1])).items
        native_scope = native_identity(owners[0])
        from mem0.memory.main import _build_session_scope
        assert memory._method.memory.db.get_last_messages(_build_session_scope(native_scope)) == []


async def test_hospital_boundary_before_delayed_add_and_idempotent_receipts(tmp_path):
    supplied = services()
    async with Memory(config_for(tmp_path, write_policy="episode"), services=supplied) as memory:
        await memory.episode_end("encounter-1", status="completed")
        assert not supplied.llm.calls
        unit = MemoryUnit.from_episode(MemoryEpisode("encounter-1", task="diagnose",
            events=[MemoryEvent("message", "Clinical observation", role="user"),
                    MemoryEvent("message", "Diagnosis", role="assistant")],
            output="Diagnosis", feedback="Reference diagnosis later available"))
        receipt = await memory.add(unit)
        assert await memory.add(unit) == receipt
        assert len(supplied.llm.calls) == 1
        contribution = await memory.search("Helios")
        assert contribution.items[0]["source_event_ids"]
        assert len(supplied.llm.calls) == 1


async def test_complete_experience_fields_reach_native_once_and_delayed_feedback_only_adds_new_input(tmp_path, monkeypatch):
    from mematlas import MemoryContribution
    supplied = services()
    checkpoint = tmp_path / "fields.json"
    old_events = [MemoryEvent("message", "Observed chest pain", role="user", event_id="clinical-source")]
    async with Memory(config_for(tmp_path), services=supplied) as memory:
        batches = []
        native_add = memory._method.memory.add
        def capture(messages, **kwargs):
            batches.append(deepcopy(messages))
            return native_add(messages, **kwargs)
        monkeypatch.setattr(memory._method.memory, "add", capture)
        initial = MemoryUnit(type="experience", episode_ref="clinical", write_ref="clinical:initial",
            task="Assess ischemia", events=old_events, output={"diagnosis": "suspected ischemia"},
            feedback={"text": "Verify the ECG", "success": True},
            applied_memory=[MemoryContribution(items=[{"content": "APPLIED_MEMORY_MUST_NOT_BE_REINGESTED"}])],
            extensions={"reference_answer": "REFERENCE_MUST_NOT_ENTER_NATIVE"})
        receipt = await memory.add(initial)
        assert await memory.add(initial) == receipt
        assert [row["content"] for row in batches[0]] == ["Assess ischemia", "Observed chest pain",
            '{"diagnosis": "suspected ischemia"}', "Verify the ECG"]
        assert all(marker not in json.dumps(batches) for marker in ("APPLIED_MEMORY", "REFERENCE_MUST"))
        row = memory._method.memory.vector_store.list(filters=native_identity(memory.config.scope), top_k=10)[0][0]
        assert {source["field"] for source in row.payload["mematlas_field_sources"].values()} == {"task", "output", "feedback"}
        assert all(source["episode_ref"] == "clinical" for source in row.payload["mematlas_field_sources"].values())
        await memory.episode_end("clinical")
        await memory.save(checkpoint)
    async with Memory(config_for(tmp_path), services=supplied) as restored:
        await restored.load(checkpoint)
        batches = []
        native_add = restored._method.memory.add
        def capture_feedback(messages, **kwargs):
            batches.append(deepcopy(messages))
            return native_add(messages, **kwargs)
        monkeypatch.setattr(restored._method.memory, "add", capture_feedback)
        delayed = MemoryUnit(type="feedback", episode_ref="clinical", write_ref="clinical:delayed",
            task="Assess ischemia", events=old_events, output={"diagnosis": "suspected ischemia"},
            feedback={"text": "Delayed evaluation recommends troponin"})
        await restored.add(delayed)
        assert batches == [[{"role": "assistant", "content": "Delayed evaluation recommends troponin"}]]
        # A distinct delivery reference carrying the same field is still deduped.
        await restored.add(MemoryUnit(type="feedback", episode_ref="clinical", write_ref="clinical:retry",
            feedback={"text": "Delayed evaluation recommends troponin"}))
        assert len(batches) == 1


async def test_field_values_already_in_native_trajectory_are_not_duplicated(tmp_path, monkeypatch):
    supplied = services()
    async with Memory(config_for(tmp_path), services=supplied) as memory:
        batches = []
        native_add = memory._method.memory.add
        def capture(messages, **kwargs):
            batches.append(deepcopy(messages))
            return native_add(messages, **kwargs)
        monkeypatch.setattr(memory._method.memory, "add", capture)
        events = [MemoryEvent("message", "Assess ischemia", role="user"),
            MemoryEvent("action", '{"diagnosis": "ischemia", "code": 1}', role="assistant"),
            MemoryEvent("feedback", '{"reflection":{"future_strategy":"Check ECG"},"evaluation":{"score":0.8}}', role="assistant")]
        await memory.add(MemoryUnit(type="experience", episode_ref="clinical", task="Assess ischemia", events=events,
            output={"code": 1, "diagnosis": "ischemia"}, feedback={"text": '{"future_strategy":"Check ECG"}'}))
        assert batches == [[{"role": event.role, "content": event.content} for event in events]]


async def test_provider_failure_is_surfaced_and_frozen_state_restored(tmp_path):
    supplied = services()
    async with Memory(config_for(tmp_path), services=supplied) as memory:
        await memory.add(MemoryUnit(events=[MemoryEvent("message", "known", role="user")]))
        before = await memory._method.save()
        supplied.llm.fail = True
        with pytest.raises(Exception, match="controlled provider failure"):
            await memory.add(MemoryUnit(events=[MemoryEvent("message", "fail", role="user")]))
        # A failed extraction is not marked as an ingested source event.
        assert len(memory._method._seen_events) == 1
        await memory._method.load(before)
        assert await memory._method.save() == before


@pytest.mark.parametrize("params", [{"implementation_mode": "offline"}, {"algorithm_profile": "classic"},
                                      {"hybrid_search": False}, {"rerank": True}])
def test_simplified_algorithms_cannot_masquerade_as_native(params):
    report = Mem0Plugin().healthcheck(MethodConfig("mem0", params), services())
    assert not report["ready"] and report["config_errors"]


async def test_legacy_storage_states_and_team_scopes_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="team"):
        native_identity({"team_id": "ward"})
    async with Memory(config_for(tmp_path / "valid"), services=services()) as memory:
        with pytest.raises(ValueError, match="old classic"):
            await memory._method.load({"version": 2, "core": {}})
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    (legacy / "mem0.db").write_text("historical")
    backend = NativeBackend(services(), {"storage_path": str(legacy)}, 16, {"model": "controlled"})
    with pytest.raises(ConfigurationError, match="unmarked/legacy"):
        await backend.setup()
    assert (legacy / "mem0.db").read_text() == "historical"


async def test_public_example_executes_native_with_controlled_models(tmp_path, monkeypatch):
    from dataclasses import replace
    from mematlas import RuntimeConfig
    from mematlas.methods.mem0 import example
    supplied = services()
    def create(config):
        return Memory(replace(config, runtime=RuntimeConfig(artifact_dir=str(tmp_path)),
                              providers={"embedding": {"dimensions": 16}}), services=supplied)
    monkeypatch.setattr(example, "Memory", create)
    prompts = []
    async def agent(messages):
        prompts.append(messages)
        return "Helios port is 8443"
    assert await example.run(["Remember the port", "What port?"], agent, providers={}) == ["Helios port is 8443"] * 2
    assert "8443" in prompts[1][0]["content"]


async def test_subprocess_checkpoint_loads_in_new_worker_with_exact_rankings(tmp_path):
    from dataclasses import replace
    import sys
    from mematlas import RuntimeConfig
    config = replace(config_for(tmp_path), runtime=RuntimeConfig(
        executor="subprocess", python_executable=sys.executable, artifact_dir=str(tmp_path)))
    checkpoint = tmp_path / "checkpoint.json"
    async with Memory(config, services=services()) as memory:
        await memory.add(MemoryUnit(events=[MemoryEvent("message", "Helios", role="user")]))
        expected = await memory.search("Helios")
        await memory.save(checkpoint)
    async with Memory(config, services=services()) as restored:
        await restored.load(checkpoint)
        assert await restored.search("Helios") == expected
        async with restored.read_only():
            assert await restored.search("Helios") == expected
        usage = restored.stats()["usage"]
        assert usage


async def test_infer_false_is_the_official_raw_branch_without_llm_calls(tmp_path):
    supplied = services()
    async with Memory(config_for(tmp_path, infer=False, threshold=0), services=supplied) as memory:
        events = [MemoryEvent("message", "Alice: I visited Paris.", role="user"),
                  MemoryEvent("message", "Recorded.", role="assistant")]
        await memory.add(MemoryUnit(events=events))
        assert not supplied.llm.calls
        rows = (await memory.search("Alice Paris")).items
        assert {row["content"] for row in rows} == {event.content for event in events}
        assert all(set(row["source_event_ids"]) == {event.event_id for event in events} for row in rows)


async def test_vision_uses_native_caption_then_native_extraction(tmp_path):
    supplied = services()
    calls = []
    async def complete(messages, **options):
        calls.append(messages)
        if isinstance(messages[0].get("content"), list):
            return "A Helios system screenshot showing port 8443"
        return '{"memory":[{"event":"ADD","text":"Helios port is 8443"}]}'
    supplied.llm.complete = complete
    async with Memory(config_for(tmp_path, enable_vision=True), services=supplied) as memory:
        await memory.add(MemoryUnit(events=[MemoryEvent("observation", [
            {"type": "text", "text": "Remember this screenshot"},
            {"type": "image", "uri": "data:image/png;base64,aGVsaW9z"}], role="user")]))
        assert len(calls) == 2
        assert any(part.get("type") == "image_url" for part in calls[0][0]["content"])
        assert "A Helios system screenshot" in json.dumps(calls[1])
        assert "8443" in (await memory.search("Helios port")).items[0]["content"]


@pytest.mark.parametrize("timing,delay", [("immediate", 0), ("batch", 0), ("delayed", 1)])
async def test_actual_hospital_bridge_record_learn_retrieve_and_freeze_with_native_backend(tmp_path, timing, delay):
    from mematlas.integrations.openhospital.bridge import OpenHospitalMemoryIntegration
    supplied = services(facts=["Helios cardiology protocol requires checking ECG before treatment"])
    memory = Memory(config_for(tmp_path, write_policy="episode"), services=supplied)
    hospital = OpenHospitalMemoryIntegration(memory, top_k=None, feedback_timing=timing, feedback_delay=delay)
    state = {"timeline": [{"kind": "message", "role": "user", "text": "Chest pain with dyspnea"}]}
    report = {"diagnosisAccuracy": 0.8, "examinationPrecision": 0.7, "treatmentOverallScore": 0.9}
    try:
        await hospital.start()
        hospital.configure_training_batch(["p1", "p2"], batch_index=1)
        for index, patient in enumerate(("p1", "p2")):
            before = len(supplied.llm.calls)
            await hospital.record_execution(patient_id=patient, case_state=state,
                                            final_result={"diagnosis": "Suspected myocardial ischemia"})
            assert len(supplied.llm.calls) == before
            await hospital.learn_case(patient_id=patient, case_state=state,
                final_result={"diagnosis": "Suspected myocardial ischemia"},
                reflection={"future_strategy": "Verify ECG evidence before diagnosis"}, evaluation_report=report)
            assert len(supplied.llm.calls) == {"immediate": index + 1, "batch": 0, "delayed": index}[timing]
        if timing == "batch":
            await hospital.deliver_feedback(batch_end=True)
            assert len(supplied.llm.calls) == 2
        assert type(memory._method.memory).__module__ == "mem0.memory.main"
        before = await memory._method.save()
        async with memory.read_only():
            contexts = await hospital.retrieve(patient_id="held-out", phase="diagnosis", mode="test",
                chat_history=[{"role": "user", "content": "ECG for chest pain"}])
            assert any("ECG" in text for text in contexts)
        assert await memory._method.save() == before
        assert {row.operation for row in memory.traces.all()} >= {"add", "search", "episode_end"}
        if timing == "immediate":
            fingerprint = await hospital.state_fingerprint()
            snapshot = await hospital.snapshot_to(tmp_path / "hospital.json", learned_count=2)
            clone_services = services()
            clone = OpenHospitalMemoryIntegration(Memory(config_for(tmp_path / "relocated", write_policy="episode"),
                services=clone_services), top_k=None, feedback_timing=timing, feedback_delay=delay)
            try:
                await clone.start()
                await clone.restore_from(snapshot)
                assert not clone_services.llm.calls and not clone_services.embedder.calls
                assert await clone.state_fingerprint() == fingerprint
            finally:
                await clone.close()
    finally:
        await hospital.close()


async def test_efficiency_inventory_all_native_collections_no_models(tmp_path):
    from mematlas.integrations.openhospital.inspection import inspect_native
    supplied = services()
    async with Memory(config_for(tmp_path), services=supplied) as memory:
        await memory.add(MemoryUnit(events=[MemoryEvent("message", "Helios port 8443", role="user")]))
        await memory.flush()
        await memory._freeze_backend()
        try:
            before = len(supplied.llm.calls), len(supplied.embedder.calls)
            result = await inspect_native(memory._method, {"method":"mem0", "tokenizer_dir":str(Path(__file__).resolve().parents[2]/"models/Qwen3-Embedding-0.6B")})
            assert result["memory_text_tokens"] > 0 and result["storage_bytes"] > 0
            assert before == (len(supplied.llm.calls), len(supplied.embedder.calls))
            assert any("history" in name for name in result["text_layers"])
        finally:
            await memory._thaw_backend()
