"""Real official Memora + Chroma with deterministic provider responses.

No database, builder, segmenter, cue generator or policy retriever is mocked.
These tests verify integration/fidelity, not clinical quality or paid models.
Run in .venv-memora via scripts/check-memory-adapters.py.
"""
import asyncio
from copy import deepcopy
from datetime import datetime, timezone
import importlib
import json
from pathlib import Path

import pytest

pytest.importorskip("chromadb")
pytest.importorskip("omegaconf")

from mematlas.core.config import MethodConfig
from mematlas.core.models import MemoryEvent, MemoryUnit
from mematlas.core.plugin import MethodServices
from mematlas.methods.memora.method import MemoraMemory
from mematlas.methods.memora.native import NativeMemora, _load
from mematlas.methods.memora.utils import get_blob
from mematlas.test.memora.native_support import ScriptedLLM, FixedEmbedding

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / ".mematlas" / "upstream" / "memora"


def test_json_transport_keyword_preserves_native_messages():
    from mematlas.methods.memora.native import ServiceCompletions, _SERVICES, _FAILURES
    from types import SimpleNamespace
    seen=[]
    class LLM:
        async def complete(self,messages,**kwargs):
            seen.append((messages,kwargs))
            return '{}'
    key='json-transport-test'
    _SERVICES[key]=MethodServices(llm=LLM())
    _FAILURES[key]=[]
    try:
        transport=ServiceCompletions(SimpleNamespace(_mematlas_service_id=key))
        original=[{'role':'user','content':'Extract the requested fields.'}]
        transport.create(messages=original,response_format={'type':'json_object'})
        assert seen[-1][0][:-1]==original and len(original)==1
        assert 'JSON' in seen[-1][0][-1]['content']
        transport.create(messages=original)
        assert seen[-1][0]==original
        present=[{'role':'user','content':'Return JSON.'}]
        transport.create(messages=present,response_format={'type':'json_object'})
        assert seen[-1][0][:-1]==present
        transport.create(messages=original,response_format={'type':'json_schema','json_schema':{'name':'test','schema':{'type':'object'}}})
        assert seen[-1][0][:-1]==original and 'JSON' in seen[-1][0][-1]['content']
    finally:
        _SERVICES.pop(key)
        _FAILURES.pop(key)


def test_parse_validates_envelopes_and_bounds_format_repairs():
    from mematlas.methods.memora.native import ServiceCompletions, _SERVICES, _FAILURES
    from pydantic import BaseModel
    from types import SimpleNamespace
    class Entry(BaseModel):
        fact: str
    class Entries(BaseModel):
        entries: list[Entry]
    class LLM:
        calls=0
        replies=[]
        async def complete(self,messages,**kwargs):
            self.calls+=1
            return self.replies.pop(0)
    llm=LLM();key='parse-format-test'
    _SERVICES[key]=MethodServices(llm=llm);_FAILURES[key]=[]
    try:
        transport=ServiceCompletions(SimpleNamespace(_mematlas_service_id=key))
        for raw in [{'fact':'penicillin allergy'},[{'fact':'penicillin allergy'}],{'entries':[{'fact':'penicillin allergy'}]}]:
            llm.replies=[json.dumps(raw)]
            result=transport.parse(messages=[{'role':'user','content':'native history'}],response_format=Entries)
            assert result.choices[0].message.parsed.entries[0].fact=='penicillin allergy'
        llm.replies=['invalid json','{"entries":[{"fact":"preserved fact"}]}']
        before=llm.calls
        assert transport.parse(messages=[],response_format=Entries).choices[0].message.parsed.entries[0].fact=='preserved fact'
        assert llm.calls-before==2
        llm.replies=['invalid json']*4;before=llm.calls
        with pytest.raises(ValueError):transport.parse(messages=[],response_format=Entries)
        assert llm.calls-before==4 and len(_FAILURES[key])==1
    finally:
        _SERVICES.pop(key);_FAILURES.pop(key)


def test_native_nullable_update_index_preserves_chosen_candidate_without_new_facts():
    from mematlas.methods.memora.native import ServiceCompletions, _SERVICES, _FAILURES
    from types import SimpleNamespace
    _load(SOURCE)
    from memora.utils.memory import MemoryUpdateDecision
    class LLM:
        calls=0
        replies=[]
        async def complete(self,messages,**kwargs):
            self.calls+=1
            return json.dumps(self.replies.pop(0))
    llm=LLM();key='nullable-update-test'
    _SERVICES[key]=MethodServices(llm=llm);_FAILURES[key]=[]
    prompt='Candidate 0:\n- Similarity Score: 0.86\n- Index: original key\n- Value: original fact\n\nCandidate 1:\n- Similarity Score: 0.84\n- Index: selected original key\n- Value: selected fact\n'
    raw={'should_update':True,'best_candidate_index':1,'updated_index':None,'updated_value':'unchanged combined facts','updated_cues':[]}
    try:
        transport=ServiceCompletions(SimpleNamespace(_mematlas_service_id=key))
        llm.replies=[raw]
        parsed=transport.parse(messages=[{'role':'user','content':prompt}],response_format=MemoryUpdateDecision).choices[0].message.parsed
        assert parsed.updated_index=='selected original key' and parsed.updated_value==raw['updated_value'] and llm.calls==1
        # A missing update value/candidate still requires model repair; never
        # manufacture content or switch the update into an additive fallback.
        llm.replies=[{**raw,'best_candidate_index':99}]*4
        with pytest.raises(ValueError,match='best_candidate_index'):
            transport.parse(messages=[{'role':'user','content':prompt}],response_format=MemoryUpdateDecision)
        llm.replies=[{**raw,'updated_value':None}]*4
        with pytest.raises(ValueError,match='updated_value'):
            transport.parse(messages=[{'role':'user','content':prompt}],response_format=MemoryUpdateDecision)
    finally:
        _SERVICES.pop(key);_FAILURES.pop(key)


def events(prefix=""):
    stamp = datetime(2026, 11, 5, 9, tzinfo=timezone.utc)
    return [MemoryEvent("message", "I am allergic to penicillin.", role="user", event_id=prefix + "p", timestamp=stamp),
            MemoryEvent("message", "I will consider the allergy.", role="assistant", event_id=prefix + "d", timestamp=stamp)]


async def make(tmp_path):
    services = MethodServices(llm=ScriptedLLM(), embedder=FixedEmbedding(), artifact_dir=tmp_path,
        source_roots={"memora": str(SOURCE)})
    adapter = MemoraMemory(MethodConfig("memora"), services)
    await adapter.setup(None)
    return adapter, services


def test_streaming_checkpoint_is_identical_and_validates_existing_blob(tmp_path):
    from mematlas.methods.memora.utils import canonical, put_json_array_blob
    rows = [{"embedding": [0.1, 0.2], "text": "中文", "id": "a"}, {"id": "b", "embedding": [0.3]}]
    ref = put_json_array_blob(tmp_path, iter(rows))
    assert get_blob(ref) == canonical(rows)
    assert put_json_array_blob(tmp_path, iter(rows)) == ref
    assert get_blob(put_json_array_blob(tmp_path, iter(()))) == b"[]"
    Path(ref["path"]).write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="Corrupt immutable"):
        put_json_array_blob(tmp_path, iter(rows))
    def broken():
        yield rows[0]
        raise MemoryError("controlled serialization interruption")
    with pytest.raises(MemoryError):
        put_json_array_blob(tmp_path, broken())
    assert not list(tmp_path.glob(".pending-*"))


@pytest.mark.asyncio
async def test_deep_chroma_storage_and_streamed_exact_vector_snapshot(tmp_path):
    from mematlas.core.files import storage_bytes
    from mematlas.methods.memora.utils import canonical
    directory = tmp_path
    while len(str(directory.resolve())) < 205:
        directory /= "deep-native-state"
    adapter, services = await make(directory)
    try:
        await adapter.add(MemoryUnit(type="experience", episode_ref="long-path", events=events()))
        native = next(iter(adapter.backends.values()))
        collection = native.store.collection
        # Cross Chroma's persistence threshold so index_metadata.pickle exists.
        collection.upsert(ids=[f"controlled-{i}" for i in range(1100)],
            documents=[f"controlled memory {i}" for i in range(1100)],
            metadatas=[{"index":f"controlled memory {i}", "value":"fixture", "user_id":native.agent.get_user_id()} for i in range(1100)],
            embeddings=[[1.0, 0.1, 0.2, 0.3] for _ in range(1100)])
        expected = collection.get(include=["documents", "metadatas", "embeddings"])
        raw_rows = [{"id":key,"document":expected["documents"][i],"metadata":expected["metadatas"][i],
                     "embedding":list(expected["embeddings"][i])} for i,key in enumerate(expected["ids"])]
        snapshot = native.export_state(adapter.snapshots)
        assert get_blob(snapshot["collection"]) == canonical(raw_rows)
        ordinary = str(native.directory).removeprefix("\\\\?\\")
        assert storage_bytes(ordinary) > 0
        metadata = list(native.directory.glob("*/index_metadata.pickle"))
        assert metadata and len(str(metadata[0])) > 260
        services.llm.deny = services.embedder.deny = True
        native.import_state(snapshot)
        assert native.export_state(adapter.snapshots) == snapshot
    finally:
        await adapter.close()


@pytest.mark.asyncio
async def test_deleting_former_cue_primary_preserves_other_facts_and_vectors(tmp_path):
    adapter, services = await make(tmp_path)
    try:
        await adapter.add(MemoryUnit(type="experience", episode_ref="cue-lifecycle", events=events()))
        native = next(iter(adapter.backends.values()))
        from memora.core.memory_entry import MemoryEntry
        owner = native.agent.get_user_id()
        primary = MemoryEntry(index="original primary", value="original retained fact", user_id=owner,
                              memory_type="factual", cue_indices="future primary||remaining cue")
        await asyncio.to_thread(native.agent.add, primary)
        # Reproduce official builder promotion of an existing cue phrase.
        converted = MemoryEntry(index="future primary", value="promoted retained fact", user_id=owner,
                                memory_type="factual")
        await asyncio.to_thread(native.store.upsert, index=converted.index, value=converted.value, metadata=converted.get_metadata())
        before = native.store.collection.get(include=["metadatas", "embeddings", "documents"])
        services.llm.deny = services.embedder.deny = True
        await asyncio.to_thread(native.agent.delete, converted.index)
        remaining = native.agent.get(primary.index)
        assert remaining.value == primary.value and remaining.get_cue_indices() == ["remaining cue"]
        after = native.store.collection.get(include=["metadatas", "embeddings", "documents"])
        vectors = {key:list(v) for key,v in zip(before["ids"],before["embeddings"])}
        assert all(list(v) == vectors[key] for key,v in zip(after["ids"],after["embeddings"]))
        assert native.agent.get("remaining cue").get_linked_memories() == [primary.index]
        services.llm.deny = services.embedder.deny = False
        await asyncio.to_thread(native.agent.delete, primary.index)  # Used to fail on the removed cue.
        assert native.agent.get(primary.index) is None and native.agent.get("remaining cue") is None
        assert native.agent.get("Patient allergy 1") is not None
    finally:
        await adapter.close()


@pytest.mark.asyncio
async def test_native_full_chain_scope_exact_vectors_and_frozen_restore(tmp_path):
    adapter, services = await make(tmp_path)
    owner = {"agent_id": "doctor", "user_id": "patient", "session_id": "one"}
    # OpenHospital boundary comes before complete experience.
    await adapter.episode_end("case", scope=owner, status="completed")
    await adapter.add(MemoryUnit(write_ref="case-experience", episode_ref="case", type="experience", events=events(), scope=owner))
    native = next(iter(adapter.backends.values()))
    assert type(native.builder).__module__ == "memora.builder.chat_memory_builder"
    assert type(native.agent).__module__ == "memora.core.memory"
    collection = native.store.collection.get(include=["metadatas", "embeddings"])
    metas = collection["metadatas"]
    assert any(meta.get("memory_type") == "episodic" for meta in metas)
    assert any(meta.get("memory_type") == "factual" and meta.get("episodic_memory_ids") for meta in metas)
    assert any(meta.get("linked_memory") for meta in metas)
    assert native.store._bm25_indices
    used = await adapter.search("penicillin allergy", level_ref=None, top_k=1, scope=owner, extensions={})
    assert used.items
    assert any(item["metadata"]["memory_type"] == "episodic" for item in used.items)
    assert [step["action"] for step in used.extensions["memora"]["trace"]] == ["INIT_RETRIEVE", "RE_QUERY", "STOP"]
    assert all(item["source_event_ids"] for item in used.items)
    native_order = collection["ids"]
    bm25_order = deepcopy(native.store._bm25_doc_ids)
    # The pinned builder can leave this optional cache empty. Seed it with
    # actual stored native vectors to exercise a nonempty LRU across restore.
    for metadata,vector in zip(collection["metadatas"][:2],collection["embeddings"][:2]):
        native.store._cache_embedding(metadata['index'],list(vector))
    native_cache = deepcopy(list(native.store._embedding_cache.items()))
    assert native_cache
    snapshot = await adapter.save()
    assert snapshot == await adapter.save()
    services.llm.deny = services.embedder.deny = True
    await adapter.reset(owner)
    await adapter.load(snapshot)
    assert snapshot == await adapter.save()
    recovered = next(iter(adapter.backends.values())).store.collection.get(include=["metadatas", "embeddings"])
    before_vectors = {key: list(vector) for key, vector in zip(collection["ids"], collection["embeddings"])}
    after_vectors = {key: list(vector) for key, vector in zip(recovered["ids"], recovered["embeddings"])}
    assert before_vectors == after_vectors
    assert recovered["ids"] == native_order
    assert next(iter(adapter.backends.values())).store._bm25_doc_ids == bm25_order
    assert list(next(iter(adapter.backends.values())).store._embedding_cache.items()) == native_cache
    services.llm.deny = services.embedder.deny = False
    repeated = await adapter.search("penicillin allergy", level_ref=None, top_k=1, scope=owner, extensions={})
    assert repeated.items == used.items
    def decisions(trace):
        return [{key: value for key, value in step.items() if key not in {"duration", "llm_duration"}} for step in trace]
    assert decisions(repeated.extensions["memora"]["trace"]) == decisions(used.extensions["memora"]["trace"])
    other = {**owner, "session_id": "two"}
    assert not (await adapter.search("q", level_ref=None, top_k=None, scope=other, extensions={})).items
    await adapter.add(MemoryUnit(write_ref="other", episode_ref="other", type="experience", events=events("other"), scope=other))
    assert len(adapter.backends) == 2
    await adapter.reset(owner)
    assert (await adapter.search("q", level_ref=None, top_k=1, scope=other, extensions={})).items
    await adapter.close()


@pytest.mark.asyncio
async def test_native_direct_pipeline_differential(tmp_path, monkeypatch):
    adapter, services = await make(tmp_path / "adapter")
    owner = {"agent_id": "doctor"}
    # Freeze only wall-clock fields for a reproducible source-vs-wrapper diff.
    _load(SOURCE)
    for module_name in ("memora.utils.misc", "memora.utils.memory", "memora.builder.memory_builder"):
        module = importlib.import_module(module_name)
        if hasattr(module, "get_current_timestamp"):
            monkeypatch.setattr(module, "get_current_timestamp", lambda: "2026-11-05 09:00:00")
    await adapter.add(MemoryUnit(write_ref="w", episode_ref="case", type="experience", events=events(), scope=owner))
    wrapped = next(iter(adapter.backends.values()))
    direct_services = MethodServices(llm=ScriptedLLM(), embedder=FixedEmbedding(), artifact_dir=tmp_path / "direct", source_roots={"memora": str(SOURCE)})
    direct_services._native_io_loop = asyncio.get_running_loop()
    direct = await asyncio.to_thread(NativeMemora, tmp_path / "direct" / "db", owner, {}, direct_services)

    def official_calls():
        # Actual public native components, following official application add.py
        # (the LocalMemoraClient facade itself never performs segmentation).
        messages = [{"role": event.role, "content": event.content} for event in events()]
        for i, segment in enumerate(direct.segmenter.segment_conversation(messages)):
            metadata = {"timestamp": events()[0].timestamp.isoformat(), "segment_topic": segment["topic"], "segment_index": i,
                        "episode_ref": "case", "episode_part": 0, "source_event_ids": json.dumps(["p", "d"])}
            direct.builder.build([messages[j] for j in segment["indices"]], metadata=metadata)
        direct.store.build_bm25_index()
    await asyncio.to_thread(official_calls)
    one = await asyncio.to_thread(wrapped.export_state, tmp_path / "snapshots")
    two = await asyncio.to_thread(direct.export_state, tmp_path / "snapshots")
    assert json.loads(get_blob(one["collection"])) == json.loads(get_blob(two["collection"]))
    assert services.llm.calls == direct_services.llm.calls
    # Transport schema is the exact OpenAI SDK strict-schema format.
    assert all(call[1]["response_format"]["json_schema"]["strict"] for call in services.llm.calls if "response_format" in call[1])
    before = await asyncio.to_thread(wrapped.search, "allergy", 1)
    after = await asyncio.to_thread(direct.search, "allergy", 1)
    assert [(row["memory_id"], row["content"]) for row in before[0]] == [(row["memory_id"], row["content"]) for row in after[0]]
    assert [step["action"] for step in before[1]] == [step["action"] for step in after[1]]
    await adapter.close()
    await asyncio.to_thread(direct.close)


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["llm", "embedder"])
async def test_native_provider_failure_is_not_successful_fallback(tmp_path, provider):
    adapter, services = await make(tmp_path)
    if provider == "llm":
        complete = services.llm.complete
        async def fail_segmentation_only(messages, **kwargs):
            if kwargs.get("response_format", {}).get("json_schema", {}).get("name") == "SegmentationOutput":
                raise RuntimeError("controlled segmentation transport failure")
            return await complete(messages, **kwargs)
        services.llm.complete = fail_segmentation_only
        expected, message = RuntimeError, "provider failed"
    else:
        services.embedder.deny = True
        expected, message = AssertionError, "Embedding called during restoration"
    try:
        with pytest.raises(expected, match=message):
            await adapter.add(MemoryUnit(write_ref="failed", episode_ref="case", type="experience", events=events(), scope={"agent_id": "doctor"}))
        assert not next(iter(adapter.scopes.values()))["seen_events"]
    finally:
        await adapter.close()


@pytest.mark.asyncio
async def test_efficiency_inventory_includes_body_index_episodic_and_cues(tmp_path):
    from mematlas.integrations.openhospital.inspection import inspect_native
    adapter, services = await make(tmp_path)
    try:
        await adapter.add(MemoryUnit(type="experience", episode_ref="inventory", events=events()))
        await adapter.episode_end("inventory", scope={}, status="completed")
        await adapter.flush()
        result = await inspect_native(adapter, {"method":"memora","tokenizer_dir":str(ROOT/"models/Qwen3-Embedding-0.6B")})
        assert result["memory_text_tokens"] > 0 and result["storage_bytes"] > 0
        assert any(row["field"] == "value" for row in result["text_objects"])
        assert any(row["field"] == "index" for row in result["text_objects"])
    finally:
        await adapter.close()
