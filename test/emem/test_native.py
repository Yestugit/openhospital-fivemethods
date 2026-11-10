"""Official EMem graph/OpenIE/retrieval against a local fixed-response API.

Only provider responses are controlled; all original algorithm/storage code runs.
Run in .venv-emem with scripts/check-memory-adapters.py. No remote model calls.
"""
import asyncio
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib
import json
from pathlib import Path
import threading
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")
pytest.importorskip("igraph")
pytest.importorskip("transformers")

from mematlas.core.config import MethodConfig
from mematlas.core.models import MemoryEvent, MemoryUnit
from mematlas.core.plugin import MethodServices
from mematlas.methods.emem.method import EMemMemory, PLUGIN
from mematlas.methods.emem.native import NativeEMem
from mematlas.methods.emem.utils import get_blob
from mematlas.test.emem.native_support import FixedEmbedding

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / ".mematlas" / "upstream" / "emem"
EDU = "The patient reports a penicillin allergy."


@pytest.fixture
def endpoint():
    calls = []
    controls = {"reject": False}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            calls.append(body)
            if controls["reject"]:
                data = json.dumps({"error": {"message": "controlled authentication failure", "type": "invalid_api_key"}}).encode()
                self.send_response(401)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            messages = body["messages"]
            prompt = messages[0]["content"]
            if "edu_after_filter" in prompt:
                value = '[[ ## edu_after_filter ## ]]\n' + json.dumps({"edu": [EDU]}) + '\n[[ ## completed ## ]]'
            elif "selected_edus" in prompt:
                value = json.dumps({"selected_edus": [EDU]})
            elif "selected_arguments" in prompt:
                value = json.dumps({"selected_arguments": ["patient", "penicillin"]})
            elif "entities_and_concepts" in prompt:
                value = json.dumps({"entities_and_concepts": ["penicillin"]})
            elif "role" in prompt.lower() and "argument" in prompt.lower() and "Elementary Discourse Unit" in prompt:
                value = json.dumps({"event_type": "Allergy Report", "role_argument_pairs": [
                    {"role": "patient", "argument": "patient"}, {"role": "allergen", "argument": "penicillin"}]})
            else:
                value = json.dumps({"edus": [{"edu_text": EDU, "source_turn_ids": [1, 2]}]})
            response = {"id": "local-fixed-response", "object": "chat.completion", "created": 1, "model": body["model"],
                "choices": [{"index": 0, "message": {"role": "assistant", "content": value}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}}
            data = json.dumps(response).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}/v1", calls, controls
    server.shutdown()
    thread.join()


async def make(tmp_path, endpoint, monkeypatch, variant="emem", **params):
    monkeypatch.setenv("EMEM_NATIVE_TEST_KEY", "local-test-placeholder")
    services = MethodServices(embedder=FixedEmbedding(), artifact_dir=tmp_path,
        source_roots={"emem": str(SOURCE)}, provider_configs={
            "llm": {"model": "gpt-4o-mini", "base_url": endpoint[0], "api_key_env": "EMEM_NATIVE_TEST_KEY"},
            "embedding": {"model": "configured-embedding"}})
    adapter = EMemMemory(MethodConfig("emem", {"variant": variant, **params}), services)
    await adapter.setup(None)
    return adapter, services


@pytest.mark.asyncio
async def test_json_unicode_uses_native_schemas_graph_cache_and_restores_without_calls(tmp_path, endpoint, monkeypatch):
    adapter, services = await make(tmp_path, endpoint, monkeypatch, json_output_mode="json_unicode_v1")
    try:
        await adapter.add(MemoryUnit(write_ref="structured", episode_ref="structured", type="experience", events=events()))
        used = await adapter.search("penicillin allergy", level_ref=None, top_k=1, scope={}, extensions={})
        assert used.items[0]['content'] == EDU
        assert all(call.get('response_format') == {'type':'json_object'} for call in endpoint[1])
        native = next(iter(adapter.backends.values()))
        journal = native.directory/'structured-responses.jsonl'
        records = [json.loads(line) for line in journal.read_text(encoding='utf8').splitlines()]
        assert len(records) == len(endpoint[1])
        assert {'ConversationEDUExtractionV1','UnifiedEventExtraction','QueryERExtraction','FilteredEDUs'} <= {r['schema'] for r in records}
        assert all(r['validation_error'] is None for r in records)
        saved = await adapter.save()
        assert saved['adapter_profile'] == 'json_unicode_v1'
        with pytest.raises(ValueError, match='profile mismatch'):
            await adapter.load({k:v for k,v in saved.items() if k != 'adapter_profile'})
        count_before = len(endpoint[1])
        services.embedder.deny = True
        await adapter.reset({})
        await adapter.load(saved)
        assert len(endpoint[1]) == count_before
    finally:
        await adapter.close()


def events(prefix=""):
    date = datetime(2026, 11, 5, 9, tzinfo=timezone.utc)
    return [MemoryEvent("message", "I am allergic to penicillin.", role="user", event_id=prefix + "p", timestamp=date),
            MemoryEvent("message", "I will consider the allergy.", role="assistant", event_id=prefix + "d", timestamp=date)]


@pytest.mark.asyncio
@pytest.mark.parametrize("variant", ["emem", "emem_g"])
async def test_official_graph_retrieval_cache_and_exact_restore(tmp_path, endpoint, monkeypatch, variant):
    adapter, services = await make(tmp_path, endpoint, monkeypatch, variant)
    owner = {"agent_id": "doctor", "user_id": "patient", "session_id": "one"}
    await adapter.episode_end("case", scope=owner, status="completed")
    await adapter.add(MemoryUnit(write_ref="case-experience", episode_ref="case", type="experience", events=events(), scope=owner))
    native = next(iter(adapter.backends.values()))
    assert type(native.engine).__module__ == "emem.EMem"
    assert native.engine.graph.vcount() >= 4
    assert native.engine.graph.ecount() > 0
    assert native.engine.edu_embedding_store.get_all_ids()
    assert native.engine.argument_embedding_store.get_all_ids()
    used = await adapter.search("penicillin allergy", level_ref=None, top_k=1, scope=owner, extensions={})
    assert used.items[0]["content"] == EDU
    assert used.items[0]["source_event_ids"] == ["p", "d"]
    assert "score" in used.items[0] if variant == "emem_g" else "score" not in used.items[0]
    original_count = len(endpoint[1])
    await adapter.search("penicillin allergy", level_ref=None, top_k=1, scope=owner, extensions={})
    assert len(endpoint[1]) == original_count
    assert any(record.provider == "emem_native_cache" for record in services.usage._records)
    assert sum(record.resource_kind == "llm" for record in services.usage._records) == original_count
    # Differential: the actual native retrieval call returns identical EDU IDs
    # and text to the adapter's contribution without a replacement retriever.
    direct, trace = await asyncio.to_thread(native.engine.retrieve_structuring_conversation_enhanced_v2,
        ["penicillin allergy"], num_to_retrieve=1, no_query_trace_saving=True, use_batched_reranking=True)
    assert [entry.edu_text for entry in direct[0].edus] == [row["content"] for row in used.items]
    snapshot = await adapter.save()
    assert snapshot == await adapter.save()
    vectors = [list(row) for row in native.engine.edu_embedding_store.get_embeddings(native.engine.edu_embedding_store.get_all_ids())]
    services.embedder.deny = True
    count_before_load = len(endpoint[1])
    await adapter.reset(owner)
    await adapter.load(snapshot)
    assert len(endpoint[1]) == count_before_load
    assert snapshot == await adapter.save()
    restored = next(iter(adapter.backends.values()))
    recovered_vectors = [list(row) for row in restored.engine.edu_embedding_store.get_embeddings(restored.engine.edu_embedding_store.get_all_ids())]
    assert vectors == recovered_vectors
    assert restored.engine.graph.vcount() == native.engine.graph.vcount()
    services.embedder.deny = False
    other = {**owner, "session_id": "two"}
    assert not (await adapter.search("q", level_ref=None, top_k=1, scope=other, extensions={})).items
    await adapter.add(MemoryUnit(write_ref="other", episode_ref="other", type="experience", events=events("other"), scope=other))
    assert len(adapter.backends) == 2
    await adapter.reset(owner)
    assert (await adapter.search("q", level_ref=None, top_k=1, scope=other, extensions={})).items
    await adapter.close()


@pytest.mark.asyncio
async def test_native_embedding_batch_size_and_normalization(tmp_path, endpoint, monkeypatch):
    adapter, services = await make(tmp_path, endpoint, monkeypatch)
    key, _ = adapter._scope({"agent_id": "doctor"})
    native = await adapter._backend(key)
    vectors = await asyncio.to_thread(native.engine.embedding_model.batch_encode, [str(i) for i in range(35)])
    assert [len(batch) for batch in services.embedder.calls] == [16, 16, 3]
    import numpy as np
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1)
    # Run the official OpenAIEmbeddingModel.batch_encode body with only its
    # encode transport bound to the same public embedding function.
    official = importlib.import_module("emem.embedding_model.OpenAI").OpenAIEmbeddingModel
    direct = official.__new__(official)
    direct.embedding_config = native.engine.embedding_model.embedding_config
    direct.encode = native.engine.embedding_model.encode
    services.embedder.calls.clear()
    expected = await asyncio.to_thread(official.batch_encode, direct, [str(i) for i in range(35)], batch_size=16)
    assert [len(batch) for batch in services.embedder.calls] == [16, 16, 3]
    assert np.array_equal(vectors, expected)
    await adapter.close()


@pytest.mark.asyncio
async def test_native_restore_to_empty_does_not_resurrect_db(tmp_path, endpoint, monkeypatch):
    adapter, services = await make(tmp_path, endpoint, monkeypatch)
    empty = await adapter.save()
    owner = {"agent_id": "doctor"}
    await adapter.add(MemoryUnit(write_ref="a", episode_ref="case", type="experience", events=events(), scope=owner))
    await adapter.load(empty)
    assert not adapter.backends
    key, _ = adapter._scope(owner)
    backend = await adapter._backend(key)
    assert not backend.engine.edu_embedding_store.get_all_ids()
    await adapter.close()


def test_preflight_checks_native_api_key_without_constructing_backend(tmp_path, monkeypatch):
    monkeypatch.delenv("EMEM_NATIVE_TEST_KEY", raising=False)
    directory = tmp_path / "must-not-create"
    supplied = MethodServices(llm=SimpleNamespace(), embedder=FixedEmbedding(), artifact_dir=directory,
        source_roots={"emem": str(SOURCE)}, provider_configs={
            "llm": {"model": "gpt-4o-mini", "api_key_env": "EMEM_NATIVE_TEST_KEY"}})
    report = PLUGIN.healthcheck(MethodConfig("emem"), supplied)
    assert not report["ready"]
    assert any("EMEM_NATIVE_TEST_KEY" in error for error in report["config_errors"])
    supplied.llm.api_key = "local-test-placeholder"
    assert PLUGIN.healthcheck(MethodConfig("emem"), supplied)["ready"]
    assert not directory.exists()


@pytest.mark.asyncio
async def test_native_provider_failure_is_not_successful_fallback(tmp_path, endpoint, monkeypatch):
    adapter, supplied = await make(tmp_path, endpoint, monkeypatch)
    scope = {"agent_id": "doctor"}
    key, _ = adapter._scope(scope)
    native = await adapter._backend(key)
    # Exercise the actual native retry/fallback path, bounded to one local 401
    # for this failure test. Successful/native acceptance keeps original defaults.
    native.engine.llm_model.max_attempts = 1
    endpoint[2]["reject"] = True
    try:
        with pytest.raises(RuntimeError, match="provider failed"):
            await adapter.add(MemoryUnit(write_ref="failing", episode_ref="case", type="experience", scope=scope, events=events()))
        failures = [record for record in supplied.usage.all() if record.status == "failed"]
        assert len(failures) == len(endpoint[1]) > 0
        assert all(not record.usage_known for record in failures)
        assert not adapter.scopes[key]["seen_events"]
    finally:
        await adapter.close()


@pytest.mark.asyncio
async def test_efficiency_inventory_includes_all_edu_argument_and_session_objects(tmp_path, endpoint, monkeypatch):
    from mematlas.integrations.openhospital.inspection import inspect_native
    adapter, services = await make(tmp_path, endpoint, monkeypatch)
    try:
        await adapter.add(MemoryUnit(type="experience", episode_ref="inventory", events=events()))
        await adapter.episode_end("inventory", scope={}, status="completed")
        await adapter.flush()
        before = len(endpoint[1])
        result = await inspect_native(adapter, {"method":"emem","tokenizer_dir":str(ROOT/"models/Qwen3-Embedding-0.6B")})
        assert result["memory_text_tokens"] > 0 and result["storage_bytes"] > 0
        assert len(endpoint[1]) == before
        assert "emem:edu_embedding_store" in result["text_layers"]
        assert "emem:session_content_store" in result["text_layers"]
    finally:
        await adapter.close()
