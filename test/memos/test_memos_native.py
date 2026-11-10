"""Run with the locked MemOS interpreter; native graph tests require Neo4j.

Set MEMATLAS_TEST_NEO4J_URI and MEMATLAS_TEST_NEO4J_PASSWORD to a separately
managed Neo4j >=5.18 test service. Tests use isolated adapter-owned tenants.
Scripted responses replace inference only, never the upstream memory algorithm.
"""
from __future__ import annotations
import asyncio
import hashlib
from importlib.util import find_spec
import json
import os
from pathlib import Path
import sys
import unittest
from uuid import uuid4

from mematlas import Memory, MemAtlasConfig, MemoryEvent, MemoryUnit
from mematlas.core.config import RuntimeConfig
from mematlas.core.plugin import MethodServices
from mematlas.methods.memos import MemOSConfig


class ScriptedLLM:
    def __init__(self, call_log=None):
        self.calls = 0
        self.call_log = call_log

    def construction_spec(self):
        return {"factory": "mematlas.test.memos.test_memos_native:ScriptedLLM",
                "kwargs": {"call_log": self.call_log}}

    async def complete(self, messages, **kwargs):
        self.calls += 1
        if self.call_log:
            with Path(self.call_log).open("a", encoding="utf-8") as log:
                log.write("llm\n")
        return json.dumps({"memory list": [{"key": "allergy",
            "memory_type": "LongTermMemory", "value": "Patient has penicillin allergy",
            "tags": ["health"]}], "summary": "allergy history"})


class ScriptedEmbedding:
    model = "deterministic-native-test"

    def __init__(self, call_log=None, dimensions=8):
        self.call_log = call_log
        self.dimensions = dimensions

    def construction_spec(self):
        return {"factory": "mematlas.test.memos.test_memos_native:ScriptedEmbedding",
                "kwargs": {"call_log": self.call_log, "dimensions": self.dimensions}}

    async def embed(self, texts):
        if self.call_log:
            with Path(self.call_log).open("a", encoding="utf-8") as log:
                log.write("embedding\n")
        return [[float(hashlib.sha256(text.encode()).digest()[index % 32] + 1)
                 for index in range(self.dimensions)] for text in texts]


def graph_config(path, label, *, executor="in_process"):
    return MemAtlasConfig(method=MemOSConfig(embedding_dims=1024, storage_path=str(path / label),
        neo4j_uri=os.environ["MEMATLAS_TEST_NEO4J_URI"],
        neo4j_user=os.environ.get("MEMATLAS_TEST_NEO4J_USER", "neo4j"),
        neo4j_password_env="MEMATLAS_TEST_NEO4J_PASSWORD",
        neo4j_database=os.environ.get("MEMATLAS_TEST_NEO4J_DATABASE", "neo4j")),
        runtime=RuntimeConfig(executor=executor, python_executable=sys.executable,
            artifact_dir=str(path / (label + "-artifacts"))), scope={"user_id": "alice"})


@unittest.skipUnless(find_spec("memos"), "Run in the method's requirements.lock environment")
class NativeReaderTests(unittest.IsolatedAsyncioTestCase):
    async def test_native_temporal_checkpoint_types_roundtrip(self):
        from neo4j.time import Date, DateTime, Time, Duration
        from mematlas.methods.memos.core import _decode, _encode
        values = [Date(2026, 1, 1), DateTime(2026, 1, 1, 12, 30),
            Time(12, 30, 45), Duration(months=1, days=2, seconds=3, nanoseconds=4), b"native-bytes"]
        self.assertEqual(_decode(_encode(values)), values)
        self.assertEqual([type(value) for value in _decode(_encode(values))], [type(value) for value in values])

    async def test_native_fine_reader_with_public_provider_bridge(self):
        from memos.configs.mem_reader import SimpleStructMemReaderConfig
        from memos.llms.factory import LLMFactory
        from memos.embedders.factory import EmbedderFactory
        from memos.mem_reader.simple_struct import SimpleStructMemReader
        from memos.mem_reader.utils import count_tokens_text
        from mematlas.methods.memos.core import _FACTORY_LOCK, _LLM, _Embedding

        llm, embedding, loop = ScriptedLLM(), ScriptedEmbedding(), asyncio.get_running_loop()
        name = "native-reader-" + uuid4().hex
        config = SimpleStructMemReaderConfig.model_validate({
            "llm": {"backend": "openai", "config": {"model_name_or_path": name,
                "api_key": "mematlas-service-bridge", "api_base": "http://127.0.0.1"}},
            "embedder": {"backend": "universal_api", "config": {"provider": "openai",
                "model_name_or_path": name, "api_key": "mematlas-service-bridge",
                "embedding_dims": 8, "max_tokens": None}},
            "chunker": {"backend": "sentence", "config": {"save_rawfile": False}}})
        config.chunker.config.tokenizer_or_token_counter = count_tokens_text

        def read():
            with _FACTORY_LOCK:
                old_llm, old_embedder = LLMFactory.backend_to_class["openai"], EmbedderFactory.backend_to_class["universal_api"]
                LLMFactory.backend_to_class["openai"] = lambda cfg: _LLM(llm, loop, cfg)
                EmbedderFactory.backend_to_class["universal_api"] = lambda cfg: _Embedding(embedding, loop, cfg, 8)
                try:
                    reader = SimpleStructMemReader(config)
                finally:
                    LLMFactory.backend_to_class["openai"], EmbedderFactory.backend_to_class["universal_api"] = old_llm, old_embedder
            return reader.get_memory([[{"role": "user", "content": "I have penicillin allergy",
                "chat_time": "2026-01-01T00:00:00+00:00"}]], type="chat", mode="fine",
                info={"user_id": "native-reader", "session_id": "episode"})

        groups = await asyncio.to_thread(read)
        rows = [row for group in groups for row in group]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].memory, "Patient has penicillin allergy")
        self.assertEqual(rows[0].metadata.memory_type, "LongTermMemory")
        self.assertEqual(len(rows[0].metadata.embedding), 8)
        self.assertEqual(rows[0].metadata.session_id, "episode")
        self.assertEqual(llm.calls, 1)


_GRAPH_READY = bool(find_spec("memos") and os.environ.get("MEMATLAS_TEST_NEO4J_URI")
    and os.environ.get("MEMATLAS_TEST_NEO4J_PASSWORD"))


@unittest.skipUnless(_GRAPH_READY,
    "Requires locked MemOS environment plus MEMATLAS_TEST_NEO4J_URI/PASSWORD for real Neo4j >=5.18")
class NativeGraphTests(unittest.IsolatedAsyncioTestCase):
    async def test_new_subprocess_restore_without_model_calls_and_continue_writing(self):
        path = Path(__file__).resolve().parents[2] / ".mematlas" / ("native-memos-worker-" + uuid4().hex)
        path.mkdir(parents=True)
        call_log = path / "provider-calls.log"
        checkpoint = path / "checkpoint.json"

        def services():
            return MethodServices(llm=ScriptedLLM(str(call_log)), embedder=ScriptedEmbedding(str(call_log), dimensions=1024))

        async with Memory(graph_config(path, "original", executor="subprocess"), services=services()) as original:
            try:
                await original.add(MemoryUnit(events=[MemoryEvent("message", "penicillin allergy", role="user")]))
                found = await original.search("penicillin allergy")
                self.assertTrue(found.items)
                await original.save(checkpoint)
                async with Memory(graph_config(path, "restored", executor="subprocess"), services=services()) as restored:
                    try:
                        before = call_log.read_text(encoding="utf-8")
                        await restored.load(checkpoint)
                        self.assertEqual(before, call_log.read_text(encoding="utf-8"))
                        async with restored.read_only():
                            self.assertEqual(found.items, (await restored.search("penicillin allergy")).items)
                        await restored.add(MemoryUnit(events=[MemoryEvent("message", "Verify allergy history", role="user")]))
                        self.assertGreater(call_log.read_text(encoding="utf-8").count("llm\n"), before.count("llm\n"))
                        self.assertTrue((await original.search("penicillin allergy")).items)
                    finally:
                        await restored.reset()
            finally:
                await original.reset()

    async def test_real_openhospital_boundary_then_delayed_experience_and_feedback(self):
        from mematlas.integrations.openhospital import OpenHospitalMemoryIntegration
        path = Path(__file__).resolve().parents[2] / ".mematlas" / ("native-memos-hospital-" + uuid4().hex)
        services = MethodServices(llm=ScriptedLLM(), embedder=ScriptedEmbedding(dimensions=1024))
        bridge = OpenHospitalMemoryIntegration.from_config(graph_config(path, "store"),
            services=services, journal_dir=path / "encounters")
        state = {"chat_history": [{"from": "patient", "text": "penicillin allergy"}],
            "timeline": [{"kind": "message", "role": "user", "text": "penicillin allergy"},
                         {"kind": "tool_result", "role": "tool", "text": "ECG normal"}]}
        final = {"diagnosis": ["follow-up required"]}
        report = {"diagnosisAccuracy": 1.0, "examinationPrecision": 0.5, "treatmentOverallScore": 0.9}
        await bridge.start()
        try:
            bridge.configure_training_batch(["p"], batch_index=1)
            await bridge.record_execution(patient_id="p", case_state=state, final_result=final)
            self.assertEqual(services.llm.calls, 0)
            await bridge.learn_case(patient_id="p", reflection={"summary": "check allergy history"},
                case_state=state, final_result=final, evaluation_report=report)
            self.assertTrue(await bridge.retrieve(patient_id="p", phase="diagnosis"))
            before = await bridge.state_fingerprint()
            async with bridge.core_memory.read_only():
                await bridge.retrieve(patient_id="heldout", phase="diagnosis", mode="test")
                await bridge.record_execution(patient_id="heldout", case_state=state, final_result=final)
            self.assertEqual(before, await bridge.state_fingerprint())
            await bridge.snapshot_to(path / "snapshot.json", learned_count=1)
        finally:
            await bridge.core_memory.reset()
            await bridge.close()

    async def test_graph_edges_tombstones_vectors_and_two_scopes_restore_exactly(self):
        from mematlas.methods.memos.core import MemOSCore
        path = Path(__file__).resolve().parents[2] / ".mematlas" / ("native-memos-graph-" + uuid4().hex)
        config = graph_config(path, "store")
        params = {"storage_path": str(path / "store"), "embedding_dims": 1024,
            "neo4j_uri": os.environ["MEMATLAS_TEST_NEO4J_URI"], "neo4j_user": "neo4j",
            "neo4j_password_env": "MEMATLAS_TEST_NEO4J_PASSWORD", "neo4j_database": "neo4j", "search_mode": "fast"}
        params.update(config.method.params)
        llm, embedder = ScriptedLLM(), ScriptedEmbedding(dimensions=1024)
        core = MemOSCore(params, MethodServices(llm=llm, embedder=embedder), "native-test")
        await core.setup()
        scopes = [{"user_id": "alice"}, {"user_id": "bob"}]
        try:
            for scope in scopes:
                await core.add([MemoryEvent("message", "penicillin allergy", role="user")], scope, "episode", {})
            from mematlas.core.errors import ConfigurationError
            incompatible = MemOSCore({**params, "embedding_dims": 8, "storage_path": str(path / "incompatible")},
                MethodServices(llm=llm, embedder=ScriptedEmbedding()), "incompatible-test")
            with self.assertRaisesRegex(ConfigurationError, "memory_vector_index has dimensions=1024"):
                await asyncio.to_thread(incompatible._verify_service)
            entry = await core._entry(scopes[0])
            graph = entry["tree"].graph_store

            def add_native_graph_properties():
                with graph.driver.session(database=graph.db_name) as session:
                    session.run("MATCH (a:Memory {user_name:$tenant}) WITH a LIMIT 1 "
                        "CREATE (b:Memory {id:$id, user_name:$tenant, memory:'deleted native evidence', "
                        "status:'deleted', delete_time:'2026-11-05T00:00:00Z', delete_record_id:'delete-1', "
                        "embedding:$vector}) CREATE (a)-[:RELATE_TO {weight:0.25, source:'native-test', "
                        "times:[1,2,3]}]->(b)", tenant=entry["tenant"], id=str(uuid4()), vector=[0.125] * 1024).consume()
            await asyncio.to_thread(add_native_graph_properties)
            await core.freeze()
            before = await core.export_state()
            self.assertEqual(len(before["scopes"]), 2)
            self.assertTrue(any(row["edges"] for row in before["scopes"]))
            self.assertTrue(any(node["props"].get("status") == "deleted"
                for row in before["scopes"] for node in row["nodes"]))
            calls = llm.calls
            await core.import_state(before)
            self.assertEqual(before, await core.export_state())
            self.assertEqual(calls, llm.calls)
            await core.thaw()
            await core.reset(scopes[0])
            self.assertTrue(await core.search("penicillin allergy", scopes[1], "default", 10))
        finally:
            for scope in scopes:
                await core.reset(scope)
            await core.close()

    async def test_efficiency_inventory_graph_and_external_neo4j_storage(self):
        from mematlas.integrations.openhospital.inspection import inspect_native
        path = Path(__file__).resolve().parents[2] / ".mematlas" / ("inventory-memos-" + uuid4().hex)
        services = MethodServices(llm=ScriptedLLM(), embedder=ScriptedEmbedding(dimensions=1024))
        async with Memory(graph_config(path, "store"), services=services) as memory:
            try:
                await memory.add(MemoryUnit(events=[MemoryEvent("message", "penicillin allergy", role="user")]))
                await memory.flush()
                await memory._freeze_backend()
                try:
                    before = services.llm.calls
                    result = await inspect_native(memory._method, {"method":"memos", "tokenizer_dir":str(path.parents[1]/"models/Qwen3-Embedding-0.6B"),
                        "neo4j_data_dir":str(path.parents[0]/"services/neo4j-community-5.26.0/data")})
                    self.assertGreater(result["memory_text_tokens"], 0)
                    self.assertGreater(result["storage_bytes"], 100000)
                    self.assertTrue(any("databases" in name for name in result["storage_files"]))
                    self.assertEqual(before, services.llm.calls)
                finally:
                    await memory._thaw_backend()
            finally:
                await memory.reset()

    async def test_public_native_graph_snapshot_clone_frozen_scope_and_usage(self):
        path = Path(__file__).resolve().parents[2] / ".mematlas" / ("native-memos-" + uuid4().hex)

        def make_config(label):
            return MemAtlasConfig(method=MemOSConfig(embedding_dims=1024, storage_path=str(path / label),
                neo4j_uri=os.environ["MEMATLAS_TEST_NEO4J_URI"],
                neo4j_user=os.environ.get("MEMATLAS_TEST_NEO4J_USER", "neo4j"),
                neo4j_password_env="MEMATLAS_TEST_NEO4J_PASSWORD",
                neo4j_database=os.environ.get("MEMATLAS_TEST_NEO4J_DATABASE", "neo4j")),
                runtime=RuntimeConfig(artifact_dir=str(path / (label + "-artifacts"))),
                scope={"user_id": "alice"})

        services = MethodServices(llm=ScriptedLLM(), embedder=ScriptedEmbedding(dimensions=1024))
        async with Memory(make_config("original"), services=services) as memory:
            try:
                self.assertEqual(len(await memory.index()), 4)
                before = await memory._capture_state()
                async with memory.read_only():
                    self.assertFalse((await memory.search("allergy")).items)
                    self.assertFalse((await memory.search("allergy", scope={"user_id": "bob"})).items)
                self.assertEqual(before, await memory._capture_state())
                events = [MemoryEvent("message", "I have penicillin allergy", role="user")]
                receipt = await memory.add(MemoryUnit(events=events))
                await memory.episode_end(receipt.episode_ref)
                await memory.add(MemoryUnit(type="experience", events=events,
                    feedback={"text": "Verify allergy history"}), episode_ref=receipt.episode_ref)
                await memory.flush()
                found = await memory.search("penicillin allergy")
                self.assertTrue(found.items)
                self.assertTrue(all("memory_id" in row for row in found.items))
                checkpoint = path / "checkpoint.json"
                await memory.save(checkpoint)
                before = await memory._capture_state()
                calls = services.llm.calls
                async with memory.read_only():
                    self.assertTrue((await memory.search("penicillin allergy")).items)
                await memory.load(checkpoint)
                self.assertEqual(calls, services.llm.calls)
                self.assertEqual(before, await memory._capture_state())
                async with Memory(make_config("clone"), services=MethodServices(
                        llm=ScriptedLLM(), embedder=ScriptedEmbedding(dimensions=1024))) as clone:
                    try:
                        await clone.load(checkpoint)
                        restored_results = (await clone.search("penicillin allergy")).items
                        # Upstream deduplicates identical text after vector
                        # ranking. Equal-score duplicate nodes have no stable
                        # tie-break in Neo4j, so either original node can win.
                        self.assertEqual([row["content"] for row in found.items], [row["content"] for row in restored_results])
                        from mematlas.core.state_files import unpack_state
                        state = unpack_state(before)
                        self.assertEqual(state, unpack_state(await clone._capture_state()))
                        nodes = {node["id"]:node for scope in state["native"]["scopes"] for node in scope["nodes"]}
                        self.assertTrue(all(row["memory_id"] in nodes for row in restored_results))
                        await clone.reset()
                        self.assertFalse((await clone.search("penicillin allergy")).items)
                        self.assertEqual(before, await memory._capture_state())
                    finally:
                        await clone.reset()
                usage = memory.stats()["usage"]
                self.assertGreater(usage["llm_calls"], 0)
                self.assertGreater(usage["embedding_calls"], 0)
            finally:
                await memory.reset()


if __name__ == "__main__":
    unittest.main()
