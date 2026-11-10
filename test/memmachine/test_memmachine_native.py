"""Native SQLite/USEARCH regressions. Run with the method's locked Python.

The scripted model only supplies deterministic inference responses; every memory
operation, schema parser, database and ANN index is the pinned upstream code.
"""
from __future__ import annotations
import asyncio
import hashlib
from importlib.util import find_spec
import json
from pathlib import Path
import sys
import unittest
from uuid import uuid4

from mematlas import Memory, MemAtlasConfig, MemoryEvent, MemoryUnit
from mematlas.core.config import RuntimeConfig
from mematlas.core.plugin import MethodServices
from mematlas.methods.memmachine import MemMachineConfig


class ScriptedLLM:
    def __init__(self, call_log=None):
        self.calls = 0
        self.call_log = call_log

    def construction_spec(self):
        return {"factory": "mematlas.test.memmachine.test_memmachine_native:ScriptedLLM", "kwargs": {"call_log": self.call_log}}

    async def complete(self, messages, **kwargs):
        self.calls += 1
        if self.call_log:
            with Path(self.call_log).open("a", encoding="utf-8") as log:
                log.write("llm\n")
        schema = kwargs.get("response_format", {}).get("json_schema", {}).get("schema", {})
        if "commands" in schema.get("properties", {}):
            return json.dumps({"commands": [{"command": "add", "feature": "allergy",
                "tag": "Health & Wellness", "value": "penicillin allergy"}]})
        return "Patient has penicillin allergy."


class ScriptedEmbedding:
    model = "deterministic-native-test-8d"

    def __init__(self, call_log=None):
        self.call_log = call_log

    def construction_spec(self):
        return {"factory": "mematlas.test.memmachine.test_memmachine_native:ScriptedEmbedding", "kwargs": {"call_log": self.call_log}}

    async def embed(self, texts):
        if self.call_log:
            with Path(self.call_log).open("a", encoding="utf-8") as log:
                log.write("embedding\n")
        return [[float(value + 1) for value in hashlib.sha256(text.encode()).digest()[:8]] for text in texts]


@unittest.skipUnless(find_spec("memmachine_server"), "Run in the method's requirements.lock environment")
class NativeMemMachineTests(unittest.IsolatedAsyncioTestCase):
    async def test_official_command_envelope_compatibility_retains_features(self):
        from mematlas.methods.memmachine.core import _providers
        from memmachine_server.semantic_memory.semantic_llm import _SemanticFeatureUpdateRes
        class RawLLM:
            text = "[]"
            calls = 0
            repair = None
            async def complete(self, messages, **kwargs):
                self.calls += 1
                return self.repair if self.repair and len(messages)>2 else self.text
        llm = RawLLM()
        model, _, client = _providers(MethodServices(llm=llm), 8)
        command = {"command":"add", "feature":"allergy", "tag":"Health & Wellness", "value":"penicillin"}
        try:
            for raw, expected in [( [],0), (command,1), ([command],1), ({"commands":[command]},1)]:
                llm.text = json.dumps(raw)
                result = await model.generate_parsed_response(output_format=_SemanticFeatureUpdateRes, user_prompt="native history")
                self.assertEqual(len(result.commands), expected)
                if expected:
                    self.assertEqual(result.commands[0].value, "penicillin")
            llm.text = '[{"command":"invented-invalid","feature":"allergy"}]'
            before = llm.calls
            with self.assertRaises(ValueError):
                await model.generate_parsed_response(output_format=_SemanticFeatureUpdateRes, user_prompt="native history")
            self.assertEqual(llm.calls-before,4)
            llm.text = '[{"command":"add","feature":"allergy","tag":"Health & Wellness"}]'
            llm.repair = json.dumps({"commands":[command]})
            before = llm.calls
            repaired = await model.generate_parsed_response(output_format=_SemanticFeatureUpdateRes, user_prompt="native history")
            self.assertEqual(llm.calls-before,2)
            self.assertEqual(repaired.commands[0].value,"penicillin")
        finally:
            await client.close()

    async def test_omitted_storage_path_uses_isolated_runtime_workspace(self):
        path = Path(__file__).resolve().parents[2] / ".mematlas" / ("native-memmachine-default-" + uuid4().hex)
        config = MemAtlasConfig(method=MemMachineConfig(embedding_dims=8),
            runtime=RuntimeConfig(artifact_dir=str(path)), scope={"user_id": "alice"})
        locations = []
        for _ in range(2):
            llm = ScriptedLLM()
            async with Memory(config, services=MethodServices(llm=llm, embedder=ScriptedEmbedding())) as memory:
                locations.append(memory._method.backend.path)
                self.assertTrue(locations[-1].is_relative_to(Path(memory.config.runtime.artifact_dir)))
                self.assertEqual(llm.calls, 0)
        self.assertNotEqual(*locations)

    def assert_native_streams_equal(self, expected, actual):
        # Native semantic search merges project/user async iterators as ready;
        # compare each independent stream's exact ranking, not race ordering.
        def streams(rows):
            groups = {}
            for row in rows:
                key = ("semantic", row["metadata"]["set_id"]) if row["memory_type"] == "semantic_feature" else ("episodic",)
                groups.setdefault(key, []).append(row)
            return groups
        self.assertEqual(streams(expected), streams(actual))

    async def test_native_semantic_provider_failure_marks_backend_uncertain(self):
        from mematlas.core.errors import MemoryStateError, MethodOperationError

        class FailingLLM(ScriptedLLM):
            def __init__(self):
                super().__init__()
                self.fail = False
                self.failures = 0
            async def complete(self, messages, **kwargs):
                if self.fail:
                    self.failures += 1
                    raise RuntimeError("controlled native semantic provider failure")
                return await super().complete(messages, **kwargs)

        path = Path(__file__).resolve().parents[2] / ".mematlas" / ("native-memmachine-failure-" + uuid4().hex)
        llm = FailingLLM()
        config = MemAtlasConfig(method=MemMachineConfig(embedding_dims=8, storage_path=str(path / "store"),
                flush_timeout_s=3), runtime=RuntimeConfig(artifact_dir=str(path / "artifacts")),
            scope={"user_id": "alice"})
        memory = Memory(config, services=MethodServices(llm=llm, embedder=ScriptedEmbedding()))
        await memory.__aenter__()
        try:
            await memory.add(MemoryUnit(events=[MemoryEvent("message", "penicillin allergy", role="user")]))
            llm.fail = True
            try:
                with self.assertRaises(MethodOperationError):
                    await memory.flush()
                self.assertGreater(llm.failures, 0)
                self.assertEqual(memory.backend_status["state"], "uncertain")
                self.assertFalse(memory.backend_status["can_continue"])
                with self.assertRaises(MemoryStateError):
                    await memory.add(MemoryUnit(task="another encounter"))
            finally:
                # Allow a current native retry to settle before closing resources;
                # the public failed instance remains unusable.
                llm.fail = False
        finally:
            backend = memory._method.backend
            with self.assertRaises(MethodOperationError):
                await memory.__aexit__(None, None, None)
            self.assertTrue(backend._client.is_closed)
            self.assertFalse(backend.resources.database_manager.vector_stores)

    async def test_checkpoint_restores_in_new_native_subprocess_without_inference(self):
        path = Path(__file__).resolve().parents[2] / ".mematlas" / ("native-memmachine-worker-" + uuid4().hex)
        path.mkdir(parents=True)
        call_log = path / "provider-calls.log"

        def config(label):
            return MemAtlasConfig(method=MemMachineConfig(embedding_dims=8, storage_path=str(path / label)),
                runtime=RuntimeConfig(executor="subprocess", python_executable=sys.executable,
                    artifact_dir=str(path / (label + "-artifacts"))), scope={"user_id": "alice"})

        def services():
            return MethodServices(llm=ScriptedLLM(str(call_log)), embedder=ScriptedEmbedding(str(call_log)))

        checkpoint = path / "checkpoint.json"
        async with Memory(config("original"), services=services()) as original:
            await original.add(MemoryUnit(events=[MemoryEvent("message", "penicillin allergy", role="user")]))
            await original.flush()
            expected = await original.search("allergy")
            await original.save(checkpoint)
        async with Memory(config("restored"), services=services()) as restored:
            calls_before_load = call_log.read_text(encoding="utf-8")
            await restored.load(checkpoint)
            after = restored.stats()["usage"]
            self.assertEqual(calls_before_load, call_log.read_text(encoding="utf-8"))
            self.assert_native_streams_equal(expected.items, (await restored.search("allergy")).items)
            self.assertEqual(calls_before_load.count("llm\n"), call_log.read_text(encoding="utf-8").count("llm\n"))
            self.assertEqual(after.get("llm_calls", 0), restored.stats()["usage"].get("llm_calls", 0))
            frozen_state = await restored._capture_state()
            async with restored.read_only():
                self.assert_native_streams_equal(expected.items, (await restored.search("allergy")).items)
            self.assertEqual(frozen_state, await restored._capture_state())
            await restored.add(MemoryUnit(events=[MemoryEvent("message", "verify allergy history", role="user")]))
            await restored.flush()
            self.assertGreater(restored.stats()["usage"]["llm_calls"], after.get("llm_calls", 0))
            self.assertGreater(len((await restored.search("allergy")).items), len(expected.items))

    async def test_efficiency_inventory_native_sql_ann_stm_no_model(self):
        from mematlas.integrations.openhospital.inspection import inspect_native
        path = Path(__file__).resolve().parents[2] / ".mematlas" / ("inventory-mm-" + uuid4().hex)
        services = MethodServices(llm=ScriptedLLM(), embedder=ScriptedEmbedding())
        config = MemAtlasConfig(method=MemMachineConfig(embedding_dims=8, storage_path=str(path / "store")),
            runtime=RuntimeConfig(artifact_dir=str(path / "artifacts")), scope={"user_id":"alice"})
        async with Memory(config, services=services) as memory:
            await memory.add(MemoryUnit(events=[MemoryEvent("message", "penicillin allergy", role="user")]))
            await memory.flush()
            await memory._freeze_backend()
            try:
                before = services.llm.calls
                result = await inspect_native(memory._method, {"method":"memmachine", "tokenizer_dir":str(path.parents[1]/"models/Qwen3-Embedding-0.6B")})
                self.assertGreater(result["memory_text_tokens"], 0)
                self.assertGreater(result["storage_bytes"], 0)
                self.assertEqual(before, services.llm.calls)
                self.assertIn("stm_episodes", result["text_layers"])
            finally:
                await memory._thaw_backend()

    async def test_public_boundary_roundtrip_read_only_usage_and_scope(self):
        path = Path(__file__).resolve().parents[2] / ".mematlas" / ("native-memmachine-" + uuid4().hex)
        services = MethodServices(llm=ScriptedLLM(), embedder=ScriptedEmbedding())
        config = MemAtlasConfig(method=MemMachineConfig(embedding_dims=8, storage_path=str(path / "store")),
            runtime=RuntimeConfig(artifact_dir=str(path / "artifacts")), scope={"user_id": "alice"})
        async with Memory(config, services=services) as memory:
            self.assertEqual(len(await memory.index()), 3)
            initial = await memory._capture_state()
            async with memory.read_only():
                self.assertFalse((await memory.search("allergy")).items)
                self.assertFalse((await memory.search("allergy", scope={"user_id": "bob"})).items)
            self.assertEqual(initial, await memory._capture_state())
            self.assertEqual(services.llm.calls, 0)
            events = [MemoryEvent("message", "I have penicillin allergy", role="user")]
            receipt = await memory.add(MemoryUnit(events=events))
            await memory.episode_end(receipt.episode_ref)
            await memory.add(MemoryUnit(type="experience", events=events, feedback={"success": True}),
                episode_ref=receipt.episode_ref)
            await memory.flush()
            found = await memory.search("allergy")
            self.assertTrue(any(row["memory_type"] == "semantic_feature" for row in found.items))
            self.assertTrue(all("memory_id" in row for row in found.items))
            self.assertEqual(len([row for row in found.items if row["memory_type"] == "short_term_episode"]), 2)
            # Event plus its separately attributed real feedback, no event duplication.
            checkpoint = path / "checkpoint.json"
            await memory.save(checkpoint)
            calls = services.llm.calls
            async with memory.read_only():
                readonly = await memory.search("allergy")
                self.assertTrue(readonly.items)
            self.assertEqual(calls, services.llm.calls)
            await memory.load(checkpoint)
            after = await memory.search("allergy")
            self.assert_native_streams_equal(found.items, after.items)
            empty = await memory.search("allergy", scope={"user_id": "bob"})
            self.assertFalse(empty.items)
            stats = memory.stats()["usage"]
            self.assertGreater(stats["llm_calls"], 0)
            self.assertGreater(stats["embedding_calls"], 0)
            await memory.reset()
            self.assertFalse((await memory.search("allergy")).items)

    async def test_openhospital_execution_then_delayed_experience(self):
        from mematlas.integrations.openhospital import OpenHospitalMemoryIntegration
        path = Path(__file__).resolve().parents[2] / ".mematlas" / ("native-memmachine-hospital-" + uuid4().hex)
        config = MemAtlasConfig(method=MemMachineConfig(embedding_dims=8, storage_path=str(path / "store")),
            runtime=RuntimeConfig(artifact_dir=str(path / "artifacts")),
            scope={"user_id": "openhospital-test", "agent_id": "doctor"})
        bridge = OpenHospitalMemoryIntegration.from_config(config,
            services=MethodServices(llm=ScriptedLLM(), embedder=ScriptedEmbedding()),
            journal_dir=path / "encounters")
        state = {"chat_history": [{"from": "patient", "text": "penicillin allergy"}],
            "timeline": [{"kind": "message", "role": "user", "text": "penicillin allergy"},
                         {"kind": "tool_result", "role": "tool", "text": "ECG normal"}]}
        final = {"diagnosis": ["follow-up required"]}
        report = {"diagnosisAccuracy": 1.0, "examinationPrecision": 0.5, "treatmentOverallScore": 0.9}
        await bridge.start()
        try:
            bridge.configure_training_batch(["p"], batch_index=1)
            await bridge.record_execution(patient_id="p", case_state=state, final_result=final)
            self.assertEqual(bridge.history.result("batch:1:patient:p", scope=bridge.scope).execution, "completed")
            await bridge.learn_case(patient_id="p", reflection={"summary": "check allergy history"},
                case_state=state, final_result=final, evaluation_report=report)
            await bridge.core_memory.flush()
            contexts = await bridge.retrieve(patient_id="p", phase="diagnosis")
            self.assertTrue(contexts)
            before = await bridge.state_fingerprint()
            async with bridge.core_memory.read_only():
                await bridge.retrieve(patient_id="heldout", phase="diagnosis", mode="test")
                await bridge.record_execution(patient_id="heldout", case_state=state, final_result=final)
            self.assertEqual(before, await bridge.state_fingerprint())
            await bridge.snapshot_to(path / "snapshot.json", learned_count=1)
        finally:
            await bridge.close()

    async def test_freeze_preserves_uningested_semantic_history(self):
        from mematlas.methods.memmachine.core import MemMachineCore
        path = Path(__file__).resolve().parents[2] / ".mematlas" / ("native-memmachine-pending-" + uuid4().hex)
        llm = ScriptedLLM()
        core = MemMachineCore({"storage_path": str(path), "embedding_dims": 8, "flush_timeout_s": 20},
            MethodServices(llm=llm, embedder=ScriptedEmbedding()), "native-test")
        await core.setup()
        try:
            scope = {"user_id": "alice"}
            await core.add([MemoryEvent("message", "penicillin allergy", role="user", scope=scope)],
                scope, "episode", {})
            await core.freeze()
            self.assertEqual(llm.calls, 0)
            state = await core.export_state()
            self.assertEqual(state, await core.export_state())
            await core.import_state(state)
            self.assertTrue(core._frozen)
            self.assertEqual(llm.calls, 0)
            self.assertTrue(await core.search("allergy", scope, "episodic", None))
            self.assertEqual(llm.calls, 0)
            self.assertEqual(state, await core.export_state())
            await core.thaw()
            await core.flush()
            self.assertGreater(llm.calls, 0)
        finally:
            await core.close()

    async def test_stm_freeze_pauses_between_native_batches(self):
        from datetime import datetime, timezone
        from types import SimpleNamespace
        from memmachine_server.common.episode_store import Episode
        from memmachine_server.common.language_model import LanguageModel
        from memmachine_server.episodic_memory.short_term_memory.short_term_memory import (
            ShortTermMemory, ShortTermMemoryParams)
        from mematlas.methods.memmachine.core import _SummaryGate

        class SlowModel(LanguageModel):
            def __init__(self):
                self.calls = 0
                self.started, self.release = asyncio.Event(), asyncio.Event()
            async def generate_parsed_response(self, *args, **kwargs):
                raise AssertionError("STM uses native unstructured summarization")
            async def generate_response_with_token_usage(self, *args, **kwargs):
                raise AssertionError("STM does not request synthetic usage")
            async def generate_response(self, *args, **kwargs):
                self.calls += 1
                self.started.set()
                await self.release.wait()
                return "summary", []

        model, owner = SlowModel(), SimpleNamespace(_frozen=False)
        memory = ShortTermMemory(ShortTermMemoryParams(session_key="native-stm-test", llm_model=model,
            message_capacity=5, summary_prompt_system="Summarize", summary_prompt_user="{episodes} {summary} {max_length}"))
        gate = _SummaryGate(owner, memory)
        episodes = [Episode(uid=uuid4(), content="a long episode", session_key="native-stm-test",
            created_at=datetime.now(timezone.utc), producer_id="alice", producer_role="user") for _ in range(2)]
        await memory.add_episodes([episodes[0]])
        await model.started.wait()
        await memory.add_episodes([episodes[1]])
        owner._frozen = True
        freezing = asyncio.create_task(gate.pause())
        model.release.set()
        await asyncio.wait_for(freezing, 3)
        self.assertEqual(model.calls, 1)
        self.assertEqual([row.uid for row in gate.blocked], [episodes[1].uid])
        await memory._consolidator.wait_until_done()
        self.assertEqual(model.calls, 1)
        owner._frozen = False
        gate.resume()
        await memory._consolidator.wait_until_done()
        self.assertEqual(model.calls, 2)
        await memory.close()


if __name__ == "__main__":
    unittest.main()
