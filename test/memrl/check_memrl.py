"""Reproducible offline native MemRL regression suite.

Run from the repository root: .venv/bin/python test/memrl/check_memrl.py
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from unittest.mock import patch
from pathlib import Path
from typing import Any

from fixtures import (
    NativeHarness,
    REPOSITORY_ROOT,
    UPSTREAM_ROOT,
    TextualMemoryItem,
    directory_digest,
    episode,
    setup,
    snapshot_path,
)
from mematlas.core.config import MemAtlasConfig, MethodConfig, RuntimeConfig
from mematlas.core.errors import CapabilityMismatchError, ConfigurationError, MethodOperationError, MemoryStateError
from mematlas.core.plugin import MethodServices
from mematlas.core.provider_config import resolve_providers
from mematlas.core.registry import default_registry
from mematlas.methods.memrl.method import PLUGIN, MemRLMemory, _filesystem_path
from mematlas.methods.memrl.core import _provider_adapters
from mematlas.methods.memrl import AddOptions, MemRLConfig, SearchOptions
from mematlas.runtime.memory import Memory
from mematlas.core.models import MemoryContribution, MemoryEvent, MemoryEpisode
from mematlas.runtime.workspace import allocate_workspace


from mematlas.evaluation.validation import submit_episode
from mematlas.test.evaluation.support import add_event


async def submit_learning(target, value, **options):
    if isinstance(target, Memory):
        return await submit_episode(target, value, **options)
    return await target._learn(value, **options)


class NativeMemRLTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="memrl-native-tests-")
        self.root = Path(self.temporary.name)
        self.harness = NativeHarness(self.root).__enter__()
        self.adapters = []
        self.scope = {"user_id": "offline-user"}

    async def asyncTearDown(self) -> None:
        for adapter in self.adapters:
            await adapter.close()
        self.harness.__exit__(None, None, None)
        # Native snapshot files exceed MAX_PATH on Windows. This directory was
        # allocated by this test; verify its parent before recursive cleanup.
        if os.name == "nt":
            if self.root.resolve().parent != Path(tempfile.gettempdir()).resolve():
                raise ValueError("Unexpected test temporary directory")
            self.temporary.name = str(_filesystem_path(self.root))
        self.temporary.cleanup()

    async def adapter(self, **params: Any) -> Any:
        adapter = self.harness.adapter(**params)
        self.adapters.append(adapter)
        await setup(adapter)
        return adapter

    async def test_builtin_discovery_and_missing_source_preflight_without_native_import(self) -> None:
        self.assertEqual(default_registry().get("memrl").descriptor.to_dict(), PLUGIN.descriptor.to_dict())
        missing_source = self.root / "absent-upstream"
        script = """
import json, sys
sys.path.insert(0, sys.argv[1])
from mematlas.core.config import MethodConfig
from mematlas.core.plugin import MethodServices
from mematlas.methods.memrl.method import PLUGIN
report = PLUGIN.healthcheck(MethodConfig('memrl'), MethodServices(
    llm=object(), embedder=object(), source_roots={'memrl': sys.argv[2]}))
print(json.dumps({'report': report, 'native_imported': any(
    name == 'memrl' or name.startswith('memrl.') for name in sys.modules)}))
"""
        environment = {**os.environ, "MEMOS_BASE_PATH": str(self.root / "preflight-import"),
                       "PYTHON_DOTENV_DISABLED": "1"}
        process = subprocess.run([sys.executable, "-c", script, str(REPOSITORY_ROOT.parent), str(missing_source)],
                                 capture_output=True, text=True, check=False, env=environment, timeout=30)
        self.assertEqual(process.returncode, 0, process.stderr)
        result = json.loads(process.stdout)
        self.assertFalse(result["native_imported"])
        self.assertFalse(result["report"]["ready"])
        self.assertFalse(result["report"]["source_ready"])
        self.assertEqual(result["report"]["source_root"], str(missing_source))
        self.assertIn("memrl/service/memory_service.py", result["report"]["missing_source_files"])

    async def test_public_provider_configuration_overrides_profile_transport_defaults(self) -> None:
        providers = {
            "llm": {"model": "custom-llm", "base_url": "http://127.0.0.1:1/v1"},
            "embedding": {"model": "custom-embedding", "base_url": "http://127.0.0.1:1/v1", "dimensions": 2},
        }
        services = MethodServices(llm=object(), embedder=object(), provider_configs=providers,
                                  artifact_dir=self.root / "public-providers")
        adapter = MemRLMemory(MethodConfig("memrl", {"profile": "hle"}), services)
        self.adapters.append(adapter)
        resolved = resolve_providers(adapter.settings.values, providers)
        self.assertEqual(resolved["llm"].model, "custom-llm")
        self.assertEqual(resolved["embedding"].model, "custom-embedding")
        self.assertEqual(adapter.settings.values["embedding_dims"], 2)
        report = PLUGIN.healthcheck(MethodConfig("memrl", {"profile": "hle"}), services)
        self.assertEqual(report["config_errors"], [])
        # Explicit conflicting values must still use the framework's error.
        self.assertTrue(PLUGIN.validate_config(MethodConfig("memrl", {"llm_model": "old-alias"})))

    async def test_runtime_default_provider_objects_allow_native_learning_and_frozen_retrieval(self) -> None:
        class AsyncProviders:
            async def complete(self, messages: Any, **kwargs: Any) -> str:
                return "Offline provider response."

            async def embed(self, texts: list[str]) -> list[list[float]]:
                return [[1.0, 0.0] for _ in texts]

        providers = AsyncProviders()
        memory = Memory(MemAtlasConfig(
            MethodConfig("memrl", {"batch_size": 1}),
            RuntimeConfig(artifact_dir=str(self.root / "facade-workspace")),
        ), services=MethodServices(llm=providers, embedder=providers,
                                   objects={"memrl.service_factory": self.harness.factory}))
        try:
            # The public runtime resolves omitted models into ProviderConfig
            # objects containing model=None; profile defaults must survive this.
            self.assertIsNone(memory.services.provider_configs["llm"].model)
            self.assertIsNone(memory.services.provider_configs["embedding"].model)
            report = PLUGIN.healthcheck(memory.config.method, memory.services)
            self.assertEqual(report["config_errors"], [])
            self.assertTrue(report["source_ready"])
            await memory._setup()
            self.assertIs(memory._method.services, memory.services)
            # Runtime provider instrumentation must remain visible to the
            # adapter, including injected providers without their own ledger.
            await memory._method.services.llm.complete([])
            await memory._method.services.embedder.embed(["metering probe"])
            self.assertEqual({record.resource_kind for record in memory.services.usage.all()},
                             {"llm", "embedding"})
            self.assertEqual(memory._method.settings.profile, "default")
            self.assertIsNotNone(memory._method.services.provider_configs["llm"].model)
            self.assertIsNotNone(memory._method.services.provider_configs["embedding"].model)
            sample = episode(self.scope, "facade-task")
            await submit_learning(memory, sample)
            state = await memory._capture_state()
            async with memory.read_only():
                for _ in range(2):
                    contribution = await memory.search("facade-task", scope=self.scope)
                    self.assertEqual(len(contribution.items), 1)
            self.assertEqual(await memory._capture_state(), state)
        finally:
            await memory._close()

    async def test_vendored_source_inventory_detects_tampering_missing_files_and_wrong_revision(self) -> None:
        def preflight(root: Path) -> dict[str, Any]:
            return PLUGIN.healthcheck(MethodConfig("memrl"), MethodServices(
                llm=object(), embedder=object(), source_roots={"memrl": str(root)}))

        intact = preflight(UPSTREAM_ROOT)
        self.assertTrue(intact["source_ready"])
        self.assertEqual(intact["source_provenance"], "vendored")
        self.assertTrue(intact["source_file_hashes_verified"])
        self.assertTrue(intact["source_revision_verified"])
        for mutation in ("tamper", "missing", "revision"):
            with self.subTest(mutation=mutation):
                source = self.root / mutation
                shutil.copytree(UPSTREAM_ROOT, source, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
                if mutation == "tamper":
                    path = source / "memrl" / "service" / "memory_service.py"
                    path.write_text(path.read_text(encoding="utf-8") + "\n# fixture tamper\n", encoding="utf-8")
                elif mutation == "missing":
                    (source / "memrl" / "configs" / "config.py").unlink()
                else:
                    path = source / "UPSTREAM.json"
                    metadata = json.loads(path.read_text(encoding="utf-8"))
                    metadata["revision"] = "0" * 40
                    path.write_text(json.dumps(metadata), encoding="utf-8")
                report = preflight(source)
                self.assertFalse(report["ready"])
                self.assertFalse(report["source_ready"])
                self.assertTrue(report["source_errors"])
                if mutation == "revision":
                    self.assertFalse(report["source_revision_matches"])
                    self.assertFalse(report["source_revision_verified"])
                else:
                    self.assertFalse(report["source_file_hashes_verified"])
                    self.assertTrue(report["source_hash_errors"])

    async def test_fresh_workspace_preserves_mos_configuration_input_path(self) -> None:
        source_path = self.root / "mos-input.json"
        source_text = '{"offline_fixture": true}\n'
        source_path.write_text(source_text, encoding="utf-8")
        requested = MemAtlasConfig(
            MethodConfig("memrl", {"mos_config_path": str(source_path)}),
            RuntimeConfig(artifact_dir=str(self.root / "workspace")),
            metadata={"run_id": "preserve-mos-input"},
        )
        allocated, run_id = allocate_workspace(requested, PLUGIN)
        self.assertTrue(run_id)
        self.assertEqual(allocated.method.params["mos_config_path"], str(source_path))
        self.assertEqual(source_path.read_text(encoding="utf-8"), source_text)
        self.assertTrue((Path(allocated.runtime.artifact_dir) / "memory-run.json").is_file())

    async def test_native_add_retrieve_update_and_log_namespaces(self) -> None:
        with self.assertLogs(level=logging.INFO) as logs:
            adapter = await self.adapter(profile="bcb", batch_size=1, epsilon=0.0)
            await submit_learning(adapter, episode(self.scope, "seed"))
            contribution = await adapter.search("seed", top_k=5, scope=self.scope, extensions={})
            self.assertEqual(len(contribution.items), 1)
            applied_memory = [contribution]
            await submit_learning(adapter, episode(self.scope, "failure", success=False, applied_memory=applied_memory))
        native = self.harness.latest(self.scope)
        memory_id = contribution.items[0]["memory_id"]
        self.assertAlmostEqual(native.text_memory().get(memory_id).metadata.model_extra["q_value"], -0.3)
        self.assertEqual(native.text_memory().get(memory_id).metadata.model_extra["q_visits"], 1)
        self.assertTrue(any(record.name == "mematlas.methods.memrl.adapter" for record in logs.records))
        self.assertTrue(any(record.name.startswith("memrl.") for record in logs.records))

    def test_provider_defaults_and_explicit_options_match_native_llm(self) -> None:
        calls = []

        class Provider:
            async def complete(self, messages, **kwargs):
                calls.append(kwargs)
                return "Recorded response."

        services = MethodServices(llm=Provider(), artifact_dir=self.root / "providers")
        llm, _ = _provider_adapters(UPSTREAM_ROOT, services,
                                   {"llm_temperature": 0.6, "llm_max_tokens": 4096})
        llm.generate([])
        llm.generate([], temperature=0.0, max_tokens=17)
        llm.generate([], max_completion_tokens=23)
        llm.generate([], max_tokens=None)
        self.assertEqual(calls, [
            {"temperature": 0.6, "max_tokens": 4096},
            {"temperature": 0.0, "max_tokens": 17},
            {"temperature": 0.6, "max_tokens": 23, "max_completion_tokens": 23},
            {"temperature": 0.6, "max_tokens": None},
        ])
        unbounded, _ = _provider_adapters(UPSTREAM_ROOT, services, {"llm_max_tokens": None})
        unbounded.generate([])
        self.assertEqual(calls[-1], {"temperature": 0.7})

    async def test_native_failure_reflection_preserves_configured_generation_budget(self) -> None:
        calls = []

        class Provider:
            async def complete(self, messages, **kwargs):
                calls.append((messages, kwargs))
                return "Check the failed step before retrying."

        adapter = await self.adapter(batch_size=1, llm_max_tokens=4096)
        await adapter.search("failed-task", top_k=3, scope=self.scope, extensions={})
        native = self.harness.latest(self.scope)
        native.llm_provider, _ = _provider_adapters(
            UPSTREAM_ROOT, MethodServices(llm=Provider(), artifact_dir=self.root / "reflection"),
            adapter.settings.values)
        await submit_learning(adapter, episode(self.scope, "failed-task", success=False))
        self.assertEqual(len(calls), 1)
        self.assertIn("This task failed", calls[0][0][0]["content"])
        self.assertEqual(calls[0][1], {"temperature": 0.3, "max_tokens": 4096})
        stored = native.text_memory().get_all()[0]
        self.assertEqual(stored.metadata.model_dump()["type"], "adjustment")
        self.assertIn("Check the failed step before retrying.", stored.metadata.model_extra["full_content"])

    async def test_hle_profiles_initialize_success_and_failure_like_native_runner(self) -> None:
        for profile in ("default", "hle", "hle_episode"):
            with self.subTest(profile=profile):
                adapter = await self.adapter(profile=profile, batch_size=1)
                await submit_learning(adapter, episode(self.scope, f"{profile}-success"))
                await submit_learning(adapter, episode(self.scope, f"{profile}-failure", success=False))
                rows = self.harness.latest(self.scope).text_memory().get_all()
                by_task = {item.memory: item.metadata.model_extra["q_value"] for item in rows}
                self.assertEqual(by_task[f"{profile}-success"], 1.0)
                self.assertEqual(by_task[f"{profile}-failure"], 0.0)

    async def test_profile_specific_native_update_order(self) -> None:
        for profile in ("hle", "bcb", "alfworld", "llb", "llb_os", "llb_db"):
            with self.subTest(profile=profile):
                adapter = await self.adapter(profile=profile, batch_size=1)
                await submit_learning(adapter, episode(self.scope, f"{profile}-success"))
                names = [name for name, _ in self.harness.latest(self.scope).calls]
                expected = ["add_memories", "update_values"] if profile == "hle" else ["update_values", "add_memories"]
                self.assertEqual(names, expected)

    async def test_bcb_inplace_preserves_native_order_observable_q_values(self) -> None:
        adapter = await self.adapter(profile="bcb", batch_size=1, adjustment_mode="inplace")
        await adapter.search("seed", top_k=5, scope=self.scope, extensions={})
        native = self.harness.latest(self.scope)
        seed = TextualMemoryItem(memory="seed", metadata={"q_value": 0.8, "q_visits": 4,
                                                         "reward_ma": 0.2, "full_content": "Original procedure."})
        native.text_memory().add([seed])
        native.dict_memory["seed"] = [seed.id]
        native.query_embeddings["seed"] = native.embedding_provider.embed(["seed"])[0]
        native.calls.clear()
        await submit_learning(adapter, episode(self.scope, "failed-task", success=False,
                                    applied_memory=[await adapter.search("seed", top_k=5, scope=self.scope, extensions={})]))
        metadata = native.text_memory().get(seed.id).metadata.model_extra
        self.assertEqual(native.calls, [("update_values", 1), ("add_memories", 1)])
        # These seemingly inconsistent native results are intentional fidelity
        # evidence: inplace metadata overwrites persisted Q after its RL update.
        self.assertEqual(metadata["q_value"], 0.0)
        self.assertEqual(metadata["q_visits"], 0)
        self.assertAlmostEqual(native._q_cache[seed.id], 0.26)

    async def test_unconfirmed_retrieval_does_not_reuse_query_candidates(self) -> None:
        adapter = await self.adapter(profile="bcb", batch_size=1, epsilon=0.0)
        await submit_learning(adapter, episode(self.scope, "previous-task"))
        contribution = await adapter.search("previous-task", top_k=5, scope=self.scope, extensions={})
        self.assertTrue(contribution.extensions["memrl"]["query_candidates"])
        native = self.harness.latest(self.scope)
        previous_ids = list(native.dict_memory["previous-task"])
        await submit_learning(adapter, episode(self.scope, "unconfirmed-task", applied_memory=()))
        self.assertEqual(native.dict_memory["previous-task"], previous_ids)
        self.assertEqual(len(native.dict_memory["unconfirmed-task"]), 1)
        self.assertNotEqual(native.dict_memory["unconfirmed-task"], previous_ids)

    async def test_automatic_evolution_completes_only_the_scope_whose_batch_is_full(self) -> None:
        adapter = await self.adapter(profile="bcb", batch_size=3)
        other = {"user_id": "other-user"}
        await submit_learning(adapter, episode(self.scope, "first"))
        await submit_learning(adapter, episode(self.scope, "second"))
        await submit_learning(adapter, episode(other, "other"))
        self.assertEqual(self.harness.latest(self.scope).calls, [])
        self.assertEqual(self.harness.latest(other).calls, [])
        await submit_learning(adapter, episode(self.scope, "third"))
        self.assertEqual(self.harness.latest(self.scope).calls, [("update_values", 3), ("add_memories", 3)])
        self.assertEqual(self.harness.latest(other).calls, [])
        await submit_learning(adapter, episode(other, "other-second"))
        await submit_learning(adapter, episode(other, "other-third"))
        self.assertEqual(self.harness.latest(other).calls, [("update_values", 3), ("add_memories", 3)])
        self.assertFalse(adapter._pending_batch)

    async def test_default_automatic_evolution_waits_for_full_batches(self) -> None:
        adapter = await self.adapter()
        await submit_learning(adapter, episode(self.scope, "task-0"))
        native = self.harness.latest(self.scope)
        await adapter.flush()
        self.assertEqual(native.calls, [])
        self.assertEqual(native.text_memory().get_all(), [])
        for index in range(1, 32):
            await submit_learning(adapter, episode(self.scope, f"task-{index}"))
        self.assertEqual(native.calls, [("add_memories", 32), ("update_values", 32)])
        self.assertEqual(len(native.text_memory().get_all()), 32)
        await submit_learning(adapter, episode(self.scope, "tail"))
        await adapter.flush()
        self.assertEqual(len(native.text_memory().get_all()), 32)
        for index in range(31):
            await submit_learning(adapter, episode(self.scope, f"tail-{index}"))
        self.assertEqual(native.calls[-2:], [("add_memories", 32), ("update_values", 32)])
        self.assertEqual(len(native.text_memory().get_all()), 64)

    async def test_automatic_evolution_does_not_reprocess_experiences(self):
        adapter = await self.adapter(batch_size=2)
        with patch.object(adapter, "_evolve_batch", wraps=adapter._evolve_batch) as evolve:
            await submit_learning(adapter, episode(self.scope, "first"))
            self.assertEqual(evolve.await_count, 0)
            await submit_learning(adapter, episode(self.scope, "second"))
            self.assertEqual(evolve.await_count, 1)
            native = self.harness.latest(self.scope)
            self.assertEqual(native.calls, [("add_memories", 2), ("update_values", 2)])
            await submit_learning(adapter, episode(self.scope, "third"))
            self.assertEqual(evolve.await_count, 1)
            self.assertEqual(native.calls, [("add_memories", 2), ("update_values", 2)])
            await submit_learning(adapter, episode(self.scope, "fourth"))
            self.assertEqual(evolve.await_count, 2)
            self.assertEqual(len(native.text_memory().get_all()), 4)

    async def test_pending_batch_survives_snapshot_without_implicit_learning(self) -> None:
        adapter = await self.adapter(profile="bcb", batch_size=2)
        await submit_learning(adapter, episode(self.scope, "queued-one"))
        state = await adapter.save()
        await submit_learning(adapter, episode(self.scope, "discarded-after-checkpoint"))
        await adapter.load(state)
        self.assertEqual(self.harness.latest(self.scope).text_memory().get_all(), [])
        self.assertEqual(await adapter.save(), state)
        await submit_learning(adapter, episode(self.scope, "completes-batch"))
        rows = self.harness.latest(self.scope).text_memory().get_all()
        self.assertEqual([item.memory for item in rows], ["queued-one", "completes-batch"])

    async def test_training_end_processes_only_its_scope_tail_and_updates_used_memory(self) -> None:
        async with self.memory(batch_size=3) as memory:
            await submit_learning(memory, replace(episode(self.scope, "seed"),
                extensions={"memrl": AddOptions(end_of_training=True)}))
            used = await memory.search("seed")
            native = self.harness.latest(self.scope)
            identity = used.items[0]["memory_id"]
            self.assertEqual(native.text_memory().get(identity).metadata.model_extra["q_value"], 1.0)
            await submit_learning(memory, episode(self.scope, "pending"))
            other = {"user_id": "other-user"}
            await submit_learning(memory, episode(other, "other-pending"))
            await memory.flush()
            await memory.save(self.root / "before-training-end.json")
            self.assertEqual(native.calls, [("add_memories", 1), ("update_values", 1)])
            final = replace(episode(self.scope, "failed-final", success=False, applied_memory=[used]),
                            extensions={"memrl": {"end_of_training": True}})
            await submit_learning(memory, final)
            self.assertEqual(native.calls[-2:], [("add_memories", 2), ("update_values", 2)])
            metadata = native.text_memory().get(identity).metadata.model_extra
            self.assertAlmostEqual(metadata["q_value"], 0.4)
            self.assertEqual(metadata["q_visits"], 1)
            self.assertEqual({row.memory for row in native.text_memory().get_all()},
                             {"seed", "pending", "failed-final"})
            self.assertFalse(memory._method._pending_batch.get(memory._method._scope_key(self.scope)))
            self.assertEqual(len(memory._method._pending_batch[memory._method._scope_key(other)]), 1)
            self.assertEqual(self.harness.latest(other).calls, [])
            self.assertNotIn("end_of_training", metadata)
            self.assertEqual(await submit_learning(memory, final), memory._history.results[
                memory._history.identity(final.episode_id, final.scope)])
            self.assertEqual(native.calls[-2:], [("add_memories", 2), ("update_values", 2)])

    async def test_training_end_restores_legacy_pending_and_full_batch_evolves_once(self) -> None:
        adapter = await self.adapter(batch_size=3)
        await submit_learning(adapter, episode(self.scope, "legacy-pending"))
        saved = await adapter.save()
        self.assertEqual(saved["schema_version"], 3)
        await adapter.close()
        restored = await self.adapter(batch_size=3)
        await restored.load(saved)
        native = self.harness.latest(self.scope)
        self.assertEqual(native.text_memory().get_all(), [])
        await submit_learning(restored, replace(episode(self.scope, "final-after-restore"),
            extensions={"memrl": {"end_of_training": True}}))
        self.assertEqual(native.calls, [("add_memories", 2), ("update_values", 2)])
        self.assertFalse(restored._pending_batch)
        for index in range(3):
            await submit_learning(restored, replace(episode(self.scope, f"full-{index}"),
                extensions={"memrl": {"end_of_training": index == 2}}))
        self.assertEqual(native.calls, [("add_memories", 2), ("update_values", 2),
                                       ("add_memories", 3), ("update_values", 3)])

    async def test_training_end_rejects_non_boolean_input_before_learning(self) -> None:
        async with self.memory(batch_size=3) as memory:
            with self.assertRaises(CapabilityMismatchError):
                await submit_learning(memory, replace(episode(self.scope, "bad-end"),
                    extensions={"memrl": {"end_of_training": "true"}}))
            self.assertFalse(memory._method._pending_batch)
            schema = PLUGIN.descriptor.input_schemas
            self.assertEqual(schema["learn"]["properties"]["memrl"]["properties"]
                             ["end_of_training"]["type"], "boolean")
            self.assertNotIn("end_of_training", schema["retrieve"]["properties"]["memrl"]["properties"])

    async def test_flush_persists_pending_work_and_new_setup_requires_explicit_load(self) -> None:
        adapter = await self.adapter(profile="bcb", batch_size=2)
        await submit_learning(adapter, episode(self.scope, "committed-queue"))
        await adapter.flush()
        self.assertEqual(self.harness.latest(self.scope).text_memory().get_all(), [])
        state_file = self.root / "artifacts" / "memrl" / "adapter_state.json"
        self.assertTrue(state_file.is_file())
        committed = json.loads(state_file.read_text(encoding="utf-8"))
        await adapter.close()
        restored = await self.adapter(profile="bcb", batch_size=2)
        self.assertEqual((await restored.save())["services"], {})
        await restored.load(committed)
        self.assertEqual(await restored.save(), committed)
        self.assertEqual(self.harness.latest(self.scope).text_memory().get_all(), [])
        await submit_learning(restored, episode(self.scope, "completes-restored-batch"))
        rows = self.harness.latest(self.scope).text_memory().get_all()
        self.assertEqual([item.memory for item in rows], ["committed-queue", "completes-restored-batch"])

    async def test_snapshots_are_immutable_and_state_is_stable_across_restores(self) -> None:
        adapter = await self.adapter(profile="bcb", batch_size=1, epsilon=0.0)
        await submit_learning(adapter, episode(self.scope, "first"))
        state_a = await adapter.save()
        original_root = snapshot_path(state_a)
        original_digest = directory_digest(original_root)
        self.assertEqual(await adapter.save(), state_a)
        await submit_learning(adapter, episode(self.scope, "second"))
        state_b = await adapter.save()
        self.assertNotEqual(snapshot_path(state_b), original_root)
        self.assertEqual(directory_digest(original_root), original_digest)
        await adapter.load(state_a)
        native = self.harness.latest(self.scope)
        self.assertNotEqual(Path(native._qdrant_dir), original_root / "qdrant")
        self.assertEqual([item.memory for item in native.text_memory().get_all()], ["first"])
        self.assertEqual(await adapter.save(), state_a)
        self.assertTrue(Path(native._qdrant_dir).is_dir())
        self.assertEqual(directory_digest(original_root), original_digest)
        for _ in range(2):
            frozen = await adapter.save()
            self.assertTrue((await adapter.search("first", top_k=5, scope=self.scope, extensions={})).items)
            await adapter.load(frozen)
            self.assertEqual(await adapter.save(), frozen)
            self.assertEqual(directory_digest(original_root), original_digest)

    async def test_loading_empty_snapshot_drops_all_scope_state(self) -> None:
        adapter = await self.adapter(profile="bcb", batch_size=1)
        empty = await adapter.save()
        await submit_learning(adapter, episode(self.scope, "must-disappear"))
        old_service = self.harness.latest(self.scope)
        await adapter.load(empty)
        self.assertEqual(await adapter.save(), empty)
        self.assertEqual((await adapter.search("must-disappear", top_k=5, scope=self.scope, extensions={})).items, [])
        self.assertIsNot(self.harness.latest(self.scope), old_service)

    def memory(self, **params):
        return Memory({"method": {"method_id": "memrl", "params": {"epsilon": 0.0, **params}},
            "scope": self.scope, "runtime": {"artifact_dir": str(self.root / "runtime")}},
            services=MethodServices(objects={"memrl.service_factory": self.harness.factory}))

    async def test_typed_configuration_and_public_example_follow_the_same_agent_flow(self):
        from mematlas.methods.memrl import example
        memories, inputs = [], []

        def create(config):
            config = replace(config, runtime=RuntimeConfig(artifact_dir=str(self.root / "example")))
            memory = Memory(config, services=MethodServices(objects={"memrl.service_factory": self.harness.factory}))
            memories.append(memory)
            return memory

        async def agent(messages):
            inputs.append(messages)
            return "verified answer"

        async def evaluate(task, answer):
            return answer == "verified answer"

        with patch.object(example, "Memory", side_effect=create):
            tasks = [f"task-{index}" for index in range(33)]
            answers = await example.run(tasks, agent, evaluate, providers={})
        self.assertEqual(answers, ["verified answer"] * 33)
        memory = memories[0]
        self.assertEqual(memory.config.method.params, {"batch_size": 32, "topk": 3})
        self.assertEqual(memory.stats()["episodes"]["learned_exposures"], 33)
        self.assertEqual(self.harness.latest({"user_id": "alice"}).calls,
                         [("add_memories", 32), ("update_values", 32),
                          ("add_memories", 1), ("update_values", 1)])
        self.assertFalse(memory._method._pending_batch)
        for key in memory._history.episodes:
            sample = memory._history.episode(key)
            self.assertEqual([event.role for event in sample.events], ["user", "assistant"])
            self.assertTrue(all(event.episode_id == sample.episode_id for event in sample.events))
            self.assertTrue(sample.success)
            self.assertEqual(len(sample.applied_memory), 1)
        self.assertEqual({memory._history.episode(key).extensions["memrl"]["task_id"]
                          for key in memory._history.episodes}, set(range(33)))
        self.assertTrue({"observe", "retrieve", "learn"}.issubset(
            {trace.operation for trace in memory.traces.all()}))
        self.assertNotIn("evolve", {trace.operation for trace in memory.traces.all()})

    async def test_optional_typed_inputs_and_plain_dicts_use_the_same_public_operations(self):
        async with self.memory(batch_size=1, sim_threshold_by_task={"os": 0.4}) as memory:
            await submit_learning(memory, episode(self.scope, "seed"))
            native = self.harness.latest(self.scope)
            inputs = [("plain", {"task_type": "os"}, {"task_id": 123, "task_type": "os"}),
                      ("typed", SearchOptions(task_type="os"), AddOptions(task_id=124, task_type="os"))]
            for identity, retrieve_options, learn_options in inputs:
                with patch.object(native, "retrieve_query", wraps=native.retrieve_query) as retrieve:
                    used = await memory.search("seed", extensions={"memrl": retrieve_options})
                self.assertEqual(retrieve.call_args.kwargs["threshold"], 0.4)
                sample = replace(episode(self.scope, identity, applied_memory=[used]),
                                 extensions={"memrl": learn_options})
                result = await submit_learning(memory, sample)
                self.assertEqual(result.learning, "submitted")
                stored = next(row for row in native.text_memory().get_all() if row.memory == identity)
                self.assertEqual(stored.metadata.model_extra["task_id"], learn_options["task_id"])
                self.assertEqual(stored.metadata.model_extra["task_type"], "os")

    async def test_freezing_pending_experiences_does_not_trigger_evolution(self):
        config = MemAtlasConfig(method=MemRLConfig(batch_size=4), scope=self.scope,
                                runtime=RuntimeConfig(artifact_dir=str(self.root / "frozen")))
        async with Memory(config, services=MethodServices(objects={"memrl.service_factory": self.harness.factory})) as memory:
            await submit_learning(memory, episode(self.scope, "pending"))
            native = self.harness.latest(self.scope)
            with patch.object(memory._method, "_evolve_batch", wraps=memory._method._evolve_batch) as evolve:
                async with memory.read_only():
                    await memory.search("pending")
                self.assertEqual(evolve.await_count, 0)
            self.assertEqual(native.calls, [])
            self.assertTrue(memory._method._pending_batch)
            self.assertFalse(hasattr(memory, "evolve"))
            self.assertEqual(native.calls, [])
            for index in range(3):
                await submit_learning(memory, episode(self.scope, f"remaining-{index}"))
            self.assertEqual(native.calls, [("add_memories", 4), ("update_values", 4)])

    async def test_public_add_collects_task_units_and_waits_for_real_feedback(self):
        from mematlas import MemoryUnit
        async with self.memory(batch_size=1) as memory:
            event = MemoryEvent("action", "actual action", episode_id="A")
            receipt = await memory.add(MemoryUnit(task="real task", events=[event], episode_ref="A"))
            self.assertFalse(memory._method._instances)
            await memory.episode_end("A")
            self.assertFalse(memory._method._instances)
            feedback = MemoryUnit(type="feedback", related_refs=(receipt.write_ref,), feedback={"success": True})
            await memory.add(feedback)
            await memory.add(feedback)
            rows = (await memory.search("real task")).items
            self.assertEqual(len(rows), 1)
            self.assertIn(event.event_id, rows[0]["source_event_ids"])

    async def test_public_save_load_preserves_pending_batch_without_implicit_evolution(self):
        path = self.root / "pending.json"
        async with self.memory(batch_size=4) as memory:
            await submit_learning(memory, episode(self.scope, "queued"))
            await memory.save(path)
            async with memory.read_only():
                self.assertFalse((await memory.search("queued")).items)
            self.assertEqual(self.harness.latest(self.scope).calls, [])
        async with self.memory(batch_size=4) as restored:
            await restored.load(path)
            self.assertFalse((await restored.search("queued")).items)
            await submit_learning(restored, MemoryEpisode("queued", task="queued", success=True, scope=self.scope))
            self.assertEqual(sum(map(len, restored._method._pending_batch.values())), 1)
            for index in range(3):
                await submit_learning(restored, episode(self.scope, f"remaining-{index}"))
            self.assertEqual(len((await restored.search("queued", top_k=10)).items), 4)
            self.assertFalse(restored._method._pending_batch)

    async def test_retrieval_attribution_uses_applied_contribution_and_preserves_empty_selection(self):
        adapter = await self.adapter(batch_size=1, profile="bcb")
        await submit_learning(adapter, episode(self.scope, "seed"))
        native = self.harness.latest(self.scope)
        contribution = await adapter.search("seed", top_k=1, scope=self.scope, extensions={})
        original = native.retrieve_query
        calls = 0
        def query(task, **options):
            nonlocal calls
            calls += 1
            result, _ = original(task, **options)
            if calls == 1:
                result["selected"] = []
            return result, [("seed" if calls == 1 else "unapplied", 1.0)]
        with patch.object(native, "retrieve_query", side_effect=query):
            empty = await adapter.search("first", top_k=1, scope=self.scope, extensions={})
            await adapter.search("later", top_k=1, scope=self.scope, extensions={})
        self.assertFalse(empty.items)
        await submit_learning(adapter, episode(self.scope, "grouped", applied_memory=[empty]))
        self.assertIn("grouped", [native.text_memory().get(identity).memory for identity in native.dict_memory["seed"]])
        self.assertNotIn("unapplied", native.dict_memory)
        used_id = contribution.items[0]["memory_id"]
        await submit_learning(adapter, episode(self.scope, "failed", success=False, applied_memory=[contribution]))
        self.assertEqual(native.text_memory().get(used_id).metadata.model_extra["q_visits"], 1)
        other = replace(contribution, extensions={"memrl": {"scope": {"user_id": "other"}}})
        with self.assertRaisesRegex(ValueError, "scope"):
            await submit_learning(adapter, episode(self.scope, "foreign", applied_memory=[other]))

    async def test_multiple_applied_queries_and_context_credit_only_selected_sources(self):
        adapter = await self.adapter(batch_size=5)
        first = MemoryContribution(items=[{"memory_id": "used", "content": "a"}, {"memory_id": "unused", "content": "b"}],
            context=[{"content": "a formatted", "source_item_indices": [0]}],
            extensions={"memrl": {"scope": self.scope, "query_candidates": [["first", 0.9]]}})
        second = MemoryContribution(items=[{"memory_id": "used", "content": "a"}],
            extensions={"memrl": {"scope": self.scope, "query_candidates": [["second", 0.8]]}})
        await submit_learning(adapter, episode(self.scope, "multiple", applied_memory=[first, second]))
        row = next(iter(adapter._pending_batch.values()))[0]
        self.assertEqual(row["memory_ids"], ["used"])
        self.assertEqual(dict(row["queries"]), {"first": 0.9, "second": 0.8})

    async def test_explicit_top_k_reaches_native_selection_and_restores_profile(self):
        adapter = await self.adapter(batch_size=1, topk=1)
        for name in ("one", "two", "three"):
            await submit_learning(adapter, episode(self.scope, name))
        result = await adapter.search("query", top_k=3, scope=self.scope, extensions={})
        self.assertEqual(len(result.items), 3)
        self.assertEqual(self.harness.latest(self.scope).rl_config.topk, 1)

    async def test_native_failure_result_is_rejected_and_learning_rolled_back(self):
        async with self.memory(batch_size=1) as memory:
            await submit_learning(memory, episode(self.scope, "initial"))
            native = self.harness.latest(self.scope)
            before = await memory._capture_state()
            with patch.object(native, "add_memories", return_value={}):
                with self.assertRaisesRegex(MethodOperationError, "complete batch"):
                    await submit_learning(memory, episode(self.scope, "failed"))
            self.assertEqual(await memory._capture_state(), before)
            self.assertEqual(memory.state.value, "started")

    async def test_native_intentional_noop_is_preserved(self):
        for options in ({"update_strategy": "validation"}, {"adjustment_mode": "inplace"}):
            adapter = await self.adapter(batch_size=1, **options)
            await submit_learning(adapter, episode(self.scope, "no-op", success=False))
            self.assertFalse(adapter._pending_batch)
            self.assertEqual(self.harness.latest(self.scope).text_memory().get_all(), [])

    async def test_nullable_parameters_and_dependency_checks_are_effective(self):
        self.assertFalse(PLUGIN.validate_config(MethodConfig("memrl", {"q_floor": None, "llm_max_tokens": None})))
        self.assertTrue(PLUGIN.validate_config(MethodConfig("memrl", {"q_floor": "bad"})))
        from importlib import metadata
        version = metadata.version
        with patch.object(metadata, "version", side_effect=lambda name: "3.0.0" if name == "openai" else version(name)):
            report = PLUGIN.healthcheck(MethodConfig("memrl"), MethodServices(objects={"memrl.service_factory": self.harness.factory}))
        self.assertFalse(report["ready"])
        self.assertTrue(any("openai" in row for row in report["incompatible_packages"]))

    async def test_continuous_evaluation_uses_native_batch_boundaries(self):
        from mematlas import ExperimentConfig, ExperimentRunner
        from mematlas.providers import CallableLLM
        from mematlas.benchmarks.datasets import InMemoryBenchmark
        from mematlas.core.evaluation import BenchmarkCase
        benchmark = InMemoryBenchmark("memrl-continuous", [
            *[BenchmarkCase(identity, "query", scope=self.scope, episode=episode(self.scope, identity),
                            metadata={"split": "train"}) for identity in ("one", "two")],
            BenchmarkCase("test", "query", scope=self.scope),
        ])
        config = ExperimentConfig.from_dict({"methods": [{"method": {"method_id": "memrl", "params": {"epsilon": 0.0, "batch_size": 2}},
            "scope": self.scope, "runtime": {"artifact_dir": str(self.root / "runtime")}}],
            "benchmark": benchmark, "protocol": "experience_learning_curve", "checkpoints": (0, 1, 2),
            "output_dir": str(self.root / "evaluation")})
        supplied = MethodServices(objects={"memrl.service_factory": self.harness.factory,
            "answer_llm": CallableLLM(lambda messages, **kwargs: "answer")})
        result = (await ExperimentRunner().evaluate(config, services=supplied))[0]
        self.assertFalse(result.failures, result.failures)
        self.assertEqual([len(row["predictions"][0]["applied_memory"][0]["items"])
                          for row in result.learning_curve], [0, 0, 2])

    async def test_generic_agent_evaluation_uses_the_same_memory_episode(self):
        from mematlas import ExperimentConfig, ExperimentRunner
        from mematlas.benchmarks.datasets import InMemoryBenchmark
        from mematlas.core.evaluation import BenchmarkCase
        benchmark = InMemoryBenchmark("memrl-native-fixture", [
            BenchmarkCase("train", "task", scope=self.scope, metadata={"split": "train"}),
            BenchmarkCase("test", "task", scope=self.scope, reference="answer")])
        async def agent(case, memory):
            contribution = await memory.search(case.query, scope=case.scope)
            return MemoryEpisode(task=case.query, success=True, output="answer", scope=case.scope,
                events=[MemoryEvent("action", "actual action")], applied_memory=[contribution])
        config = ExperimentConfig.from_dict({"methods": [{"method": {"method_id": "memrl", "params": {"epsilon": 0.0, "batch_size": 1}},
            "scope": self.scope, "runtime": {"artifact_dir": str(self.root / "runtime")}}],
            "benchmark": benchmark, "protocol": "experience_train_test", "output_dir": str(self.root / "evaluation")})
        result = (await ExperimentRunner().evaluate(config, agent=agent,
            services=MethodServices(objects={"memrl.service_factory": self.harness.factory})))[0]
        self.assertFalse(result.failures, result.failures)
        self.assertEqual(result.metrics["common.exact_match"], 1.0)


class RealQdrantMemRLTests(unittest.IsolatedAsyncioTestCase):
    async def test_cosine_float32_restore_drift_reuses_snapshot_but_real_vector_change_does_not(self):
        directory = tempfile.TemporaryDirectory(prefix="memrl-cosine-snapshot-")
        temporary = directory.name
        directory.name = str(_filesystem_path(Path(directory.name)))
        with directory:
            with NativeHarness(Path(temporary), real_qdrant=True) as harness:
                scope = {"user_id": "cosine-snapshot"}
                adapter = harness.adapter(batch_size=1)
                try:
                    await setup(adapter)
                    await submit_learning(adapter, episode(scope, "first"))
                    native = harness.latest(scope)
                    vector_db = native.text_memory().vector_db
                    point = vector_db.get_all()[0]
                    point.vector = [0.1] * len(point.vector)
                    vector_db.update(point.id, point)
                    state = await adapter.save()
                    original_root = snapshot_path(state)
                    original_bytes = directory_digest(original_root)
                    original_vector = native.text_memory().vector_db.get_by_id(point.id).vector
                    await adapter.load(state)
                    restored_vector = native.text_memory().vector_db.get_by_id(point.id).vector
                    self.assertNotEqual(original_vector, restored_vector)
                    for _ in range(3):
                        self.assertEqual(await adapter.save(), state)
                        await adapter.load(state)
                        self.assertEqual(directory_digest(original_root), original_bytes)

                    vector_db = native.text_memory().vector_db
                    changed = vector_db.get_by_id(point.id)
                    changed.vector = [1.0] + [0.0] * (len(changed.vector) - 1)
                    vector_db.update(changed.id, changed)
                    different_state = await adapter.save()
                    self.assertNotEqual(different_state, state)
                    self.assertFalse(adapter._equivalent_snapshot(snapshot_path(different_state), original_root))
                    self.assertEqual(directory_digest(original_root), original_bytes)
                finally:
                    await adapter.close()

    async def test_native_retrieval_strategies_use_their_declared_paths(self):
        with tempfile.TemporaryDirectory(prefix="memrl-strategy-") as temporary:
            with NativeHarness(Path(temporary), real_qdrant=True) as harness:
                scope = {"user_id": "strategy"}
                for strategy, value in (("query", False), ("avefact", True), ("random", True)):
                    adapter = harness.adapter(profile="bcb", batch_size=1, epsilon=0.0,
                        retrieve_strategy=strategy, enable_value_driven=value, sim_threshold=0.0)
                    try:
                        await setup(adapter)
                        await submit_learning(adapter, episode(scope, "first"))
                        result = await adapter.search("first", top_k=1, scope=scope, extensions={})
                        self.assertEqual(len(result.items), 1)
                        self.assertIn("first", result.items[0]["content"])
                    finally:
                        await adapter.close()

    async def test_real_sqlite_reopen_and_repeated_native_snapshot_restores(self) -> None:
        directory = tempfile.TemporaryDirectory(prefix="memrl-real-qdrant-")
        temporary = directory.name
        directory.name = str(_filesystem_path(Path(directory.name)))
        with directory:
            with NativeHarness(Path(temporary), real_qdrant=True) as harness:
                scope = {"user_id": "real-qdrant"}
                adapter = harness.adapter(profile="bcb", batch_size=1, epsilon=0.0)
                restored_adapter = None
                try:
                    await setup(adapter)
                    await submit_learning(adapter, episode(scope, "first"))
                    state = await adapter.save()
                    root = snapshot_path(state)
                    original_digest = directory_digest(root)
                    native = harness.latest(scope)
                    original_client = native.text_memory().vector_db.client
                    await submit_learning(adapter, episode(scope, "second"))
                    await adapter.load(state)
                    self.assertTrue(original_client._client._closed)
                    self.assertEqual([item.memory for item in native.text_memory().get_all()], ["first"])
                    for _ in range(3):
                        self.assertEqual(await adapter.save(), state)
                        self.assertEqual(len((await adapter.search("first", top_k=5, scope=scope, extensions={})).items), 1)
                        await adapter.load(state)
                        self.assertEqual(directory_digest(root), original_digest)
                    await adapter.close()
                    restored_adapter = harness.adapter(profile="bcb", batch_size=1, epsilon=0.0)
                    await setup(restored_adapter)
                    await restored_adapter.load(state)
                    self.assertEqual(await restored_adapter.save(), state)
                    self.assertEqual(len((await restored_adapter.search("first", top_k=5, scope=scope, extensions={})).items), 1)
                    self.assertEqual(directory_digest(root), original_digest)
                finally:
                    await adapter.close()
                    if restored_adapter is not None:
                        await restored_adapter.close()


class BundledSourceTests(unittest.TestCase):
    def test_complete_native_suite_runs_with_historical_source_inaccessible(self) -> None:
        script = """
import importlib.abc, importlib.machinery, io, json, os, sys, unittest
from pathlib import Path

repository = Path(sys.argv[1]).resolve()
forbidden_root = repository / 'plan'
vendor_root = repository / 'methods' / 'memrl' / '_vendor'

def forbidden(path):
    if not isinstance(path, (str, bytes, os.PathLike)):
        return False
    return Path(os.fsdecode(path)).resolve().is_relative_to(forbidden_root)

class RejectHistoricalSource(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if spec and (forbidden(spec.origin) or any(forbidden(location)
                for location in (spec.submodule_search_locations or ()))):
            raise ImportError('Historical source import is forbidden: ' + fullname)
        return None

def reject_historical_file_access(event, args):
    if event == 'open' and args and forbidden(args[0]):
        raise PermissionError('Historical source access is forbidden: ' + os.fsdecode(args[0]))

sys.meta_path.insert(0, RejectHistoricalSource())
sys.addaudithook(reject_historical_file_access)
sys.path[:0] = [str(repository / 'test' / 'memrl'), str(repository.parent)]
import check_memrl

# Run all memory/SQLite cases, excluding this subprocess wrapper to avoid recursion.
suite = unittest.TestSuite([
    unittest.defaultTestLoader.loadTestsFromTestCase(check_memrl.NativeMemRLTests),
    unittest.defaultTestLoader.loadTestsFromTestCase(check_memrl.RealQdrantMemRLTests),
])
output = io.StringIO()
result = unittest.TextTestRunner(stream=output, verbosity=0).run(suite)
modules = {name: str(Path(module.__file__).resolve()) for name, module in sys.modules.items()
           if (name == 'memrl' or name.startswith('memrl.')) and getattr(module, '__file__', None)}
assert 'memrl.service.memory_service' in modules, modules
assert all(Path(path).is_relative_to(vendor_root) for path in modules.values()), modules
report = check_memrl.PLUGIN.healthcheck(check_memrl.MethodConfig('memrl'),
    check_memrl.MethodServices(llm=object(), embedder=object()))
assert report['source_root'] == str(vendor_root), report
assert report['source_file_hashes_verified'] is True, report
assert report['source_revision_verified'] is True, report
if not result.wasSuccessful():
    print(output.getvalue(), file=sys.stderr)
    sys.exit(1)
print(json.dumps({'tests_run': result.testsRun, 'native_module_count': len(modules),
    'all_native_modules_vendored': True, 'source_hashes_verified': True,
    'historical_source_blocked': True}))
"""
        with tempfile.TemporaryDirectory(prefix="memrl-vendor-guard-") as temporary:
            environment = {**os.environ, "MEMOS_BASE_PATH": temporary, "PYTHON_DOTENV_DISABLED": "1"}
            process = subprocess.run([sys.executable, "-c", script, str(REPOSITORY_ROOT)],
                                     capture_output=True, text=True, check=False, env=environment, timeout=60)
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
        result = json.loads(process.stdout.splitlines()[-1])
        expected = sum(unittest.defaultTestLoader.loadTestsFromTestCase(case).countTestCases()
                       for case in (NativeMemRLTests, RealQdrantMemRLTests))
        self.assertEqual(result["tests_run"], expected)
        self.assertGreater(result["native_module_count"], 0)
        self.assertTrue(result["all_native_modules_vendored"])
        self.assertTrue(result["source_hashes_verified"])
        self.assertTrue(result["historical_source_blocked"])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s", force=True)
    unittest.main(verbosity=2)
