"""Offline fidelity regression: .venv/bin/python test/memq/check_memq.py"""

from __future__ import annotations

import copy
import json
import logging
import os
import random
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fixtures import (
    MemoryScope, MemoryService, MemoryStore, NativeHarness, REPOSITORY_ROOT,
    REFLECTION, RLConfig, SCRIPT, UPSTREAM_ROOT, directory_digest, episode,
    native_instance, q_value, query, record_native_calls, scope_key, store_row,
)
from mematlas.core.config import MethodConfig
from mematlas.core.errors import ConfigurationError, MethodOperationError
from mematlas.core.models import ContentBlock, EventKind, MemoryEvent
from mematlas.core.plugin import MethodServices
from mematlas.core.provider_config import resolve_providers
from mematlas.core.registry import default_registry
from mematlas.methods.memq.method import PLUGIN, MemQMemory
from memq.providers.embedding import OpenAIEmbedder
from memq.providers.llm import OpenAILLM


class NativeMemQTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="memq-native-tests-")
        self.root = Path(self.temporary.name)
        self.harness = NativeHarness(self.root)
        self.scope = MemoryScope(user_id="offline-user")

    async def asyncTearDown(self) -> None:
        await self.harness.close()
        self.temporary.cleanup()

    async def test_builtin_discovery_and_missing_source_preflight_without_native_import(self) -> None:
        self.assertIs(default_registry().get("memq"), PLUGIN)
        script = """
import json, sys
sys.path.insert(0, sys.argv[1])
from mematlas.core.config import MethodConfig
from mematlas.core.plugin import MethodServices
from mematlas.methods.memq.method import PLUGIN
report = PLUGIN.healthcheck(MethodConfig('memq'), MethodServices(
    llm=object(), embedder=object(), source_roots={'memq': sys.argv[2]}))
print(json.dumps({'report': report, 'native_imported': any(
    name == 'memq' or name.startswith('memq.') for name in sys.modules)}))
"""
        missing = self.root / "absent-source"
        process = subprocess.run([sys.executable, "-c", script, str(REPOSITORY_ROOT.parent), str(missing)],
                                 capture_output=True, text=True, timeout=30, check=False)
        self.assertEqual(process.returncode, 0, process.stderr)
        result = json.loads(process.stdout)
        self.assertFalse(result["native_imported"])
        self.assertFalse(result["report"]["ready"])
        self.assertFalse(result["report"]["source_ready"])
        self.assertEqual(result["report"]["source_root"], str(missing))
        self.assertIn("memq/service/memory_service.py", result["report"]["missing_source_files"])

    async def test_vendored_source_inventory_detects_tampering_missing_files_and_wrong_revision(self) -> None:
        def preflight(root: Path) -> dict:
            return PLUGIN.healthcheck(MethodConfig("memq"), MethodServices(
                llm=object(), embedder=object(), source_roots={"memq": str(root)}))

        intact = preflight(UPSTREAM_ROOT)
        self.assertTrue(intact["source_ready"], intact)
        self.assertEqual(intact["source_provenance"], "vendored")
        self.assertTrue(intact["source_file_hashes_verified"])
        self.assertTrue(intact["source_revision_verified"])
        for mutation in ("tamper", "missing", "revision"):
            with self.subTest(mutation=mutation):
                source = self.root / mutation
                shutil.copytree(UPSTREAM_ROOT, source,
                                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
                if mutation == "tamper":
                    path = source / "memq" / "service" / "memory_service.py"
                    path.write_text(path.read_text(encoding="utf-8") + "\n# fixture tamper\n", encoding="utf-8")
                elif mutation == "missing":
                    (source / "memq" / "service" / "memory_store.py").unlink()
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
                    self.assertFalse(report["source_revision_verified"])
                else:
                    self.assertFalse(report["source_file_hashes_verified"])

    async def test_official_defaults_and_native_numpy_store_initialization(self) -> None:
        raw = MemQMemory(MethodConfig("memq"), MethodServices(llm=object(), embedder=object()))
        self.assertEqual(raw.settings.profile, "default")
        values = raw.settings.values
        self.assertEqual((values["batch_size"], values["num_epochs"]), (50, 5))
        self.assertEqual((values["memrl_update_mode"], values["memq_update_mode"]), ("batch", "epoch"))
        self.assertEqual((values["alpha"], values["gamma"], values["lambd"]), (.3, .5, .8))
        self.assertEqual((values["weight_sim"], values["weight_q"]), (.7, .3))
        self.assertEqual((values["q_init_pos"], values["q_init_neg"]), (.66, .66))
        self.assertEqual((values["build_strategy"], values["retrieve_strategy"],
                          values["update_strategy"]), ("proceduralization", "query", "adjustment"))
        adapter = await self.harness.adapter()
        contribution = await adapter.retrieve(query(self.scope, "initial task"))
        native = native_instance(adapter, self.scope)
        self.assertIsInstance(native, MemoryService)
        self.assertIsInstance(native._store, MemoryStore)
        self.assertIsInstance(native.rl_config, RLConfig)
        self.assertEqual(native._store._vectors.shape, (0, 4))
        self.assertEqual(str(native._store._vectors.dtype), "float32")
        self.assertEqual(contribution.items, ())

    async def test_default_commit_keeps_pending_learning_until_explicit_epoch_boundary(self) -> None:
        adapter = await self.harness.adapter()
        await adapter.learn(episode(self.scope, "first task"))
        native = native_instance(adapter, self.scope)
        self.assertEqual(len(native._store), 0)
        await adapter.commit()
        self.assertEqual(len(native._store), 0)
        await adapter.command("end_epoch", {})
        self.assertEqual(len(native._store), 1)
        self.assertAlmostEqual(q_value(native, native._store._ids[0]), .66)
        await adapter.commit()
        self.assertEqual(len(native._store), 1)

    async def test_native_add_retrieve_two_stage_q_update_and_log_namespaces(self) -> None:
        with self.assertLogs(level=logging.INFO) as logs:
            adapter = await self.harness.adapter(batch_size=1)
            await adapter.learn(episode(self.scope, "seed task"))
            await adapter.command("end_epoch", {})
            native = native_instance(adapter, self.scope)
            parent_id = native._store._ids[0]
            self.assertIn(SCRIPT, native._store.get(parent_id)["metadata"]["full_content"])
            contribution = await adapter.retrieve(query(self.scope, "child task"))
            self.assertEqual(contribution.items[0].memory_id, parent_id)
            self.assertAlmostEqual(contribution.items[0].metadata["q_normalized"], 1.0)
            self.assertAlmostEqual(contribution.items[0].score, 1.0)
            self.assertEqual(native._store.get(parent_id)["metadata"]["retrieval_count"], 1)
            supplied = tuple(item.to_dict() for item in contribution.items)
            await adapter.learn(episode(self.scope, "child task", retrieved=supplied))
            self.assertEqual(len(native._store), 2)
            # Official MemRL batch stage executes before adding the child.
            self.assertAlmostEqual(q_value(native, parent_id), .66 + .3 * (1.0 - .66))
            self.assertEqual(native._store.get(parent_id)["metadata"]["q_visits"], 1)
            await adapter.commit()
            self.assertAlmostEqual(q_value(native, parent_id), .762)
            await adapter.command("end_epoch", {})
            # MemQ epoch stage uses the already updated parent and the child Q.
            self.assertAlmostEqual(q_value(native, parent_id), .762 + .3 * (1.0 + .5 * .66 - .762))
            self.assertAlmostEqual(native._q_max, .9324)
            child_id = next(identifier for identifier in native._store._ids if identifier != parent_id)
            self.assertEqual(native._memid_pair[child_id], [parent_id])
        self.assertTrue(any(record.name == "mematlas.methods.memq.adapter" for record in logs.records))
        self.assertTrue(any(record.name.startswith("memq.") for record in logs.records))
        self.assertTrue(any("High-level script:" in str(messages[-1]["content"])
                            for messages, _ in self.harness.llm.calls))

    async def test_explicit_native_batch_preserves_update_add_and_boundary_order(self) -> None:
        observed = ("update_values", "_update_q_max", "decay_traces", "record_retrieval_traces",
                    "add_memories", "compute_bellman_td_and_accumulate", "flush_memq_updates", "reset_traces")
        for mode in ("batch", "epoch"):
            with self.subTest(mode=mode):
                adapter = await self.harness.adapter(batch_size=50, memq_update_mode=mode)
                await adapter.retrieve(query(self.scope, "seed task"))
                native = native_instance(adapter, self.scope)
                store_row(native, "parent", .66)
                await adapter.command("begin_batch", {})
                supplied = ({"memory_id": "parent", "metadata": {
                    "memq_query_candidates": [["parent", 1.0]]}},)
                with record_native_calls(native, observed) as calls:
                    await adapter.learn(episode(self.scope, "one", retrieved=supplied))
                    await adapter.learn(episode(self.scope, "two", success=False, retrieved=supplied))
                    await adapter.commit()
                    self.assertEqual(calls, [])
                    await adapter.command("end_batch", {})
                    names = [name for name, _, _ in calls]
                    expected = ["update_values", "_update_q_max", "decay_traces",
                                "record_retrieval_traces", "add_memories"]
                    if mode == "batch":
                        expected += ["compute_bellman_td_and_accumulate", "flush_memq_updates",
                                     "_update_q_max", "reset_traces"]
                    self.assertEqual(names, expected)
                    self.assertEqual(calls[0][1], ([True, False], [["parent"], ["parent"]]))
                    add_kwargs = next(kwargs for name, _, kwargs in calls if name == "add_memories")
                    self.assertEqual(add_kwargs["task_descriptions"], ["one", "two"])
                    self.assertEqual(add_kwargs["successes"], [True, False])
                with record_native_calls(native, observed) as boundary_calls:
                    await adapter.command("end_epoch", {})
                if mode == "epoch":
                    self.assertEqual([name for name, _, _ in boundary_calls],
                                     ["compute_bellman_td_and_accumulate", "flush_memq_updates",
                                      "reset_traces", "_update_q_max"])
                self.assertEqual(len(native._store), 3)
                self.assertIn(REFLECTION, native._store.get(native._store._ids[-1])["metadata"]["full_content"])

    async def test_cosine_epoch_schedule_and_native_single_step_formula(self) -> None:
        adapter = await self.harness.adapter(alpha_schedule="cosine", batch_size=1)
        await adapter.command("set_epoch_info", {"current_epoch": 1, "total_epochs": 2})
        await adapter.retrieve(query(self.scope, "seed"))
        native = native_instance(adapter, self.scope)
        self.assertAlmostEqual(native.get_alpha(), .15)
        store_row(native, "parent", .66)
        await adapter.learn(episode(self.scope, "child", retrieved=({"memory_id": "parent"},)))
        self.assertAlmostEqual(q_value(native, "parent"), .66 + .15 * (1.0 - .66))
        await adapter.command("end_epoch", {})
        self.assertAlmostEqual(q_value(native, "parent"), .711 + .15 * (1.0 + .5 * .66 - .711))

    async def test_official_profile_runner_exceptions_and_adaptive_child_q_are_preserved(self) -> None:
        observed = ("update_values", "_update_q_max", "decay_traces", "record_retrieval_traces",
                    "add_memories", "compute_bellman_td_and_accumulate", "flush_memq_updates", "reset_traces")
        common_add = ["decay_traces", "record_retrieval_traces", "add_memories"]
        cases = (
            ("llb", {}, common_add,
             ["update_values", "_update_q_max", "compute_bellman_td_and_accumulate",
              "flush_memq_updates", "reset_traces", "_update_q_max"], .9324),
            ("gpqa", {"memq_update_mode": "batch"},
             common_add + ["compute_bellman_td_and_accumulate", "flush_memq_updates", "reset_traces"],
             ["_update_q_max"], .8313),
            ("bfcl", {"memrl_update_mode": "epoch", "memq_update_mode": "batch"}, common_add,
             ["compute_bellman_td_and_accumulate", "flush_memq_updates", "reset_traces", "_update_q_max"], .861),
            ("dsbench", {"memq_update_mode": "batch"},
             ["update_values", "_update_q_max"] + common_add + ["compute_bellman_td_and_accumulate",
              "flush_memq_updates", "reset_traces", "_update_q_max"], [], .9477),
        )
        for profile, params, batch_order, epoch_order, final_q in cases:
            with self.subTest(profile=profile):
                adapter = await self.harness.adapter(profile=profile, **params)
                await adapter.command("begin_epoch", {"current_epoch": 0, "total_epochs": 2})
                await adapter.retrieve(query(self.scope, "seed"))
                native = native_instance(adapter, self.scope)
                store_row(native, "parent", .66)
                await adapter.command("begin_batch", {})
                await adapter.learn(episode(self.scope, "child", retrieved=({"memory_id": "parent"},)))
                with record_native_calls(native, observed) as calls:
                    await adapter.command("end_batch", {})
                self.assertEqual([name for name, _, _ in calls], batch_order)
                child_id = next(identifier for identifier in native._store._ids if identifier != "parent")
                if profile == "dsbench":
                    self.assertAlmostEqual(q_value(native, child_id), .762)
                with record_native_calls(native, observed) as calls:
                    await adapter.command("end_epoch", {})
                self.assertEqual([name for name, _, _ in calls], epoch_order)
                self.assertAlmostEqual(q_value(native, "parent"), final_q)
                after = await adapter.save_state()
                await adapter.command("end_epoch", {})
                self.assertEqual(await adapter.save_state(), after)

    async def test_provenance_fifo_limit_and_dsbench_unbounded_variant(self) -> None:
        for profile in ("default", "dsbench"):
            with self.subTest(profile=profile):
                adapter = await self.harness.adapter(profile=profile)
                await adapter.retrieve(query(self.scope, "seed"))
                key = scope_key(self.scope)
                dag = adapter._dags[key]
                dag.update((f"existing-{index}", ["ancestor"]) for index in range(10000))
                adapter._append_provenance(key, "fresh-child", ["parent"])
                self.assertEqual(dag["fresh-child"], ["parent"])
                self.assertEqual(len(dag), 10000 if profile == "default" else 10001)
                self.assertEqual(next(iter(dag)), "existing-1" if profile == "default" else "existing-0")
                self.assertIs(native_instance(adapter, self.scope)._memid_pair, dag)

    async def test_native_bellman_all_path_lambda_credit_and_contribution_normalization(self) -> None:
        adapter = await self.harness.adapter(normalize_dq=True)
        await adapter.retrieve(query(self.scope, "seed"))
        native = native_instance(adapter, self.scope)
        for identifier, value in {"parent": .6, "child": .7, "A": .2, "B": .3, "X": .1}.items():
            store_row(native, identifier, value)
        dag = {"parent": ["A", "B", "A"], "A": ["X"], "B": ["X"]}
        native.set_memid_pair(dag)
        result = native.compute_bellman_td_and_accumulate([
            {"reward": 1.0, "retrieved_ids": ["parent"], "new_mem_id": "child"},
            {"reward": 0.0, "retrieved_ids": ["parent"], "new_mem_id": "child"},
        ], dag)
        self.assertAlmostEqual(result[0], .5)
        self.assertAlmostEqual(result[1], .75 ** 2 + (-.25) ** 2)
        self.assertEqual(result[2], 2)
        self.assertAlmostEqual(native._accumulated_dq["X"], .3 * .5 * (2 * (.5 * .8) ** 2))
        self.assertEqual(native._accumulated_dq_count["X"], 2)
        native.flush_memq_updates()
        self.assertAlmostEqual(q_value(native, "parent"), .6 + .075)
        self.assertAlmostEqual(q_value(native, "A"), .2 + .03)
        self.assertAlmostEqual(q_value(native, "B"), .3 + .03)
        self.assertAlmostEqual(q_value(native, "X"), .1 + .024)
        self.assertEqual(native._accumulated_dq, {})
        self.assertEqual(native._accumulated_dq_count, {})

    async def test_q_max_normalization_preserves_native_weighted_ranking(self) -> None:
        adapter = await self.harness.adapter()
        await adapter.retrieve(query(self.scope, "new task"))
        native = native_instance(adapter, self.scope)
        store_row(native, "similar", .1, vector=[1.0, 0.0, 0.0, 0.0])
        store_row(native, "valuable", 1.0, vector=[.8, .6, 0.0, 0.0])
        native._q_max = 1.0
        contribution = await adapter.retrieve(query(self.scope, "new task"))
        self.assertEqual([item.memory_id for item in contribution.items], ["valuable", "similar"])
        self.assertAlmostEqual(contribution.items[0].metadata["q_normalized"], 1.0)
        self.assertAlmostEqual(contribution.items[0].score, .8 * .7 + 1.0 * .3)
        self.assertAlmostEqual(contribution.items[1].score, 1.0 * .7 + .1 * .3)

    async def test_native_provider_methods_preserve_script_keyword_chunking_and_logs(self) -> None:
        adapter = await self.harness.adapter(embedding_max_text_len=3)
        await adapter.retrieve(query(self.scope, "seed"))
        native = native_instance(adapter, self.scope)
        llm, embedder = native.llm_provider, native.embedding_provider
        self.assertIs(type(llm).generate, OpenAILLM.generate)
        self.assertIs(type(llm).generate_script, OpenAILLM.generate_script)
        self.assertIs(type(llm).extract_keywords, OpenAILLM.extract_keywords)
        self.assertIs(type(embedder).embed, OpenAIEmbedder.embed)
        self.assertEqual(llm.generate_script("action: perform\nobservation: success"), SCRIPT)
        messages, options = self.harness.llm.calls[-1]
        expected = """
        Analyze the following detailed task trajectory and create a concise,
        high-level script that captures the essential steps and decision points.

        The script should be:
        1. Generic enough to apply to similar tasks
        2. Specific enough to provide useful guidance
        3. 3-5 high-level steps maximum
        4. Focus on the strategy and key decisions, not detailed actions

        Trajectory:
        action: perform\nobservation: success

        High-level script:"""
        self.assertEqual(messages, [{"role": "user", "content": expected}])
        self.assertEqual(options["temperature"], 0.0)
        self.assertEqual(options["max_tokens"], 4096)
        self.assertEqual(llm.extract_keywords("Inspect the procedure", 2), ["inspect", "procedure"])
        self.harness.embedding.vectors.update({"abc": [1.0, 0.0, 0.0, 0.0],
                                               "def": [0.0, 1.0, 0.0, 0.0],
                                               "g": [0.0, 0.0, 1.0, 0.0]})
        merged = embedder.embed(["abcdefg", "abc"])
        self.assertEqual(self.harness.embedding.calls[-1], ["abc", "def", "g", "abc"])
        self.assertEqual(merged, [[1 / 3, 1 / 3, 1 / 3, 0.0], [1.0, 0.0, 0.0, 0.0]])
        self.assertTrue(llm._token_log_path.is_file())
        self.assertTrue(embedder._token_log_path.is_file())
        self.assertEqual(json.loads(llm._token_log_path.read_text().splitlines()[-1])["provider"], "llm")
        self.assertEqual(json.loads(embedder._token_log_path.read_text().splitlines()[-1])["provider"], "embedding")

    async def test_empty_confirmed_retrieval_does_not_credit_recent_query_candidates(self) -> None:
        adapter = await self.harness.adapter(batch_size=1)
        await adapter.learn(episode(self.scope, "parent"))
        await adapter.command("end_epoch", {})
        native = native_instance(adapter, self.scope)
        parent_id = native._store._ids[0]
        contribution = await adapter.retrieve(query(self.scope, "unconfirmed child"))
        self.assertEqual(len(contribution.items), 1)
        await adapter.learn(episode(self.scope, "unconfirmed child", retrieved=()))
        await adapter.command("end_epoch", {})
        self.assertEqual(q_value(native, parent_id), .66)
        self.assertEqual(native._store.get(parent_id)["metadata"]["q_visits"], 0)
        child_id = next(identifier for identifier in native._store._ids if identifier != parent_id)
        self.assertNotIn(child_id, native._memid_pair)
        self.assertNotIn("related_memory_ids", native._store.get(child_id)["metadata"])

    async def test_inline_snapshot_covers_qmax_retrieval_traces_pending_scopes_and_is_immutable(self) -> None:
        adapter = await self.harness.adapter(batch_size=1)
        await adapter.learn(episode(self.scope, "seed"))
        await adapter.command("end_epoch", {})
        await adapter.retrieve(query(self.scope, "child"))
        native = native_instance(adapter, self.scope)
        native._q_max = 9.0
        native._eligibility_traces = {"trace": .75}
        native._accumulated_dq = {"delta": .125}
        native._accumulated_dq_count = {"delta": 2}
        native.set_epoch_info(1, 2)
        native._store.search([1, 0, 0, 0])
        other_scope = MemoryScope(user_id="offline-user", team_id="other-team")
        await adapter.observe(MemoryEvent(EventKind.OBSERVATION, other_scope,
                                          (ContentBlock.text("Buffered scope metadata."),)))
        await adapter.command("begin_batch", {})
        supplied = ({"memory_id": native._store._ids[0]},)
        await adapter.learn(episode(self.scope, "pending child", retrieved=supplied))
        await adapter.retrieve(query(self.scope, "last retrieval to preserve"))
        await adapter.commit()
        persisted = directory_digest(self.root)
        state = await adapter.save_state()
        encoded = json.dumps(state, sort_keys=True)
        rng = random.getstate()
        retrieval = copy.deepcopy(adapter._last_retrievals)
        self.assertTrue(retrieval)
        native._q_max = -1
        native._eligibility_traces["trace"] = -1
        native._store._payloads[native._store._ids[0]]["metadata"]["q_value"] = -99
        await adapter.retrieve(query(other_scope, "new scope"))
        random.random()
        self.assertEqual(json.dumps(state, sort_keys=True), encoded)
        await adapter.load_state(state)
        self.assertEqual(random.getstate(), rng)
        restored_native = native_instance(adapter, self.scope)
        self.assertEqual(restored_native._q_max, 9.0)
        self.assertEqual(restored_native._eligibility_traces, {"trace": .75})
        self.assertEqual(restored_native._accumulated_dq, {"delta": .125})
        self.assertEqual(restored_native._accumulated_dq_count, {"delta": 2})
        self.assertEqual((restored_native._current_epoch, restored_native._total_epochs), (1, 2))
        self.assertEqual(adapter._last_retrievals, retrieval)
        for _ in range(3):
            self.assertEqual(await adapter.save_state(), state)
            await adapter.load_state(state)
        fresh = await self.harness.adapter(batch_size=1)
        await fresh.load_state(state)
        self.assertEqual(await fresh.save_state(), state)
        with self.assertRaisesRegex(ValueError, "schema"):
            await fresh.load_state({"schema_version": 1, "method_id": "memq", "scopes": {}})
        self.assertEqual(await fresh.save_state(), state)
        for relative, digest in persisted.items():
            self.assertEqual(directory_digest(self.root)[relative], digest)

    async def test_committed_pending_batch_and_edges_survive_restart_and_reset_does_not_resurrect(self) -> None:
        storage = self.root / "restarted-bank"
        adapter = await self.harness.adapter(batch_size=1, storage_path=str(storage))
        await adapter.learn(episode(self.scope, "seed"))
        await adapter.command("end_epoch", {})
        native = native_instance(adapter, self.scope)
        parent_id = native._store._ids[0]
        supplied = ({"memory_id": parent_id},)
        await adapter.learn(episode(self.scope, "first child", retrieved=supplied))
        self.assertAlmostEqual(q_value(native, parent_id), .762)
        await adapter.command("begin_batch", {})
        await adapter.learn(episode(self.scope, "second child", retrieved=supplied))
        await adapter.commit()
        state = await adapter.save_state()
        old_files = directory_digest(storage)
        await adapter.close()
        reopened = await self.harness.adapter(batch_size=1, storage_path=str(storage))
        self.assertEqual(await reopened.save_state(), state)
        native = native_instance(reopened, self.scope)
        self.assertEqual(len(native._store), 2)
        self.assertAlmostEqual(q_value(native, parent_id), .762)
        await reopened.command("end_epoch", {})
        self.assertEqual(len(native._store), 3)
        single_step_q = .762 + .3 * (1 - .762)
        self.assertAlmostEqual(q_value(native, parent_id),
                               single_step_q + 2 * .3 * (1 + .5 * .66 - single_step_q))
        await reopened.commit()
        for relative, digest in old_files.items():
            if relative != "adapter_state.json":
                self.assertEqual(directory_digest(storage)[relative], digest, relative)
        await reopened.reset(self.scope)
        await reopened.close()
        empty = await self.harness.adapter(batch_size=1, storage_path=str(storage))
        contribution = await empty.retrieve(query(self.scope, "seed"))
        self.assertEqual(contribution.items, ())
        self.assertEqual(len(native_instance(empty, self.scope)._store), 0)

    async def test_epoch_direct_value_stage_is_deferred_until_all_children_exist(self) -> None:
        adapter = await self.harness.adapter(batch_size=1, memrl_update_mode="epoch")
        await adapter.retrieve(query(self.scope, "parent"))
        native = native_instance(adapter, self.scope)
        store_row(native, "parent", .66)
        supplied = ({"memory_id": "parent"},)
        await adapter.learn(episode(self.scope, "success child", retrieved=supplied))
        await adapter.learn(episode(self.scope, "failure child", success=False, retrieved=supplied))
        self.assertEqual(len(native._store), 3)
        self.assertEqual(q_value(native, "parent"), .66)
        await adapter.commit()
        self.assertEqual(q_value(native, "parent"), .66)
        state = await adapter.save_state()
        restored = await self.harness.adapter(batch_size=1, memrl_update_mode="epoch")
        await restored.load_state(state)
        self.assertEqual(await restored.save_state(), state)
        await restored.command("end_epoch", {})
        native = native_instance(restored, self.scope)
        # Native sequential single-step updates: .66 -> .762 -> .5334.
        # Both epoch TD errors then read that same final parent estimate.
        parent = (.66 + .3 * (1 - .66)) * .7
        expected = parent + .3 * ((1 + .5 * .66 - parent) + (0 + .5 * .66 - parent))
        self.assertAlmostEqual(q_value(native, "parent"), expected)

    async def test_runtime_readonly_preserves_native_retrieval_metadata_qmax_and_scope_state(self) -> None:
        memory = await self.harness.memory(batch_size=1)
        sample = episode(self.scope, "seed")
        await memory.learn(sample.result, events=sample.events)
        await memory.command("end_epoch", {})
        await memory.recall("seed", scope=self.scope)
        # read_only() begins with the public runtime's flush barrier. Persist
        # the preceding training retrieval before taking the file baseline.
        await memory.commit()
        before = await memory.save_state()
        persisted = directory_digest(self.root)
        async with memory.read_only():
            for _ in range(2):
                contribution = await memory.recall("seed", scope=self.scope)
                self.assertEqual(len(contribution.items), 1)
                self.assertEqual(await memory.save_state(), before)
            contribution = await memory.recall("unknown", scope=MemoryScope(user_id="new-user"))
            self.assertEqual(contribution.items, ())
            self.assertEqual(await memory.save_state(), before)
        self.assertEqual(await memory.save_state(), before)
        for relative, digest in persisted.items():
            self.assertEqual(directory_digest(self.root)[relative], digest, relative)

    async def test_runtime_readonly_provider_error_restores_state_after_native_retries(self) -> None:
        memory = await self.harness.memory(batch_size=1)
        sample = episode(self.scope, "seed")
        await memory.learn(sample.result, events=sample.events)
        await memory.command("end_epoch", {})
        before = await memory.save_state()
        calls_before = len(self.harness.embedding.calls)
        async with memory.read_only():
            self.harness.embedding.failure = RuntimeError("offline query provider failure")
            try:
                # Only the I/O backoff clock is removed; native five-attempt
                # retry control and error propagation still execute.
                with patch("memq.service.memory_service.time.sleep", return_value=None), \
                        self.assertRaisesRegex(MethodOperationError, "query provider failure"):
                    await memory.recall("seed", scope=self.scope)
            finally:
                self.harness.embedding.failure = None
            self.assertEqual(await memory.save_state(), before)
        self.assertEqual(len(self.harness.embedding.calls) - calls_before, 5)
        self.assertEqual(await memory.save_state(), before)

    async def test_runtime_provider_defaults_metering_and_public_overrides(self) -> None:
        providers = {
            "llm": {"model": "custom-llm", "base_url": "http://127.0.0.1:1/v1"},
            "embedding": {"model": "custom-embedding", "base_url": "http://127.0.0.1:1/v1", "dimensions": 4},
        }
        services = MethodServices(llm=object(), embedder=object(), provider_configs=providers,
                                  artifact_dir=self.root / "public-providers")
        adapter = MemQMemory(MethodConfig("memq", {"profile": "lcb"}), services)
        resolved = resolve_providers(adapter.settings.values, providers)
        self.assertEqual(resolved["llm"].model, "custom-llm")
        self.assertEqual(resolved["embedding"].model, "custom-embedding")
        self.assertEqual(adapter.settings.values["vector_dim"], 4)
        report = PLUGIN.healthcheck(MethodConfig("memq", {
            "q_floor": None, "q_ceiling": None, "llm_max_tokens": None,
        }), services)
        self.assertEqual(report["config_errors"], [])
        with self.assertRaises(ConfigurationError):
            MemQMemory(MethodConfig("memq", {"llm_model": "conflicting-explicit-model"}), services)
        memory = await self.harness.memory(batch_size=1)
        sample = episode(self.scope, "metered task")
        await memory.learn(sample.result, events=sample.events)
        await memory.command("end_epoch", {})
        self.assertEqual({record.resource_kind for record in memory.services.usage.all()}, {"llm", "embedding"})
        self.assertTrue(all(record.status == "success" for record in memory.services.usage.all()))


class BundledSourceTests(unittest.TestCase):
    def test_complete_native_suite_runs_with_historical_source_inaccessible(self) -> None:
        script = """
import importlib.abc, importlib.machinery, io, json, os, sys, unittest
from pathlib import Path
repository = Path(sys.argv[1]).resolve()
forbidden_root = repository / 'plan'
vendor_root = repository / 'methods' / 'memq' / '_vendor'
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
sys.path[:0] = [str(repository / 'test' / 'memq'), str(repository.parent)]
import check_memq
suite = unittest.defaultTestLoader.loadTestsFromTestCase(check_memq.NativeMemQTests)
output = io.StringIO()
result = unittest.TextTestRunner(stream=output, verbosity=0).run(suite)
modules = {name: str(Path(module.__file__).resolve()) for name, module in sys.modules.items()
           if (name == 'memq' or name.startswith('memq.')) and getattr(module, '__file__', None)}
assert 'memq.service.memory_service' in modules, modules
assert all(Path(path).is_relative_to(vendor_root) for path in modules.values()), modules
report = check_memq.PLUGIN.healthcheck(check_memq.MethodConfig('memq'),
    check_memq.MethodServices(llm=object(), embedder=object()))
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
        process = subprocess.run([sys.executable, "-c", script, str(REPOSITORY_ROOT)],
                                 capture_output=True, text=True, timeout=60, check=False)
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
        result = json.loads(process.stdout.splitlines()[-1])
        self.assertEqual(result["tests_run"],
                         unittest.defaultTestLoader.loadTestsFromTestCase(NativeMemQTests).countTestCases())
        self.assertGreater(result["native_module_count"], 0)
        self.assertTrue(result["all_native_modules_vendored"])
        self.assertTrue(result["source_hashes_verified"])
        self.assertTrue(result["historical_source_blocked"])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s", force=True)
    unittest.main(verbosity=2)
