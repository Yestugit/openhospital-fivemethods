from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import logging
from pathlib import Path
import random
import shutil
from typing import Any, Mapping

from mematlas.core.config import MethodConfig
from mematlas.core.models import (
    ContentBlock,
    DecisionContext,
    MemoryContribution,
    MemoryEpisode,
    MemoryEvent,
    MemoryItem,
    MemoryScope,
)
from mematlas.core.plugin import (
    BaseMemoryMethod,
    MethodContext,
    MethodPlugin,
    MethodServices,
    call_maybe_async,
)
from mematlas.methods.manifest import load_method_descriptor
from mematlas.methods.services import prepare_services
from mematlas.methods.source import inspect_source

from .core import (
    MemQSettings,
    bundled_source_root,
    configured_providers,
    create_upstream_service,
    load_upstream,
    resolve_settings,
    scope_key,
)
from .state import capture_service, decode_state, encode_state, restore_service


DESCRIPTOR = load_method_descriptor(__file__)
LOGGER = logging.getLogger("mematlas.methods.memq.adapter")


class MemQMemory(BaseMemoryMethod):
    def __init__(self, config: MethodConfig, services: MethodServices):
        self.params = dict(config.params)
        variants = DESCRIPTOR.config_schema.get("x-variants", {})
        self.settings: MemQSettings = resolve_settings(self.params, variants, services.provider_configs)
        from mematlas.core.provider_config import resolve_providers
        # The runtime instruments this service container after construction.
        services.provider_configs = resolve_providers(
            self.settings.values, configured_providers(services.provider_configs))
        self.services = services
        self._source_root = services.source_root("memq", str(bundled_source_root()))
        self._instances: dict[str, Any] = {}
        self._scopes: dict[str, MemoryScope] = {}
        self._dags: dict[str, OrderedDict[str, list[str]]] = {}
        self._pending_edges: dict[str, list[dict[str, Any]]] = {}
        self._last_retrievals: dict[str, dict[str, Any]] = {}
        self._pending_batch: dict[str, list[dict[str, Any]]] = {}
        self._deferred_values: dict[str, list[dict[str, Any]]] = {}
        self._batch_open = False
        self._epoch_open = False
        self._current_epoch = 0
        self._total_epochs = int(self.settings.values.get("num_epochs", 5))
        self._storage_root = Path(
            str(self.settings.values.get("storage_path") or (services.artifact_dir / "memq"))
        ).expanduser().resolve()
        self._logger = LOGGER

    async def setup(self, context: MethodContext) -> None:
        prepare_services(self.settings.values, self.services, method_id="memq", llm=True, embedding=True)
        if self.services.llm is None or self.services.embedder is None:
            raise ValueError("MemQ requires both framework LLM and embedding services")
        await call_maybe_async(load_upstream, self._source_root, run_sync_in_thread=True)
        state_path = self._storage_root / "adapter_state.json"
        if state_path.is_file():
            await self.load_state(json.loads(state_path.read_text(encoding="utf-8")))
        self._logger.info(
            "[memq-adapter] setup profile=%s update_mode=%s source=%s",
            self.settings.profile,
            self.settings.values.get("memq_update_mode", "epoch"),
            self._source_root,
        )

    def _scope_dir(self, key: str) -> Path:
        return self._storage_root / "scopes" / key[:24]

    async def _service_for_scope(self, scope: MemoryScope) -> tuple[str, Any]:
        key = scope_key(scope)
        self._scopes[key] = scope
        if key in self._instances:
            return key, self._instances[key]
        service = await call_maybe_async(
            create_upstream_service,
            self.settings,
            self.services,
            scope,
            source_root=self._source_root,
            run_sync_in_thread=True,
        )
        self._instances[key] = service
        self._dags[key] = OrderedDict()
        self._pending_edges[key] = []
        self._pending_batch[key] = []
        self._deferred_values[key] = []
        self._last_retrievals.pop(key, None)
        service.set_memid_pair(self._dags[key])
        service.set_epoch_info(self._current_epoch, self._total_epochs)
        self._logger.info(
            "[memq-adapter] initialized scope=%s memories=%d",
            key[:16],
            len(service._store),
        )
        return key, service

    async def observe(self, event: MemoryEvent) -> None:
        self._scopes[scope_key(event.scope)] = event.scope

    async def retrieve(self, context: DecisionContext) -> MemoryContribution:
        if context.metadata.get("read_only"):
            state = await self.save_state()
            try:
                return await self._retrieve_native(context)
            finally:
                await self.load_state(state)
        return await self._retrieve_native(context)

    async def _retrieve_native(self, context: DecisionContext) -> MemoryContribution:
        key, service = await self._service_for_scope(context.scope)
        values = self.settings.values
        native = await call_maybe_async(
            service.retrieve_query,
            context.text,
            k=int(values.get("k_retrieve", 10)),
            threshold=float(values.get("sim_threshold", 0.0)),
            run_sync_in_thread=True,
        )
        result, query_scores = native if isinstance(native, tuple) else (native, [])
        selected = list(result.get("selected", []))
        native_count = len(selected)
        if context.top_k is not None:
            selected = selected[:max(0, int(context.top_k))]

        rows = []
        selected_ids = []
        for candidate in selected:
            memory_id = candidate.get("memory_id")
            if not memory_id:
                continue
            selected_ids.append(str(memory_id))
            metadata = candidate.get("metadata")
            metadata = dict(metadata) if isinstance(metadata, Mapping) else {}
            metadata.update({
                "method_id": "memq",
                "q_estimate": candidate.get("q_estimate"),
                "q_normalized": candidate.get("q_z"),
                "native_score": candidate.get("score"),
                "memq_query_candidates": [[str(query), float(score)] for query, score in query_scores],
            })
            content = candidate.get("content")
            if not content:
                native_item = candidate.get("memory_item")
                content = native_item.get("memory", "") if isinstance(native_item, Mapping) else ""
            rows.append(MemoryItem(
                content=(ContentBlock.text(str(content)),),
                memory_id=str(memory_id),
                score=float(candidate.get("score", candidate.get("similarity", 0.0))),
                memory_type="procedural",
                metadata=metadata,
            ))

        self._last_retrievals[key] = {
            "ids": selected_ids,
            "queries": [(str(query), float(score)) for query, score in query_scores],
        }
        self._logger.info(
            "[memq-adapter] retrieve scope=%s query_candidates=%d selected=%d returned=%d",
            key[:16], len(query_scores), native_count, len(rows),
        )
        return MemoryContribution.from_items(rows, metadata={
            "method_id": "memq",
            "profile": self.settings.profile,
            "native_selected_count": native_count,
            "query_candidates": len(query_scores),
            "q_values_normalized": bool(values.get("normalize_q", True)),
            "implementation_mode": values.get("implementation_mode", "standard"),
        })

    @staticmethod
    def _episode_task(episode: MemoryEpisode) -> str:
        values = episode.result.metadata
        task = (values.get("task") or values.get("task_description")
                or episode.metadata.get("task") or episode.metadata.get("task_description"))
        if not isinstance(task, str) or not task.strip():
            raise ValueError("MemQ requires task text in EpisodeOutcome.metadata['task']")
        return task.strip()

    @staticmethod
    def _trajectory(episode: MemoryEpisode) -> str:
        lines = [
            f"{event.role or event.kind.value}: {event.text}"
            for event in episode.events
            if event.text.strip()
        ]
        trajectory = "\n".join(lines)
        if not trajectory.strip():
            raise ValueError("MemQ requires a nonempty episode trajectory")
        return trajectory

    @staticmethod
    def _episode_images(episode: MemoryEpisode) -> list[dict[str, str]]:
        images = []
        for event in episode.events:
            for block in event.content:
                if block.modality.value != "image":
                    continue
                uri = block.uri or block.data
                if uri:
                    images.append({"url": str(uri)})
        return images

    @staticmethod
    def _episode_retrieval(episode: MemoryEpisode) -> tuple[list[str], list[tuple[str, float]]]:
        memory_ids: list[str] = []
        queries: list[tuple[str, float]] = []
        for row in episode.retrieved:
            identifier = row.get("memory_id") or row.get("id")
            if identifier:
                memory_ids.append(str(identifier))
            metadata = row.get("metadata")
            if isinstance(metadata, Mapping):
                for candidate in metadata.get("memq_query_candidates", ()):
                    if isinstance(candidate, (list, tuple)) and len(candidate) >= 2:
                        pair = (str(candidate[0]), float(candidate[1]))
                        if pair not in queries:
                            queries.append(pair)
        return memory_ids, queries

    async def learn(self, episode: MemoryEpisode) -> None:
        if not isinstance(episode.result.success, bool):
            raise ValueError("MemQ requires an external boolean EpisodeOutcome.success")
        task = self._episode_task(episode)
        trajectory = self._trajectory(episode)
        images = self._episode_images(episode)
        key, service = await self._service_for_scope(episode.scope)
        retrieved_ids, queries = self._episode_retrieval(episode)
        metadata = dict(episode.metadata)
        metadata.update(dict(episode.result.metadata))
        metadata.update(episode_id=episode.episode_id, success=episode.result.success)
        benchmarks = {"default": "livecodebench", "lcb": "livecodebench",
                      "llb": f"llb_{metadata.get('task_type', 'os')}", "bfcl": "BFCL", "gpqa": "gpqa",
                      "dsbench": "dsbench_data_analysis", "erqa": "erqa", "mmmu": "mmmu_pro"}
        metadata.setdefault("source_benchmark", benchmarks[self.settings.profile])
        metadata.setdefault("phase", "train")
        epoch_field = "lb_epoch" if self.settings.profile in {"llb", "gpqa"} else "epoch"
        metadata.setdefault(epoch_field, self._current_epoch + 1)
        if metadata.get("task_id") is not None:
            metadata.setdefault("sample_index", metadata["task_id"])
        # The official runners supply fixed success-dependent initialization.
        # Native add_memories still owns its optional adaptive-Q override.
        metadata["q_value"] = float(self.settings.values[
            "q_init_pos" if episode.result.success else "q_init_neg"])
        self._pending_batch.setdefault(key, []).append({
            "task": task, "trajectory": trajectory, "success": episode.result.success,
            "memory_ids": retrieved_ids, "queries": queries,
            "metadata": metadata, "images": images,
        })
        self._last_retrievals.pop(key, None)
        self._epoch_open = True
        if not self._batch_open and len(self._pending_batch[key]) >= int(self.settings.values.get("batch_size", 50)):
            await self._flush_batch(key, service)
        self._logger.info(
            "[memq-adapter] learn episode=%s success=%s credited=%d pending_batch=%d",
            episode.episode_id, episode.result.success, len(retrieved_ids), len(self._pending_batch[key]),
        )

    def _append_provenance(self, key: str, child: str, parents: list[str]) -> None:
        dag = self._dags[key]
        if self.settings.profile == "dsbench":
            # DSBench's runner does not use the FIFO helper.
            dag.setdefault(child, []).extend(parents)
            return
        if child not in dag:
            if len(dag) >= 10000:
                dag.popitem(last=False)
            dag[child] = []
        dag[child].extend(parents)
        dag.move_to_end(child)

    async def _flush_batch(self, key: str, service: Any) -> None:
        rows = self._pending_batch.setdefault(key, [])
        if not rows:
            return
        successes = [row["success"] for row in rows]
        retrieved_ids = [row["memory_ids"] for row in rows]
        if self.settings.profile == "gpqa":
            # GPQA's runner gates the single-step stage on `"memrl" in
            # algorithm`, so it does not execute it for algorithm="memq".
            pass
        elif self.settings.values.get("memrl_update_mode", "batch") == "batch":
            await call_maybe_async(service.update_values, successes, retrieved_ids, run_sync_in_thread=True)
            await call_maybe_async(service._update_q_max, run_sync_in_thread=True)
        elif self.settings.profile != "bfcl":
            # BFCL's native `"memrl" in algorithm` condition excludes MemQ
            # from deferred single-step updates; retain that source behavior.
            self._deferred_values[key].extend(
                {"success": row["success"], "memory_ids": list(row["memory_ids"])} for row in rows)
        await call_maybe_async(service.decay_traces, run_sync_in_thread=True)
        await call_maybe_async(service.record_retrieval_traces, retrieved_ids, run_sync_in_thread=True)
        edges = [{
            "reward": float(self.settings.values["success_reward" if row["success"] else "failure_reward"]),
            "retrieved_ids": list(row["memory_ids"]), "new_mem_id": None,
        } for row in rows]
        self._pending_edges[key].extend(edges)
        result = await call_maybe_async(
            service.add_memories,
            task_descriptions=[row["task"] for row in rows],
            trajectories=[row["trajectory"] for row in rows],
            successes=successes,
            retrieved_memory_queries=[row["queries"] for row in rows],
            retrieved_memory_ids_list=retrieved_ids,
            metadatas=[row["metadata"] for row in rows],
            images_list=[row["images"] for row in rows],
            run_sync_in_thread=True,
        )
        if self.settings.profile == "dsbench":
            # This is DSBenchRunner._new_memory_ids' native result alignment.
            if isinstance(result, Mapping):
                ids = [result.get(row["task"]) for row in rows]
            else:
                ids = [item[1] if isinstance(item, (tuple, list)) and len(item) >= 2 else None
                       for item in result or ()]
            for index, identifier in enumerate((ids + [None] * len(rows))[:len(rows)]):
                if identifier:
                    child = str(identifier)
                    if retrieved_ids[index]:
                        self._append_provenance(key, child, retrieved_ids[index])
                    edges[index]["new_mem_id"] = child
        else:
            # Keep the official runners' result-list order and edge backfill
            # offset, including their behavior when native preparation fails.
            pairs = list(result) if isinstance(result, (tuple, list)) else []
            offset = len(self._pending_edges[key]) - len(pairs)
            for index, pair in enumerate(pairs):
                _, identifier = pair
                if identifier:
                    child = str(identifier)
                    if retrieved_ids[index]:
                        self._append_provenance(key, child, retrieved_ids[index])
                    if 0 <= offset + index < len(self._pending_edges[key]):
                        self._pending_edges[key][offset + index]["new_mem_id"] = child
        self._pending_batch[key] = []
        # BFCL's runner performs MemQ's backward pass at epoch end even when
        # memq_update_mode is set to batch.
        if self.settings.values.get("memq_update_mode", "epoch") == "batch" and self.settings.profile != "bfcl":
            await self._flush_bellman(key, service, boundary="batch")
        self._logger.info("[memq-adapter] batch_flushed scope=%s size=%d", key[:16], len(rows))

    async def _flush_bellman(self, key: str, service: Any, *, boundary: str) -> None:
        edges = self._pending_edges.setdefault(key, [])
        if edges:
            await call_maybe_async(
                service.compute_bellman_td_and_accumulate,
                list(edges),
                self._dags[key],
                run_sync_in_thread=True,
            )
            edges.clear()
        await call_maybe_async(service.flush_memq_updates, run_sync_in_thread=True)
        if boundary == "batch" and self.settings.profile not in {"dsbench", "gpqa"}:
            await call_maybe_async(service._update_q_max, run_sync_in_thread=True)
            await call_maybe_async(
                service.reset_traces,
                keep_eligibility=True,
                run_sync_in_thread=True,
            )
        else:
            await call_maybe_async(
                service.reset_traces,
                keep_eligibility=True,
                run_sync_in_thread=True,
            )
            if boundary == "epoch" or self.settings.profile != "gpqa":
                await call_maybe_async(service._update_q_max, run_sync_in_thread=True)

    async def command(self, name: str, payload: Mapping[str, Any]) -> Any:
        if name in {"begin_epoch", "set_epoch_info"}:
            current = payload.get("current_epoch", self._current_epoch)
            total = payload.get("total_epochs", self._total_epochs)
            if (type(current) is not int or type(total) is not int
                    or total < 1 or not 0 <= current < total):
                raise ValueError("MemQ epoch indices require 0 <= current_epoch < total_epochs")
            if name == "begin_epoch" and self._epoch_open:
                raise ValueError("A MemQ epoch is already open; call end_epoch first")
            if (current, total) != (self._current_epoch, self._total_epochs) and (
                any(self._pending_batch.values()) or any(self._pending_edges.values()) or any(self._deferred_values.values())
            ):
                raise ValueError("Finish pending MemQ epoch updates before changing epoch information")
            self._current_epoch, self._total_epochs = current, total
            for service in self._instances.values():
                await call_maybe_async(service.set_epoch_info, current, total, run_sync_in_thread=True)
            if name == "begin_epoch":
                self._epoch_open = True
            return {"current_epoch": current, "total_epochs": total, "epoch_open": self._epoch_open}
        if name == "begin_batch":
            if self._batch_open:
                raise ValueError("A MemQ batch is already open")
            self._batch_open = True
            return {"batch_open": True}
        if name == "end_batch":
            for key in list(self._pending_batch):
                await self._flush_batch(key, self._instances[key])
            self._batch_open = False
            return {"batch_open": False}
        if name == "end_epoch":
            if not self._epoch_open and not any(self._pending_batch.values()):
                return {"epoch_open": False, "current_epoch": self._current_epoch}
            for key, service in self._instances.items():
                await self._flush_batch(key, service)
                deferred = self._deferred_values[key]
                if deferred:
                    await call_maybe_async(service.update_values,
                        [row["success"] for row in deferred], [row["memory_ids"] for row in deferred],
                        run_sync_in_thread=True)
                    await call_maybe_async(service._update_q_max, run_sync_in_thread=True)
                    deferred.clear()
                if self.settings.values.get("memq_update_mode", "epoch") == "epoch" or self.settings.profile == "bfcl":
                    await self._flush_bellman(key, service, boundary="epoch")
                elif self.settings.profile == "gpqa":
                    await call_maybe_async(service._update_q_max, run_sync_in_thread=True)
            self._batch_open = False
            self._epoch_open = False
            return {"epoch_open": False, "current_epoch": self._current_epoch}
        return await super().command(name, payload)

    async def commit(self) -> None:
        # Runtime commits after each episode. Persisting pending work must not
        # turn that scheduling hook into an upstream batch/epoch boundary.
        from mematlas.methods.storage import atomic_write_json

        state = await self.save_state()
        for key, service in self._instances.items():
            native = state["scopes"][key]["native"]
            checkpoint_id = hashlib.sha256(json.dumps(
                native, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")).hexdigest()
            directory = self._scope_dir(key)
            if not (directory / "snapshot" / checkpoint_id / "snapshot_meta.json").is_file():
                await call_maybe_async(service.save_checkpoint_snapshot, str(directory), checkpoint_id,
                                      run_sync_in_thread=True)
            self.services.register_resource(directory)
        atomic_write_json(self.services.register_resource(self._storage_root / "adapter_state.json"), state)
        self._logger.info(
            "[memq-adapter] commit profile=%s scopes=%d",
            self.settings.profile,
            len(self._instances),
        )

    async def save_state(self) -> Mapping[str, Any]:
        scopes = {}
        for key, service in self._instances.items():
            scopes[key] = {
                "scope": self._scopes[key].to_dict(),
                "native": await call_maybe_async(capture_service, service, run_sync_in_thread=True),
            }
        # Typed mappings encode insertion order because native cache ties and
        # the provenance FIFO both observe it. State has no changing paths or
        # checkpoint timestamps and is independently JSON serializable.
        return deepcopy({
            "schema_version": 2, "method_id": "memq", "profile": self.settings.profile,
            "scopes": scopes,
            "observed_scopes": {key: scope.to_dict() for key, scope in self._scopes.items()},
            "adapter": encode_state({
                "dags": self._dags, "pending_edges": self._pending_edges,
                "last_retrievals": self._last_retrievals, "pending_batch": self._pending_batch,
                "deferred_values": self._deferred_values,
                "batch_open": self._batch_open, "epoch_open": self._epoch_open,
                "current_epoch": self._current_epoch, "total_epochs": self._total_epochs,
            }),
            # Native epsilon-greedy uses Python's process-global random module.
            "random_state": encode_state(random.getstate()),
        })

    async def load_state(self, state: Mapping[str, Any]) -> None:
        if state.get("schema_version") != 2 or state.get("method_id") != "memq":
            raise ValueError("Invalid MemQ snapshot schema or method identity")
        scopes = state.get("scopes")
        if not isinstance(scopes, Mapping):
            raise ValueError("Invalid MemQ scope snapshots")
        validated = []
        for key, record in scopes.items():
            if not isinstance(record, Mapping):
                raise ValueError("Invalid MemQ scope snapshot record")
            scope = MemoryScope.from_dict(record.get("scope"))
            if scope_key(scope) != key:
                raise ValueError("MemQ scope identity does not match snapshot key")
            if not isinstance(record.get("native"), Mapping):
                raise ValueError("MemQ native store snapshot is missing")
            validated.append((str(key), scope, record["native"]))
        adapter = decode_state(state["adapter"])
        restored = {}
        for key, scope, native in validated:
            _, service = await self._service_for_scope(scope)
            await call_maybe_async(restore_service, service, native, run_sync_in_thread=True)
            restored[key] = service
        # MemoryStore is entirely in RAM and exposes no close API. Excluded
        # scopes are removed, and future access creates a fresh native service;
        # it never reloads snapshots written after this state was captured.
        self._instances = restored
        self._scopes = {key: MemoryScope.from_dict(value) for key, value in state.get("observed_scopes", {}).items()}
        self._scopes.update({key: scope for key, scope, _ in validated})
        self._dags = adapter["dags"]
        self._pending_edges = adapter["pending_edges"]
        self._last_retrievals = adapter["last_retrievals"]
        self._pending_batch = adapter["pending_batch"]
        self._deferred_values = adapter["deferred_values"]
        self._batch_open = adapter["batch_open"]
        self._epoch_open = adapter["epoch_open"]
        self._current_epoch = adapter["current_epoch"]
        self._total_epochs = adapter["total_epochs"]
        for key, service in restored.items():
            service.set_memid_pair(self._dags[key])
        random.setstate(decode_state(state["random_state"]))

    async def reset(self, scope: MemoryScope | None = None) -> None:
        keys = [scope_key(scope)] if scope is not None else list(self._scopes)
        for key in keys:
            self._instances.pop(key, None)
            self._scopes.pop(key, None)
            self._dags.pop(key, None)
            self._pending_edges.pop(key, None)
            self._last_retrievals.pop(key, None)
            self._pending_batch.pop(key, None)
            self._deferred_values.pop(key, None)
            directory = self._scope_dir(key)
            if directory.is_dir():
                shutil.rmtree(directory)
        if scope is None:
            self._batch_open = False
            self._epoch_open = False
            scopes_dir = self._storage_root / "scopes"
            if scopes_dir.is_dir():
                for directory in scopes_dir.iterdir():
                    if directory.is_dir():
                        shutil.rmtree(directory)
        # Reset also persists the bank selection so reopening cannot resurrect
        # a scope through a previously committed adapter_state.json.
        await self.commit()
        self._logger.info("[memq-adapter] reset scope=%s", scope_key(scope)[:16] if scope else "all")

    async def close(self) -> None:
        self._instances.clear()


class MemQPlugin(MethodPlugin):
    descriptor = DESCRIPTOR

    def learning_schedule(self, config: MethodConfig) -> Mapping[str, Any]:
        settings = resolve_settings(config.params, DESCRIPTOR.config_schema["x-variants"])
        return {"epochs": int(settings.values.get("num_epochs", 5)),
                "epoch_start": "begin_epoch", "epoch_resume": "set_epoch_info",
                "batch_end": "end_batch", "epoch_end": "end_epoch"}

    def validate_config(self, config: MethodConfig) -> list[str]:
        from mematlas.core.validation import validate_parameters

        schema = deepcopy(dict(DESCRIPTOR.config_schema))
        params = dict(config.params)
        for name, rule in schema.get("properties", {}).items():
            types = rule.get("type")
            if isinstance(types, list) and "null" in types:
                if name in params and params[name] is None:
                    params.pop(name)
                rule["type"] = next(value for value in types if value != "null")
        return validate_parameters(params, schema)

    def create_method(self, config: MethodConfig, services: MethodServices) -> BaseMemoryMethod:
        return MemQMemory(config, services)

    def healthcheck(self, config: MethodConfig, services: MethodServices) -> Mapping[str, Any]:
        if self.validate_config(config):
            return super().healthcheck(config, services)
        settings = resolve_settings(config.params, DESCRIPTOR.config_schema["x-variants"], services.provider_configs)
        report = inspect_source(
            services, method_id="memq", fallback=bundled_source_root(),
            required_files=("memq/service/memory_service.py", "memq/service/memory_store.py",
                            "memq/configs/config.py"),
            expected_revision="9d66abf4f0252b084f088d5081c44b840842e47b",
        )
        from mematlas.core.provider_config import resolve_providers

        resolved = replace(services, source_roots={**services.source_roots, "memq": report["source_root"]},
                           provider_configs=resolve_providers(settings.values, configured_providers(services.provider_configs)))
        result = {**super().healthcheck(replace(config, params=dict(settings.values)), resolved), **report}
        if config.params.get("implementation_mode") == "offline":
            missing = []
            if services.llm is None:
                missing.append("llm")
            if services.embedder is None:
                missing.append("embedding")
            if missing:
                result.update(ready=False, status="unavailable", missing_services=missing)
        result["ready"] = bool(result["ready"] and report["source_ready"])
        result["status"] = "ready" if result["ready"] else "unavailable"
        return result


PLUGIN = MemQPlugin()
