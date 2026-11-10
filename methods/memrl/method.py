from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import shutil
import tempfile
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, TypedDict

from mematlas.core.config import MethodConfig
from mematlas.core.errors import ConfigurationError
from mematlas.core.models import MemoryContribution, MemoryEpisode, MemoryEvent, Scope, content_text
from mematlas.core.contributions import applied_sources
from mematlas.core.scope import normalize_scope
from mematlas.core.plugin import BaseMemoryMethod, MethodContext, MethodPlugin, MethodServices, call_maybe_async
from mematlas.methods.services import prepare_services
from mematlas.methods.source import inspect_source

from .core import (MemRLSettings, bundled_source_root, configured_providers,
                   create_upstream_service, resolve_settings, upstream_user_id)
from .config import _DESCRIPTOR as DESCRIPTOR


LOGGER = logging.getLogger("mematlas.methods.memrl.adapter")


class SearchOptions(TypedDict, total=False):
    """单次检索的可选参数，放入 retrieve(..., extensions={"memrl": ...})。

    Attributes:
        task_type: 可选任务类型；存在对应类型门槛时选用，否则沿用默认门槛。

    也可直接填写 {"task_type": "os"}，普通检索省略 extensions。
    """

    task_type: str


class AddOptions(TypedDict, total=False):
    """单次学习的可选参数，放入 MemoryUnit.extensions["memrl"]。

    Attributes:
        task_id: 可选任务编号，用于追踪；原生去重策略启用时参与去重。
        task_type: 可选原生任务类型，用于学习元数据。
        end_of_training: 当前经历结束本训练段时填 True；吸收后内部处理当前归属的尾批。

    也可直接填写 {"task_id": 123}；公共 task、success 等字段直接填写在经历中。
    普通交互省略 end_of_training，未满批继续积累；保存和关闭不触发学习。
    """

    task_id: str | int
    task_type: str
    end_of_training: bool


def _metadata_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "model_dump"):
        return dict(value.model_dump())
    if hasattr(value, "__dict__"):
        return dict(value.__dict__)
    return {}


def _query_candidates(value: Any) -> list[tuple[str, float]]:
    result = []
    for item in value or ():
        if isinstance(item, Mapping):
            query, score = item.get("query"), item.get("score")
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            query, score = item[0], item[1]
        else:
            continue
        if isinstance(query, str):
            result.append((query, float(score)))
    return result


def _filesystem_path(path: Path) -> Path:
    """Allow native checkpoint trees to exceed Windows' legacy path limit."""
    path = path.resolve()
    if os.name == "nt" and not str(path).startswith("\\\\?\\"):
        value = str(path)
        path = Path("\\\\?\\UNC\\" + value[2:] if value.startswith("\\\\") else "\\\\?\\" + value)
    return path


class MemRLMemory(BaseMemoryMethod):
    def __init__(self, config: MethodConfig, services: MethodServices):
        self.config = config
        self.params = dict(config.params)
        variants = DESCRIPTOR.config_schema.get("x-variants", {})
        self.settings: MemRLSettings = resolve_settings(self.params, variants, services.provider_configs)
        from mematlas.core.provider_config import resolve_providers
        # Keep the runtime-owned service container: Memory instruments its
        # providers after constructing the method and again after setup.
        services.provider_configs = resolve_providers(
            self.settings.values, configured_providers(services.provider_configs))
        self.services = services
        self._instances: dict[str, Any] = {}
        self._scope_by_key: dict[str, Mapping[str, Any]] = {}
        self._pending_batch: dict[str, list[dict[str, Any]]] = {}
        self._last_snapshots: dict[str, Path] = {}
        self._data_paths: set[Path] = set()
        self._injected_instance = services.object("memrl.instance")
        self._service_factory = services.object("memrl.service_factory")
        self._source_root = services.source_root("memrl", str(bundled_source_root()))
        self._logger = LOGGER

    async def setup(self, context: MethodContext) -> None:
        report = PLUGIN.healthcheck(self.config, self.services)
        if not report["ready"]:
            raise ConfigurationError(f"MemRL requirements are not satisfied: {report}")
        if self._injected_instance is None and self._service_factory is None:
            prepare_services(self.settings.values, self.services, method_id="memrl", llm=True, embedding=True)
        self._logger.info("[memrl-adapter] setup profile=%s", self.settings.profile)

    def _scope_key(self, scope: Scope) -> str:
        encoded = json.dumps(normalize_scope(scope), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    async def _service_for_scope(self, scope: Scope) -> Any:
        key = self._scope_key(scope)
        if key in self._instances:
            return self._instances[key]
        if self._injected_instance is not None:
            if self._instances:
                raise ValueError("A single memrl.instance supports only one scope; use memrl.service_factory for multiple scopes")
            service = self._injected_instance
            user_id = str(self.settings.values.get("user_id", "memp_user"))
        else:
            # Native paths contain user_id and second-resolution timestamps.
            # Include this run's identity so fresh runs never reuse a database.
            run_token = hashlib.sha256(str(self.services.artifact_dir.resolve()).encode()).hexdigest()[:12]
            prefix = f"{run_token}_{self.settings.values.get('user_id', 'memrl_user')}"
            user_id = upstream_user_id(scope, prefix)
            if self._service_factory is not None:
                service = await call_maybe_async(
                    self._service_factory,
                    settings=self.settings,
                    services=self.services,
                    user_id=user_id,
                    scope=scope,
                    run_sync_in_thread=True,
                )
            else:
                service = await call_maybe_async(
                    create_upstream_service,
                    self.settings,
                    self.services,
                    user_id=user_id,
                    source_root=self._source_root,
                    run_sync_in_thread=True,
                )
        self._instances[key] = service
        self._scope_by_key[key] = dict(scope)
        self._refresh_data_paths()
        self._logger.info("[memrl-adapter] initialized upstream service user_id=%s", user_id)
        return service

    def _refresh_data_paths(self) -> None:
        paths = {Path(value).expanduser().resolve()
                 for service in self._instances.values()
                 for name in ("_cube_dir", "_qdrant_dir")
                 if (value := getattr(service, name, None))}
        self.services.resource_paths.difference_update(self._data_paths - paths)
        self.services.resource_paths.update(paths)
        self._data_paths = paths

    async def _observe(self, event: MemoryEvent) -> None:
        return None

    async def search(self, query, *, level_ref=None, top_k, scope, extensions) -> MemoryContribution:
        task = content_text(query)
        if not task.strip():
            raise ValueError("MemRL retrieval requires a nonempty text query")
        service = await self._service_for_scope(scope)
        values = self.settings.values
        threshold = self._retrieval_threshold(extensions.get("memrl", {}))
        options = {"k": int(values.get("k_retrieve", 5)), "threshold": threshold}
        query_candidates = []
        if values.get("enable_value_driven", True) and values.get("retrieve_strategy", "query") == "query":
            previous_topk = service.rl_config.topk
            try:
                if top_k is not None:
                    service.rl_config.topk = top_k
                raw = await call_maybe_async(service.retrieve_query, task, **options, run_sync_in_thread=True)
            finally:
                service.rl_config.topk = previous_topk
            result, queries = raw if isinstance(raw, tuple) and len(raw) == 2 else (raw, ())
            selected = list((result or {}).get("selected", []))
            query_candidates = _query_candidates(queries)
        elif values.get("enable_value_driven", True):
            previous_topk = service.rl_config.topk
            try:
                if top_k is not None:
                    service.rl_config.topk = top_k
                result = await call_maybe_async(service.retrieve_value_aware, task, **options, run_sync_in_thread=True)
            finally:
                service.rl_config.topk = previous_topk
            selected = result.get("selected") or []
            selected = [selected] if isinstance(selected, Mapping) else selected
        else:
            selected = await call_maybe_async(service.retrieve, task,
                k=top_k if top_k is not None else int(values.get("topk", 3)), threshold=threshold, run_sync_in_thread=True)
            result = {}
        items = []
        for row in selected:
            metadata = _metadata_dict(row.get("metadata"))
            content = row.get("content") or metadata.get("full_content")
            if not isinstance(content, str) or not content:
                content = getattr(row.get("memory_item"), "memory", "")
            item = {"content": str(content), "memory_type": str(metadata.get("type", "procedure")),
                "source_event_ids": list(metadata.get("source_event_ids", [])),
                "metadata": {**{name: value for name, value in metadata.items() if name != "full_content"},
                    "method_id": "memrl", **{name: row[name] for name in
                        ("q_estimate", "similarity", "similarity_z", "q_z") if name in row}}}
            if row.get("memory_id") is not None:
                item["memory_id"] = str(row["memory_id"])
            score = row.get("score", row.get("similarity"))
            if score is not None:
                item["score"] = float(score)
            items.append(item)
        self._logger.info("[memrl-adapter] retrieve scope=%s queries=%d selected=%d",
                          self._scope_key(scope), len(query_candidates), len(items))
        return MemoryContribution(items=items, extensions={"memrl": {
            "scope": dict(scope), "profile": self.settings.profile,
            "simmax": result.get("simmax", 0.0), "query_candidates": [list(row) for row in query_candidates]}})

    def _retrieval_threshold(self, extensions) -> float:
        task_type = str(extensions.get("task_type", "")).strip().lower()
        return float(self.settings.values.get("sim_threshold_by_task", {}).get(
            task_type, self.settings.values.get("sim_threshold", 0.25)))

    @staticmethod
    def _episode_retrieval(episode: MemoryEpisode):
        memory_ids, queries = [], {}
        for contribution in episode.applied_memory:
            info = contribution.extensions.get("memrl")
            if not isinstance(info, Mapping) or info.get("scope") != episode.scope:
                raise ValueError("MemRL applied_memory must contain its retrieval output for this episode scope")
            for row in applied_sources([contribution]):
                identity = row.get("memory_id")
                if identity is not None and identity not in memory_ids:
                    memory_ids.append(str(identity))
            # Preserve query grouping even if the applied contribution has no
            # selected memories. Never credit a later, unapplied retrieval.
            for query, score in _query_candidates(info.get("query_candidates", [])):
                queries[query] = max(queries.get(query, float("-inf")), score)
        return memory_ids, list(queries.items())

    async def _learn(self, episode: MemoryEpisode, *, observed_event_ids=frozenset()) -> None:
        """Absorb feedback; evolve at a full batch or an explicit training end."""
        if not isinstance(episode.success, bool):
            raise ValueError("MemRL requires a real boolean MemoryEpisode.success")
        if not episode.task or not episode.task.strip():
            raise ValueError("MemRL requires MemoryEpisode.task")
        trajectory = "\n".join(f"{event.role or event.kind.value}: {content_text(event.content)}"
                               for event in episode.events or [] if content_text(event.content))
        if not trajectory.strip():
            raise ValueError("MemRL requires a nonempty episode trajectory")
        metadata = dict(episode.extensions.get("memrl", {}))
        end_of_training = metadata.pop("end_of_training", False)
        if not isinstance(end_of_training, bool):
            raise ValueError("MemRL end_of_training must be a boolean")
        memory_ids, queries = self._episode_retrieval(episode)
        key = self._scope_key(episode.scope)
        await self._service_for_scope(episode.scope)
        metadata.update(success=episode.success, episode_id=episode.episode_id,
                        source_event_ids=[event.event_id for event in episode.events])
        source_benchmark = {"default": "HLE", "hle": "HLE", "hle_episode": "HLE",
                            "alfworld": "alfworld_build", "bcb": "bigcodebench"}
        if self.settings.profile.startswith("llb"):
            task_type = self.settings.profile.removeprefix("llb_")
            if task_type == "llb":
                task_type = str(metadata.get("task_type", "os"))
            source_benchmark[self.settings.profile] = f"llb_{task_type}"
            metadata.setdefault("phase", "train")
            task_id = metadata.get("task_id")
            if task_id is not None:
                metadata["task_id"] = int(task_id) if str(task_id).isdigit() else task_id
                metadata.setdefault("sample_index", metadata["task_id"])
        metadata.setdefault("source_benchmark", source_benchmark.get(self.settings.profile, "mematlas"))
        if self.settings.profile == "bcb":
            metadata.setdefault("outcome", "success" if episode.success else "failure")
            metadata.setdefault("outcome_success", episode.success)
        if self.settings.profile in {"default", "hle", "hle_episode"}:
            metadata["q_value"] = 1.0 if episode.success else 0.0
            metadata.setdefault("image_ids", [])
        self._pending_batch.setdefault(key, []).append({
            "task": episode.task, "trajectory": trajectory, "success": episode.success,
            "memory_ids": memory_ids, "queries": queries, "metadata": metadata})
        if end_of_training or len(self._pending_batch[key]) >= int(self.settings.values.get("batch_size", 32)):
            await self._evolve_batch(key, self._instances[key])
        self._logger.info("[memrl-adapter] learn episode=%s feedback=%s credited=%d pending=%d",
                          episode.episode_id, episode.success, len(memory_ids), len(self._pending_batch.get(key, [])))

    async def _evolve_batch(self, key: str, service: Any) -> None:
        """Build/refine native memories and update their values in profile order."""
        rows = self._pending_batch.get(key, [])
        if not rows:
            return

        async def add():
            result = await call_maybe_async(
                service.add_memories,
                task_descriptions=[row["task"] for row in rows],
                trajectories=[row["trajectory"] for row in rows],
                successes=[row["success"] for row in rows],
                retrieved_memory_queries=[row["queries"] or None for row in rows],
                retrieved_memory_ids_list=[row["memory_ids"] or None for row in rows],
                metadatas=[row["metadata"] for row in rows],
                run_sync_in_thread=True,
            )
            # Native add_memories catches exceptions and may return {}.
            if not isinstance(result, list) or len(result) != len(rows):
                raise RuntimeError("MemRL did not confirm ingestion of the complete batch")
            for row, confirmation in zip(rows, result):
                strategy = self.settings.values.get("update_strategy")
                skipped = not row["success"] and (strategy == "validation" or (
                    strategy == "adjustment" and self.settings.values.get("adjustment_mode") == "inplace" and not row["memory_ids"]))
                if (not isinstance(confirmation, (list, tuple)) or len(confirmation) != 2
                        or confirmation[0] != row["task"][:4096] or confirmation[1] is None and not skipped):
                    raise RuntimeError("MemRL failed to confirm a requested memory update")

        async def update():
            result = await call_maybe_async(service.update_values,
                [float(row["success"]) for row in rows], [row["memory_ids"] for row in rows],
                run_sync_in_thread=True)
            if self.settings.values.get("enable_value_driven", True):
                expected = {identity for row in rows for identity in row["memory_ids"]}
                if not isinstance(result, Mapping) or any(result.get(identity) is None for identity in expected):
                    raise RuntimeError("MemRL did not confirm all requested Q-value updates")

        if self.settings.profile in {"bcb", "alfworld", "llb", "llb_os", "llb_db"}:
            await update()
            await add()
        else:
            await add()
            await update()
        self._pending_batch.pop(key, None)
        self._logger.info("[memrl-adapter] evolved profile=%s batch_size=%d", self.settings.profile, len(rows))

    async def flush(self) -> None:
        from mematlas.core.files import atomic_write_json
        target = self.services.artifact_dir / "memrl" / "adapter_state.json"
        atomic_write_json(target, await self.save())

    @staticmethod
    def _snapshot_payload(root: Path) -> dict[str, Any]:
        # Physical Qdrant/SQLite bytes, dump paths and snapshot timestamps change
        # on reopening. The native cube dump contains the full vectors/payloads;
        # together with native caches it is the logical checkpoint identity.
        files = [root / "cube" / "textual_memory.json", *(
            root / "local_cache" / f"{name}.json"
            for name in ("dict_memory", "query_embeddings", "mem_cache", "q_cache"))]
        if not files[0].is_file():
            raise ValueError("MemRL snapshot is missing the native textual memory dump")
        payload = {path.relative_to(root).as_posix(): json.loads(path.read_text(encoding="utf-8")) for path in files}
        for name, value in list(payload.items()):
            if name.startswith("local_cache/") and isinstance(value, dict):
                # Native retrieval and FIFO cache eviction depend on key order.
                payload[name] = list(value.items())
        memories = payload["cube/textual_memory.json"]
        if isinstance(memories, list) and all(isinstance(row, dict) and "id" in row for row in memories):
            payload["cube/textual_memory.json"] = sorted(memories, key=lambda row: str(row["id"]))
        return payload

    @classmethod
    def _snapshot_digest(cls, root: Path) -> str:
        return hashlib.sha256(json.dumps(cls._snapshot_payload(root), sort_keys=True, ensure_ascii=False,
                                        separators=(",", ":")).encode("utf-8")).hexdigest()

    @classmethod
    def _equivalent_snapshot(cls, captured: Path, previous: Path) -> bool:
        """Allow only float32 cosine re-normalization; retain all payloads and caches."""
        current, prior = cls._snapshot_payload(captured), cls._snapshot_payload(previous)
        current_rows, prior_rows = current["cube/textual_memory.json"], prior["cube/textual_memory.json"]
        if len(current_rows) != len(prior_rows):
            return False
        for row, old_row in zip(current_rows, prior_rows):
            vector, old_vector = row.pop("vector", None), old_row.pop("vector", None)
            if vector == old_vector:
                continue
            if (not isinstance(vector, list) or not isinstance(old_vector, list)
                    or len(vector) != len(old_vector)):
                return False
            # Qdrant cosine collections normalize float32 vectors again on load.
            # Reuse the immutable source only within four float32 epsilons per
            # coordinate; genuine vector changes still create a new snapshot.
            if not all(isinstance(a, (int, float)) and isinstance(b, (int, float))
                       and math.isfinite(a) and math.isfinite(b)
                       and math.isclose(a, b, rel_tol=4 * 2 ** -23, abs_tol=0.0)
                       for a, b in zip(vector, old_vector)):
                return False
        return (json.dumps(current, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
                == json.dumps(prior, sort_keys=True, ensure_ascii=False, separators=(",", ":")))

    async def save(self) -> Mapping[str, Any]:
        entries = {}
        for key, service in self._instances.items():
            save = getattr(service, "save_checkpoint_snapshot", None)
            if not callable(save):
                raise NotImplementedError("This MemRL service does not expose native checkpoint snapshots")
            scope_token = key[:24]
            target = self.services.register_resource(self.services.artifact_dir / "memrl" / "snapshots" / scope_token)
            target.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix="capture-", dir=target) as scratch:
                await call_maybe_async(save, scratch, "0", run_sync_in_thread=True)
                captured = _filesystem_path(Path(scratch) / "snapshot" / "0")
                digest = self._snapshot_digest(captured)
                previous = self._last_snapshots.get(key)
                if (previous is not None and previous.is_dir()
                        and self._snapshot_digest(previous) == previous.name
                        and (digest == previous.name or self._equivalent_snapshot(captured, previous))):
                    frozen = previous
                else:
                    frozen = _filesystem_path(target / digest)
                    if not frozen.exists():
                        captured.rename(frozen)
                self._last_snapshots[key] = frozen
            entries[key] = {
                "scope": self._scope_by_key.get(key),
                "snapshot_root": str(frozen),
            }
        return deepcopy({
            "schema_version": 3,
            "services": entries,
            "pending_batch": self._pending_batch,
        })

    async def load(self, state: Mapping[str, Any]) -> None:
        if state.get("schema_version") != 3 or not isinstance(state.get("services"), Mapping):
            raise ValueError("Invalid MemRL snapshot reference")
        validated = []
        for key, value in state["services"].items():
            if not isinstance(value, Mapping) or not isinstance(value.get("scope"), Mapping):
                raise ValueError("MemRL snapshot entry is missing scope metadata")
            scope = normalize_scope(value["scope"])
            if str(key) != self._scope_key(scope):
                raise ValueError("MemRL snapshot scope identity does not match its key")
            snapshot_root = Path(str(value.get("snapshot_root", ""))).expanduser()
            if not snapshot_root.is_dir():
                raise FileNotFoundError(f"MemRL native snapshot was not found: {snapshot_root}")
            if self._snapshot_digest(snapshot_root) != snapshot_root.name:
                raise ValueError("MemRL native snapshot checksum mismatch")
            validated.append((str(key), scope, snapshot_root))
        pending = deepcopy(dict(state["pending_batch"]))
        if set(pending) - set(state["services"]):
            raise ValueError("MemRL pending batch refers to a missing scope")
        restored = {}
        for key, scope, snapshot_root in validated:
            service = await self._service_for_scope(scope)
            load = getattr(service, "load_checkpoint_snapshot", None)
            if not callable(load):
                raise NotImplementedError("This MemRL service does not expose native checkpoint loading")
            working = self.services.artifact_dir / "memrl" / "working"
            working.mkdir(parents=True, exist_ok=True)
            copy_root = _filesystem_path(Path(tempfile.mkdtemp(prefix=f"{key[:16]}-", dir=working)))
            await call_maybe_async(shutil.copytree, snapshot_root, copy_root,
                                  dirs_exist_ok=True, run_sync_in_thread=True)
            meta_path = copy_root / "snapshot_meta.json"
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            meta.update(cube_dir=str(copy_root / "cube"), qdrant_dir=str(copy_root / "qdrant"))
            meta_path.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
            old_cubes = dict(getattr(getattr(service, "mos", None), "mem_cubes", {}))
            await call_maybe_async(load, str(copy_root), run_sync_in_thread=True)
            await self._close_cubes(old_cubes, service)
            restored[key] = service
        for key, service in self._instances.items():
            if key not in restored:
                await self._close_service(service)
        self._instances = restored
        self._refresh_data_paths()
        self._scope_by_key = {
            str(key): dict(value["scope"])
            for key, value in state["services"].items()
        }
        self._pending_batch = pending
        self._last_snapshots = {key: root for key, _, root in validated}

    @staticmethod
    async def _close_cubes(cubes: Mapping[str, Any], service: Any, *, all_cubes: bool = False) -> None:
        active = getattr(service, "default_cube_id", None)
        current = getattr(getattr(service, "mos", None), "mem_cubes", {})
        closed = set()
        for cube_id, cube in cubes.items():
            if not all_cubes and cube is current.get(active):
                continue
            client = getattr(getattr(getattr(cube, "text_mem", None), "vector_db", None), "client", None)
            close = getattr(client, "close", None)
            if callable(close) and id(client) not in closed:
                await call_maybe_async(close, run_sync_in_thread=True)
                closed.add(id(client))
            if current.get(cube_id) is cube:
                current.pop(cube_id, None)

    async def _close_service(self, service: Any) -> None:
        close = getattr(service, "close", None)
        if callable(close):
            await call_maybe_async(close, run_sync_in_thread=True)
        else:
            cubes = dict(getattr(getattr(service, "mos", None), "mem_cubes", {}))
            await self._close_cubes(cubes, service, all_cubes=True)

    async def close(self) -> None:
        for service in self._instances.values():
            await self._close_service(service)
        self._instances.clear()


class MemRLPlugin(MethodPlugin):
    qa_factory = "mematlas.methods.memrl.qa:MemRLQA"
    descriptor = DESCRIPTOR

    def healthcheck(self, config: MethodConfig, services: MethodServices) -> Mapping[str, Any]:
        if self.validate_config(config):
            return super().healthcheck(config, services)
        settings = resolve_settings(config.params, DESCRIPTOR.config_schema["x-variants"], services.provider_configs)
        report = inspect_source(services, method_id="memrl", fallback=bundled_source_root(),
            required_files=("memrl/service/memory_service.py", "memrl/configs/config.py"),
            expected_revision="c1b322ca43de36ddf64c6712f89d0095bfc35ce0",
            required_text={"memrl/service/memory_service.py": (
                'self.vector_dimension: int = int(kwargs.get("vector_dimension", 3072))',
                '\"collection_name\": f\"memp_{self.user_id}_{ts_str}\",\n                    \"vector_dimension\": self.vector_dimension,',
                '\"collection_name\": f\"memp_{self.user_id}_snapshot\",\n                            \"vector_dimension\": self.vector_dimension,',
            )} if int(settings.values.get("embedding_dims") or 3072) != 3072 else None)
        from importlib import metadata
        from mematlas.core.provider_config import resolve_providers
        providers = resolve_providers(settings.values, configured_providers(services.provider_configs))
        resolved = replace(services, source_roots={**services.source_roots, "memrl": report["source_root"]},
                           provider_configs=providers)
        result = {**super().healthcheck(config, resolved), **report}
        result["incompatible_packages"] = []
        if "packaging" not in result["missing_packages"]:
            from packaging.specifiers import SpecifierSet
            for name, constraint in self.descriptor.requirements.get("python_constraints", {}).items():
                if name not in result["missing_packages"]:
                    version = metadata.version(name)
                    if version not in SpecifierSet(constraint):
                        result["incompatible_packages"].append(f"{name}=={version} requires {constraint}")
        injected = services.object("memrl.instance") is not None or services.object("memrl.service_factory") is not None
        result["missing_backend_credentials"] = []
        if injected:
            result["missing_services"] = []
            instance = services.object("memrl.instance")
            if instance is not None:
                retrieval = "retrieve" if not settings.values.get("enable_value_driven", True) else (
                    "retrieve_query" if settings.values.get("retrieve_strategy") == "query" else "retrieve_value_aware")
                required = ("add_memories", "update_values", "save_checkpoint_snapshot", "load_checkpoint_snapshot", retrieval)
                result["config_errors"].extend(f"memrl.instance requires {name}()" for name in required
                    if not callable(getattr(instance, name, None)))
        else:
            config_path = settings.values.get("mos_config_path")
            if config_path:
                if not Path(config_path).expanduser().is_file():
                    result["config_errors"].append(f"MemRL mos_config_path does not exist: {config_path}")
            else:
                result["missing_backend_credentials"] = sorted({provider.api_key_env for provider in providers.values()
                    if not os.environ.get(provider.api_key_env)})
        result["ready"] = not (result["config_errors"] or result["missing_packages"] or result["missing_services"]
            or result["incompatible_packages"] or result["missing_backend_credentials"]) and report["source_ready"]
        result["status"] = "ready" if result["ready"] else "unavailable"
        return result

    def create_method(self, config: MethodConfig, services: MethodServices) -> BaseMemoryMethod:
        return MemRLMemory(config, services)


PLUGIN = MemRLPlugin()
