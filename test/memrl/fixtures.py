"""Offline I/O fixtures; all memory algorithms execute bundled MemRL source.

The normal fixture replaces MOS registration and cube storage.  The Qdrant
fixture replaces only MOS registration and provider factories: GeneralMemCube,
GeneralTextMemory, QdrantVecDB, SQLite, and native snapshot/cache code are real.
"""

from __future__ import annotations

import atexit
import copy
import hashlib
import json
import logging
import os
import sys
import tempfile
import threading
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT.parent))
from mematlas.methods.memrl.core import bundled_source_root

UPSTREAM_ROOT = bundled_source_root()
sys.path.insert(0, str(UPSTREAM_ROOT))

# MemOS configures logging at import time.  Keep its files away from the checkout
# and prevent dotenv from reading local credentials during these offline tests.
_IMPORT_ARTIFACTS = tempfile.TemporaryDirectory(prefix="memrl-test-import-")
atexit.register(_IMPORT_ARTIFACTS.cleanup)
with patch.dict(os.environ, {"MEMOS_BASE_PATH": _IMPORT_ARTIFACTS.name,
                             "PYTHON_DOTENV_DISABLED": "1"}):
    from memos.configs.mem_cube import GeneralMemCubeConfig
    from memos.mem_cube.general import GeneralMemCube
    from memos.memories.textual.item import TextualMemoryItem
    from memrl.configs.config import RLConfig
    from memrl.service.memory_service import MemoryService
    from memrl.service.strategies import StrategyConfiguration
    from memrl.service.value_driven import QValueUpdater

# Close MemOS file handlers before TemporaryDirectory cleanup on Windows.
atexit.register(logging.shutdown)

from mematlas.core.config import MethodConfig, RuntimeConfig
from mematlas.core.models import MemoryContribution, MemoryEpisode, MemoryEvent, Scope
from mematlas.core.plugin import MethodContext, MethodServices
from mematlas.methods.memrl.method import MemRLMemory


VECTOR_DIMENSION = 3072


class OfflineLLM:
    """Deterministic provider response, leaving native prompt/build logic intact."""

    def generate(self, messages: Any, **kwargs: Any) -> str:
        return "Check the failed step before repeating the procedure."

    def generate_script(self, trajectory: str, **kwargs: Any) -> str:
        return "Perform the recorded steps and verify the result."


class OfflineEmbedder:
    """A fixed vector makes similarity deterministic without network requests."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0] + [0.0] * (VECTOR_DIMENSION - 1) for _ in texts]


def cube_config(user_id: str, qdrant_dir: Path) -> GeneralMemCubeConfig:
    return GeneralMemCubeConfig(
        user_id=user_id,
        text_mem={
            "backend": "general_text",
            "config": {
                "extractor_llm": {
                    "backend": "openai",
                    "config": {"model_name_or_path": "offline-test", "api_key": "offline-test",
                               "api_base": "http://127.0.0.1:1/v1"},
                },
                "embedder": {
                    "backend": "universal_api",
                    "config": {"provider": "openai", "model_name_or_path": "offline-test",
                               "api_key": "offline-test", "base_url": "http://127.0.0.1:1/v1"},
                },
                "vector_db": {
                    "backend": "qdrant",
                    "config": {"collection_name": f"memp_{user_id}_offline", "path": str(qdrant_dir),
                               "vector_dimension": VECTOR_DIMENSION, "distance_metric": "cosine"},
                },
            },
        },
        act_mem={"backend": "uninitialized", "config": {}},
        para_mem={"backend": "uninitialized", "config": {}},
    )


class FileClient:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class FileTextMemory:
    """Only the database API; Q updates/builds/ranking remain native."""

    def __init__(self, qdrant_dir: Path, items: list[dict[str, Any]] | None = None) -> None:
        self.qdrant_dir = qdrant_dir
        self.qdrant_dir.mkdir(parents=True, exist_ok=True)
        self.items = {str(row["id"]): TextualMemoryItem(**row) for row in (items or [])}
        self.vector_db = SimpleNamespace(client=FileClient())
        self._persist()

    def _persist(self) -> None:
        payload = [item.model_dump(mode="json") for item in self.items.values()]
        (self.qdrant_dir / "database.json").write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    def add(self, items: list[TextualMemoryItem]) -> None:
        for item in items:
            self.items[str(item.id)] = item.model_copy(deep=True)
        self._persist()

    def get(self, memory_id: str) -> TextualMemoryItem:
        return self.items[memory_id].model_copy(deep=True)

    def get_all(self) -> list[TextualMemoryItem]:
        return [item.model_copy(deep=True) for item in self.items.values()]

    def update(self, memory_id: str, payload: dict[str, Any]) -> None:
        self.items[memory_id] = TextualMemoryItem(**copy.deepcopy(payload))
        self._persist()


class FileCube:
    def __init__(self, config: GeneralMemCubeConfig, items: list[dict[str, Any]] | None = None) -> None:
        self.config = config
        path = Path(config.text_mem.config.vector_db.config.path)
        self.text_mem = FileTextMemory(path, items)

    def dump(self, directory: str) -> None:
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        (target / "config.json").write_text(self.config.model_dump_json(), encoding="utf-8")
        payload = [item.model_dump(mode="json") for item in self.text_mem.get_all()]
        (target / "textual_memory.json").write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    @classmethod
    def init_from_dir(cls, directory: str, *, default_config: GeneralMemCubeConfig) -> "FileCube":
        payload = json.loads((Path(directory) / "textual_memory.json").read_text(encoding="utf-8"))
        return cls(default_config, payload)


class OfflineMOS:
    """An isolated cube registry; native GeneralTextMemory can be used unchanged."""

    def __init__(self, cube_id: str, cube: Any) -> None:
        self.mem_cubes = {cube_id: cube}
        self.user_manager = SimpleNamespace(get_user_cubes=lambda user_id: [
            SimpleNamespace(cube_id=identity) for identity in self.mem_cubes])

    def search(self, *, query: str, top_k: int, **kwargs: Any) -> dict[str, Any]:
        return {"text_mem": [{"memories": cube.text_mem.search(query, top_k)} for cube in self.mem_cubes.values()]}

    def get(self, *, mem_cube_id: str, memory_id: str, **kwargs: Any) -> TextualMemoryItem:
        return self.mem_cubes[mem_cube_id].text_mem.get(memory_id)

    def get_all(self, *, mem_cube_id: str | None = None, **kwargs: Any) -> dict[str, Any]:
        cubes = [self.mem_cubes[mem_cube_id]] if mem_cube_id else self.mem_cubes.values()
        return {"text_mem": [{"memories": cube.text_mem.get_all()} for cube in cubes]}

    def register_mem_cube(self, cube: Any, *, mem_cube_id: str, **kwargs: Any) -> None:
        self.mem_cubes[mem_cube_id] = cube

    def unregister_mem_cube(self, mem_cube_id: str, **kwargs: Any) -> None:
        self.mem_cubes.pop(mem_cube_id, None)


class RecordingNativeService(MemoryService):
    """Record calls before dispatching directly to the original algorithms."""

    def add_memories(self, *args: Any, **kwargs: Any) -> Any:
        tasks = kwargs.get("task_descriptions", args[0] if args else [])
        self.calls.append(("add_memories", len(tasks)))
        return super().add_memories(*args, **kwargs)

    def update_values(self, successes: list[float], retrieved_ids_list: list[list[str]]) -> Any:
        self.calls.append(("update_values", len(successes)))
        return super().update_values(successes, retrieved_ids_list)

    def text_memory(self) -> Any:
        return self.mos.mem_cubes[self.default_cube_id].text_mem


class NativeHarness:
    def __init__(self, root: Path, *, real_qdrant: bool = False) -> None:
        self.root = root
        self.real_qdrant = real_qdrant
        self.stack = ExitStack()
        self.created: list[tuple[Scope, RecordingNativeService]] = []

    def __enter__(self) -> "NativeHarness":
        # Patch only provider construction in the real Qdrant tests.
        self.stack.enter_context(patch("memos.llms.factory.LLMFactory.from_config", return_value=OfflineLLM()))
        self.stack.enter_context(patch("memos.embedders.factory.EmbedderFactory.from_config", return_value=OfflineEmbedder()))
        if not self.real_qdrant:
            self.stack.enter_context(patch("memrl.service.memory_service.GeneralMemCube.init_from_dir",
                                           side_effect=FileCube.init_from_dir))
        return self

    def __exit__(self, *exc: Any) -> None:
        # Also close clients belonging to tests that failed before adapter.close.
        seen: set[int] = set()
        for _, service in self.created:
            for cube in service.mos.mem_cubes.values():
                client = cube.text_mem.vector_db.client
                if id(client) not in seen:
                    client.close()
                    seen.add(id(client))
        self.stack.close()

    def factory(self, *, settings: Any, services: Any, user_id: str, scope: Scope) -> RecordingNativeService:
        service = RecordingNativeService.__new__(RecordingNativeService)
        directory = self.root / f"native-{len(self.created)}"
        qdrant_dir = directory / "qdrant"
        config = cube_config(user_id, qdrant_dir)
        cube = GeneralMemCube(config) if self.real_qdrant else FileCube(config)
        service.default_cube_id = f"cube_{len(self.created)}"
        service.mos = OfflineMOS(service.default_cube_id, cube)
        service.user_id = user_id
        service.num_workers = 1
        service.vector_dimension = VECTOR_DIMENSION
        service._qdrant_dir = str(qdrant_dir)
        service._cube_dir = str(directory / "cube")
        service.calls = []
        values = settings.values
        service.strategy_config = StrategyConfiguration.from_strings(
            values["build_strategy"], values["retrieve_strategy"], values["update_strategy"])
        service.llm_provider = OfflineLLM()
        service.embedding_provider = OfflineEmbedder()
        service.max_keywords = int(values["max_keywords"])
        service._init_key_generators()
        service.memory_confidence = float(values["memory_confidence"])
        service.add_similarity_threshold = float(values["add_similarity_threshold"])
        service.adjustment_mode = str(values["adjustment_mode"])
        service.enable_value_driven = bool(values["enable_value_driven"])
        service.rl_config = RLConfig(**{name: values[name] for name in RLConfig.model_fields if name in values})
        service._q_updater = QValueUpdater(service.mos, user_id, service.rl_config,
                                          default_cube_id=service.default_cube_id)
        service._q_cache = {}
        service._q_cache_max_size = 10000
        service._mem_cache = {}
        service._mem_cache_max_size = 10000
        service._db_gate = threading.BoundedSemaphore(1)
        service.dict_memory = {}
        service.query_embeddings = {}
        service.weight_sim = float(values["weight_sim"])
        service.weight_q = float(values["weight_q"])
        service.sim_norm_mean = float(values["sim_norm_mean"])
        service.sim_norm_std = float(values["sim_norm_std"]) or 1.0
        service.use_z_score_normalization = bool(values["use_z_score_normalization"])
        service.dedup_by_task_id = bool(values["dedup_by_task_id"])
        service.mos_config = SimpleNamespace(
            chat_model=config.text_mem.config.extractor_llm,
            mem_reader=SimpleNamespace(config=SimpleNamespace(embedder=config.text_mem.config.embedder)),
        )
        self.created.append((scope, service))
        return service

    def latest(self, scope: Scope) -> RecordingNativeService:
        return next(service for saved_scope, service in reversed(self.created) if saved_scope == scope)

    def adapter(self, **params: Any) -> MemRLMemory:
        return MemRLMemory(MethodConfig("memrl", params), MethodServices(
            objects={"memrl.service_factory": self.factory}, artifact_dir=self.root / "artifacts"))


async def setup(adapter: MemRLMemory) -> None:
    await adapter.setup(MethodContext("offline-test", MethodConfig("memrl", adapter.params), RuntimeConfig()))


def episode(scope: Scope, task: str, *, success: bool = True,
            applied_memory=()) -> MemoryEpisode:
    return MemoryEpisode(episode_id=task, task=task, scope=scope, success=success,
        events=[MemoryEvent(kind="message", scope=scope, role="assistant", episode_id=task,
                           content=f"Execute task {task}; verify the observed result.")],
        applied_memory=list(applied_memory))


def snapshot_path(state: dict[str, Any]) -> Path:
    return Path(next(iter(state["services"].values()))["snapshot_root"])


def directory_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_file():
            digest.update(str(path.relative_to(root)).encode("utf-8"))
            digest.update(path.read_bytes())
    return digest.hexdigest()
