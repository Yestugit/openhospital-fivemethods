"""Deterministic provider I/O around the native MemQ service and NumPy store."""

from __future__ import annotations

import atexit
import copy
import hashlib
import os
import sys
import tempfile
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any, Iterator
from unittest.mock import patch


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT.parent))

from mematlas.core.config import MemAtlasConfig, MethodConfig, RuntimeConfig
from mematlas.core.models import (
    ContentBlock, DecisionContext, EpisodeOutcome, EventKind, MemoryEpisode,
    MemoryEvent, MemoryScope,
)
from mematlas.core.plugin import MethodContext, MethodServices
from mematlas.methods.memq.core import bundled_source_root, load_upstream, scope_key
from mematlas.methods.memq.method import MemQMemory
from mematlas.runtime.memory import Memory


UPSTREAM_ROOT = bundled_source_root()
# MemOS configures logging while the native optional memory backend imports.
# Isolate those files and keep offline fixture imports from reading credentials.
_IMPORT_ARTIFACTS = tempfile.TemporaryDirectory(prefix="memq-test-import-")
atexit.register(_IMPORT_ARTIFACTS.cleanup)
with patch.dict(os.environ, {"MEMOS_BASE_PATH": _IMPORT_ARTIFACTS.name,
                             "PYTHON_DOTENV_DISABLED": "1"}):
    UPSTREAM = load_upstream(UPSTREAM_ROOT)
    from memq.service.memory_service import MemoryService
    from memq.service.memory_store import MemoryStore
    from memq.service.strategies import StrategyConfiguration
    from memq.configs.config import RLConfig


SCRIPT = "1. Inspect the task.\n2. Complete the required procedure.\n3. Verify the result."
REFLECTION = "Check the missing precondition before attempting the final action."
VECTOR_DIMENSION = 4


class OfflineLLM:
    model = "offline-memq-llm"

    def __init__(self) -> None:
        self.calls: list[tuple[list[dict[str, Any]], dict[str, Any]]] = []
        self.failure: Exception | None = None

    async def complete(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        self.calls.append((copy.deepcopy(messages), copy.deepcopy(kwargs)))
        if self.failure is not None:
            raise self.failure
        content = messages[-1]["content"]
        if isinstance(content, list):
            content = "\n".join(block.get("text", "") for block in content if block.get("type") == "text")
        if "Keywords:" in content:
            return '"Inspect", procedure, verification'
        if "Provide a brief reflection:" in content:
            return REFLECTION
        if "High-level script:" in content:
            return SCRIPT
        raise AssertionError(f"Unexpected native MemQ LLM prompt: {content[:160]!r}")


class OfflineEmbedder:
    model = "offline-memq-embedding"

    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.vectors: dict[str, list[float]] = {}
        self.failure: Exception | None = None

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        if self.failure is not None:
            raise self.failure
        return [list(self.vectors.get(text, [1.0, 0.0, 0.0, 0.0])) for text in texts]


class NativeHarness:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.llm = OfflineLLM()
        self.embedding = OfflineEmbedder()
        self.adapters: list[MemQMemory] = []
        self.memories: list[Memory] = []

    def services(self, name: str) -> MethodServices:
        return MethodServices(llm=self.llm, embedder=self.embedding,
                              artifact_dir=self.root / name)

    async def adapter(self, **params: Any) -> MemQMemory:
        params = {"vector_dim": VECTOR_DIMENSION, "embedding_dims": VECTOR_DIMENSION,
                  "num_workers": 1, "epsilon": 0.0, **params}
        adapter = MemQMemory(MethodConfig("memq", params),
                             self.services(f"adapter-{len(self.adapters)}"))
        self.adapters.append(adapter)
        await adapter.setup(MethodContext("offline", MethodConfig("memq", params), RuntimeConfig()))
        return adapter

    async def memory(self, **params: Any) -> Memory:
        params = {"vector_dim": VECTOR_DIMENSION, "embedding_dims": VECTOR_DIMENSION,
                  "num_workers": 1, "epsilon": 0.0, **params}
        directory = self.root / f"runtime-{len(self.memories)}"
        memory = Memory(MemAtlasConfig(MethodConfig("memq", params),
                                      RuntimeConfig(artifact_dir=str(directory))),
                        services=self.services(directory.name))
        self.memories.append(memory)
        await memory.setup()
        return memory

    async def close(self) -> None:
        for memory in self.memories:
            await memory.close()
        for adapter in self.adapters:
            await adapter.close()


def query(scope: MemoryScope, task: str, *, top_k: int | None = None,
          **metadata: Any) -> DecisionContext:
    return DecisionContext(scope=scope, query=(ContentBlock.text(task),),
                           top_k=top_k, metadata={"task": task, **metadata})


def episode(scope: MemoryScope, task: str, *, success: bool | None = True,
            retrieved: tuple[dict[str, Any], ...] = ()) -> MemoryEpisode:
    return MemoryEpisode(scope=scope, events=(
        MemoryEvent(EventKind.ACTION, scope, (ContentBlock.text("Perform the procedure."),)),
        MemoryEvent(EventKind.OBSERVATION, scope, (ContentBlock.text("External environment result."),)),
    ), result=EpisodeOutcome(scope, success=success, feedback="External feedback.",
                             metadata={"task": task, "task_id": task}),
        retrieved=retrieved)


def native_instance(adapter: MemQMemory, scope: MemoryScope) -> MemoryService:
    return adapter._instances[scope_key(scope)]


def q_value(native: MemoryService, memory_id: str) -> float:
    return native._store.get(memory_id)["metadata"]["q_value"]


def store_row(native: MemoryService, memory_id: str, q: float,
              *, vector: list[float] | None = None, query_text: str | None = None) -> None:
    """Seed controlled values in the actual MemoryStore for formula checks."""
    task = query_text or memory_id
    vector = vector or [1.0, 0.0, 0.0, 0.0]
    native._store.add(memory_id, vector, {
        "id": memory_id, "memory": task,
        "metadata": {"q_value": q, "q_visits": 0, "retrieval_count": 0,
                     "full_content": f"Task: {task}\n\nNative procedure {memory_id}.",
                     "task_id": memory_id, "success": True},
    })
    native.dict_memory.setdefault(task, []).append(memory_id)
    native.query_embeddings[task] = list(vector)


@contextmanager
def record_native_calls(native: MemoryService, names: tuple[str, ...]) -> Iterator[list[tuple[str, tuple, dict]]]:
    """Observe native dispatch order while executing every original method."""
    calls = []
    with ExitStack() as stack:
        for name in names:
            original = getattr(native, name)

            def recording(*args: Any, _name=name, _original=original, **kwargs: Any) -> Any:
                calls.append((_name, copy.deepcopy(args), copy.deepcopy(kwargs)))
                return _original(*args, **kwargs)

            stack.enter_context(patch.object(native, name, side_effect=recording))
        yield calls


def directory_digest(root: Path) -> dict[str, str]:
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in root.rglob("*") if path.is_file() and path.suffix not in {".log", ".jsonl"}
            and "provider_logs" not in path.parts}
