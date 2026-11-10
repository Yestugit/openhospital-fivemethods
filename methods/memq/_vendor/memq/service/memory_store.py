"""In-memory vector store backed by numpy arrays with JSON/npy persistence.

Replaces MemOS + Qdrant for MemRL's use case (~10K memories, sequential
retrieval, brute-force cosine search is <20ms vs LLM latency of seconds).
"""

from __future__ import annotations

import json
import logging
import os
import threading
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class SearchHit:
    """Lightweight search result, mimics qdrant ScoredPoint interface."""

    __slots__ = ("payload", "score")

    def __init__(self, payload: dict[str, Any], score: float) -> None:
        self.payload = payload
        self.score = score

    def __repr__(self) -> str:
        return f"SearchHit(score={self.score:.4f}, id={self.payload.get('id', '?')})"


class MemoryStore:
    """In-memory vector store — source of truth for vectors, payloads, and Q-values.

    All data lives in RAM.  Persistence is explicit via :meth:`save` / :meth:`load`.

    Thread safety: ``_lock`` protects mutations (add / update).  Reads (get /
    search) are safe under the GIL for dict lookups and numpy slicing.
    """

    # ------------------------------------------------------------------ init

    def __init__(self, vector_dim: int = 4096) -> None:
        self._dim = vector_dim
        self._ids: list[str] = []
        self._vectors: np.ndarray = np.empty((0, vector_dim), dtype=np.float32)
        self._payloads: dict[str, dict[str, Any]] = {}
        self._id_to_idx: dict[str, int] = {}
        self._lock = threading.Lock()
        # Pre-computed L2 norms for cosine similarity (invalidated on add).
        self._norms: np.ndarray | None = None

    # ------------------------------------------------------------------ CRUD

    def add(self, mem_id: str, vector: list[float] | np.ndarray, payload: dict[str, Any]) -> None:
        """Insert a new memory.  Overwrites silently if *mem_id* already exists."""
        vec = np.asarray(vector, dtype=np.float32).reshape(1, -1)
        if vec.shape[1] != self._dim:
            raise ValueError(f"Vector dim {vec.shape[1]} != store dim {self._dim}")
        with self._lock:
            if mem_id in self._id_to_idx:
                # Overwrite existing entry.
                idx = self._id_to_idx[mem_id]
                self._vectors[idx] = vec[0]
                self._payloads[mem_id] = dict(payload)
            else:
                idx = len(self._ids)
                self._ids.append(mem_id)
                self._id_to_idx[mem_id] = idx
                self._vectors = np.vstack([self._vectors, vec])
                self._payloads[mem_id] = dict(payload)
            self._norms = None  # invalidate

    def get(self, mem_id: str) -> dict[str, Any] | None:
        """Return payload dict for *mem_id*, or ``None``."""
        return self._payloads.get(mem_id)

    def update(self, mem_id: str, data: dict[str, Any]) -> None:
        """Merge *data* into the payload for *mem_id*.

        Typically called with ``{"id": id, "memory": text, "metadata": new_meta}``.
        """
        with self._lock:
            existing = self._payloads.get(mem_id)
            if existing is None:
                logger.warning("[MemoryStore] update called for unknown id %s", mem_id)
                return
            existing.update(data)

    def get_all(self) -> list[dict[str, Any]]:
        """Return all payloads as a flat list of dicts."""
        return list(self._payloads.values())

    def __len__(self) -> int:
        return len(self._ids)

    def __contains__(self, mem_id: str) -> bool:
        return mem_id in self._id_to_idx

    # ------------------------------------------------------------------ search

    def search(self, query_vec: list[float] | np.ndarray, k: int = 10) -> list[SearchHit]:
        """Brute-force cosine similarity search.  Returns up to *k* hits, sorted descending."""
        n = len(self._ids)
        if n == 0:
            return []
        k = min(k, n)
        qv = np.asarray(query_vec, dtype=np.float32).ravel()
        q_norm = float(np.linalg.norm(qv))
        if q_norm == 0:
            return []
        qv_normed = qv / q_norm

        # Lazy-compute row norms.
        if self._norms is None:
            norms = np.linalg.norm(self._vectors[:n], axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            self._norms = norms

        normed = self._vectors[:n] / self._norms
        sims = normed @ qv_normed  # (N,)

        if k >= n:
            top_idx = np.argsort(-sims)
        else:
            # Partial sort for efficiency.
            part_idx = np.argpartition(-sims, k)[:k]
            top_idx = part_idx[np.argsort(-sims[part_idx])]

        hits: list[SearchHit] = []
        for i in top_idx:
            mid = self._ids[i]
            hits.append(SearchHit(payload=self._payloads[mid], score=float(sims[i])))
        return hits

    # ------------------------------------------------------------------ persistence

    def save(self, directory: str) -> None:
        """Persist store to *directory* as ``vectors.npy`` + ``payloads.json``."""
        os.makedirs(directory, exist_ok=True)
        np.save(os.path.join(directory, "vectors.npy"), self._vectors[:len(self._ids)])
        with open(os.path.join(directory, "payloads.json"), "w", encoding="utf-8") as f:
            json.dump(
                {"ids": self._ids, "payloads": dict(self._payloads)},
                f,
                ensure_ascii=False,
                default=str,
            )
        logger.info("[MemoryStore] Saved %d memories to %s", len(self._ids), directory)

    @classmethod
    def load(cls, directory: str) -> MemoryStore:
        """Load from ``vectors.npy`` + ``payloads.json`` written by :meth:`save`.

        Falls back to legacy ``textual_memory.json`` (cube dump) if present.
        """
        vec_path = os.path.join(directory, "vectors.npy")
        payload_path = os.path.join(directory, "payloads.json")

        # Native format.
        if os.path.isfile(vec_path) and os.path.isfile(payload_path):
            vectors = np.load(vec_path).astype(np.float32)
            with open(payload_path, encoding="utf-8") as f:
                data = json.load(f)
            ids: list[str] = data["ids"]
            payloads: dict[str, dict] = data["payloads"]

            store = cls(vector_dim=vectors.shape[1] if vectors.ndim == 2 else 4096)
            store._ids = ids
            store._vectors = vectors
            store._payloads = payloads
            store._id_to_idx = {mid: i for i, mid in enumerate(ids)}
            logger.info("[MemoryStore] Loaded %d memories from %s", len(ids), directory)
            return store

        # Legacy: textual_memory.json inside a cube dump.
        legacy_path = os.path.join(directory, "textual_memory.json")
        if not os.path.isfile(legacy_path):
            # Also check cube/ subdirectory (snapshot layout).
            legacy_path = os.path.join(directory, "cube", "textual_memory.json")
        if os.path.isfile(legacy_path):
            return cls.load_legacy(legacy_path)

        raise FileNotFoundError(f"No store data found in {directory}")

    @classmethod
    def load_legacy(cls, textual_memory_json: str) -> MemoryStore:
        """Load from MemOS cube dump ``textual_memory.json``.

        Each entry is ``{"id": str, "vector": list[float], "payload": {"id", "memory", "metadata"}}``.
        """
        with open(textual_memory_json, encoding="utf-8") as f:
            items = json.load(f)

        if not items:
            return cls()

        # Detect vector dim from first entry.
        first_vec = items[0].get("vector")
        dim = len(first_vec) if first_vec else 4096
        store = cls(vector_dim=dim)

        ids: list[str] = []
        vectors: list[np.ndarray] = []
        for item in items:
            mid = item.get("id") or (item.get("payload", {}).get("id"))
            vec = item.get("vector")
            payload = item.get("payload", item)
            if mid is None:
                continue
            mid = str(mid)
            ids.append(mid)
            if vec is not None:
                vectors.append(np.asarray(vec, dtype=np.float32))
            else:
                vectors.append(np.zeros(dim, dtype=np.float32))
            store._payloads[mid] = dict(payload)

        store._ids = ids
        store._id_to_idx = {mid: i for i, mid in enumerate(ids)}
        if vectors:
            store._vectors = np.stack(vectors)
        else:
            store._vectors = np.empty((0, dim), dtype=np.float32)

        logger.info(
            "[MemoryStore] Loaded %d memories from legacy %s", len(ids), textual_memory_json
        )
        return store
