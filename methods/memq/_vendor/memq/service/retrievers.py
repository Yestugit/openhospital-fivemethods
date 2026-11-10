"""
Retrievers for different retrieve strategies in the Memp system.

This module provides a strategy-pattern implementation for retrieval logic,
mirroring builders.py for build strategies. It centralizes:
- Formatting each memory item into a consistent dict
- Concrete retrievers for RANDOM, QUERY, AVEFACT
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from ..providers.embedding import AverageEmbedder
from .memory_store import MemoryStore, SearchHit
from .strategies import RetrieveStrategy

# ---------- Utilities ----------

def _extract_item_and_score(hit: Any) -> Tuple[Optional[Dict[str, Any]], Optional[float]]:
    """Normalize various hit shapes into (payload_dict, score).

    Supported shapes:
    - dict with id/memory/metadata keys
    - SearchHit (has .payload and .score)
    - {'item': ..., 'score': ...}
    """
    if hit is None:
        return None, None

    # Dict hit: {'item': ..., 'score': ...}
    if isinstance(hit, dict) and "item" in hit:
        item = hit.get("item")
        score = hit.get("score")
        try:
            score_f = float(score) if score is not None else None
        except Exception:
            score_f = None
        return item, score_f

    # SearchHit (from MemoryStore.search)
    if isinstance(hit, SearchHit):
        return hit.payload, hit.score

    # Qdrant-like hit: has payload + score attrs
    payload = getattr(hit, "payload", None)
    if isinstance(payload, dict):
        score = getattr(hit, "score", None)
        try:
            score_f = float(score) if score is not None else None
        except Exception:
            score_f = None
        return payload, score_f

    # Plain dict with memory keys
    if isinstance(hit, dict) and ("memory" in hit or "id" in hit):
        return hit, None

    return None, None


def _extract_full_content(metadata: Any) -> Optional[str]:
    """Best-effort extraction of full_content from metadata."""
    if metadata is None:
        return None
    if isinstance(metadata, dict):
        fc = metadata.get("full_content")
        if isinstance(fc, str) and fc:
            return fc
        return None
    try:
        if hasattr(metadata, "model_extra"):
            fc = metadata.model_extra.get("full_content")
            if isinstance(fc, str) and fc:
                return fc
    except Exception:
        pass
    try:
        fc = getattr(metadata, "full_content", None)
        if isinstance(fc, str) and fc:
            return fc
    except Exception:
        pass
    return None


def _extract_similarity_fallback(metadata: Any) -> float:
    """Fallback similarity when a hit has no explicit score."""
    try:
        if metadata is None:
            return 0.0
        if isinstance(metadata, dict):
            return float(metadata.get("relativity", 0.0) or 0.0)
        rel = getattr(metadata, "relativity", None)
        if rel is not None:
            return float(rel)
    except Exception:
        pass
    return 0.0


def _format_memory_result(item: Any) -> Dict[str, Any]:
    """Format a memory item (dict or SearchHit) to a consistent dict.

    Returns keys:
    - memory_id: str
    - content: str (prefer metadata.full_content; fallback to item.memory)
    - metadata: Any
    - similarity: float
    - memory_item: original object
    """
    mem_item, score = _extract_item_and_score(item)
    if mem_item is None:
        return {
            "memory_id": "unknown",
            "content": str(item),
            "metadata": {},
            "similarity": 0.0,
            "memory_item": item,
        }

    if isinstance(mem_item, dict):
        metadata = mem_item.get("metadata")
        mem_id = mem_item.get("id", "unknown")
        memory_text = mem_item.get("memory", "")
    else:
        metadata = getattr(mem_item, "metadata", None)
        mem_id = getattr(mem_item, "id", "unknown")
        memory_text = getattr(mem_item, "memory", "")

    full_content = _extract_full_content(metadata)
    content = full_content or memory_text
    similarity = float(score) if score is not None else _extract_similarity_fallback(metadata)

    return {
        "memory_id": mem_id,
        "content": content,
        "metadata": metadata,
        "similarity": similarity,
        "memory_item": mem_item,
    }


# ---------- Strategy Base Class ----------

class BaseRetriever(ABC):
    def __init__(self, store: MemoryStore, user_id: str):
        self.store = store
        self.user_id = user_id

    @abstractmethod
    def retrieve(self, task_description: str, k: int, threshold: float) -> List[Dict[str, Any]]:
        ...


# ---------- Concrete Retrievers ----------

class RandomRetriever(BaseRetriever):
    def retrieve(self, task_description: str, k: int, threshold: float) -> List[Dict[str, Any]]:
        all_items = self.store.get_all()
        if not all_items:
            return []
        sel = random.sample(all_items, min(k, len(all_items)))
        return [_format_memory_result(x) for x in sel]


class QueryRetriever(BaseRetriever):
    """Vector similarity retriever using MemoryStore.search."""

    def __init__(self, store: MemoryStore, user_id: str, embedder: Any | None = None):
        super().__init__(store, user_id)
        self.embedder = embedder

    def retrieve(self, task_description: str, k: int, threshold: float) -> List[Dict[str, Any]]:
        if self.embedder is None:
            return []
        try:
            vecs = self.embedder.embed([task_description])
            if not vecs:
                return []
            qv = vecs[0]
        except Exception:
            return []

        hits = self.store.search(qv, k=max(k * 2, k))
        out: List[Dict[str, Any]] = []
        for hit in hits:
            if threshold > 0 and hit.score < threshold:
                continue
            fm = _format_memory_result(hit)
            fm["similarity"] = hit.score
            out.append(fm)
            if len(out) >= k:
                break
        return out


class AveFactRetriever(BaseRetriever):
    def __init__(
        self,
        store: MemoryStore,
        user_id: str,
        llm: Any,
        keyer: Any,
        max_keywords: int = 8,
        embedder: Any | None = None,
    ):
        super().__init__(store, user_id)
        self.llm = llm
        self.keyer = keyer
        self.max_keywords = max_keywords
        self.embedder = embedder

    def _build_query_vector(self, text: str) -> List[float] | None:
        if self.embedder is None:
            return None
        try:
            keywords = self.llm.extract_keywords(text, self.max_keywords)
        except Exception:
            keywords = None
        try:
            if keywords:
                vecs = self.embedder.embed(keywords)
                if not vecs:
                    return None
                return (
                    AverageEmbedder.average_embeddings(vecs)
                    if len(vecs) > 1
                    else vecs[0]
                )
            single = getattr(self.embedder, "embed_single", None)
            if callable(single):
                return single(text)
            vecs = self.embedder.embed([text])
            return vecs[0] if vecs else None
        except Exception:
            return None

    def retrieve(self, task_description: str, k: int, threshold: float) -> List[Dict[str, Any]]:
        # Try vector path first
        qv = self._build_query_vector(task_description)
        if qv is not None:
            hits = self.store.search(qv, k=max(k * 2, k))
            if hits:
                items: List[Dict[str, Any]] = []
                for hit in hits:
                    if threshold > 0 and hit.score < threshold:
                        continue
                    fm = _format_memory_result(hit)
                    fm["similarity"] = hit.score
                    items.append(fm)
                if items:
                    return items[:k]

        # Fallback: embed keyword query and search
        try:
            keywords = self.llm.extract_keywords(task_description, self.max_keywords)
        except Exception:
            keywords = None
        keyword_text = " ".join(keywords) if keywords else task_description
        if self.embedder is not None:
            try:
                vecs = self.embedder.embed([keyword_text])
                if vecs:
                    hits = self.store.search(vecs[0], k=k)
                    out = []
                    for hit in hits:
                        if threshold > 0 and hit.score < threshold:
                            continue
                        fm = _format_memory_result(hit)
                        fm["similarity"] = hit.score
                        out.append(fm)
                    return out[:k]
            except Exception:
                pass
        return []


# ---------- Factory ----------

def get_retriever(
    strategy: RetrieveStrategy,
    *,
    store: MemoryStore,
    user_id: str,
    llm: Optional[Any] = None,
    keyer: Optional[Any] = None,
    max_keywords: int = 8,
    embedder: Optional[Any] = None,
) -> BaseRetriever:
    if strategy == RetrieveStrategy.RANDOM:
        return RandomRetriever(store, user_id)
    if strategy == RetrieveStrategy.QUERY:
        return QueryRetriever(store, user_id, embedder=embedder)
    # default to AVEFACT
    return AveFactRetriever(
        store,
        user_id,
        llm=llm,
        keyer=keyer,
        max_keywords=max_keywords,
        embedder=embedder,
    )
