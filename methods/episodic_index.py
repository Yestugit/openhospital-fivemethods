"""Dependency-free derived index for episode/query vectors."""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence


def checked_vector(vector: Sequence[float]) -> list[float]:
    if not vector or any(not isinstance(value, (int, float)) or isinstance(value, bool)
                         or not math.isfinite(value) for value in vector):
        raise ValueError("Episode vectors must be finite, nonempty numeric vectors")
    if not any(vector):
        raise ValueError("Episode vectors must have a nonzero norm")
    return [float(value) for value in vector]


class EpisodeQueryIndex:
    """In-memory cosine index rebuilt from method-owned snapshot vectors.

    The canonical query vectors live in each method's snapshot. Keeping this
    index derived makes checkpoint restore deterministic and avoids requiring
    a database SDK solely for local similarity search.
    """

    def __init__(self) -> None:
        self._rows: dict[str, dict[str, Any]] = {}
        self.dimension: int | None = None

    def replace(self, rows: Mapping[str, Mapping[str, Any]]) -> None:
        dimensions = {len(checked_vector(row["vector"])) for row in rows.values()}
        if len(dimensions) > 1:
            raise ValueError("Episode query index cannot contain mixed dimensions")
        self.dimension = next(iter(dimensions), None)
        self._rows = {
            str(key): {"vector": checked_vector(row["vector"]), "scope_key": str(row["scope_key"])}
            for key, row in rows.items()
        }

    def add(self, key: str, row: Mapping[str, Any]) -> None:
        vector = checked_vector(row["vector"])
        if self.dimension is None:
            self.dimension = len(vector)
        if len(vector) != self.dimension:
            raise ValueError("Episode query embedding dimension does not match the index")
        self._rows[str(key)] = {"vector": vector, "scope_key": str(row["scope_key"])}

    def search(self, vector: Sequence[float], *, scope_key: str, count: int) -> list[tuple[str, float]]:
        vector = checked_vector(vector)
        if count <= 0 or self.dimension is None:
            return []
        if len(vector) != self.dimension:
            raise ValueError("Episode query embedding dimension does not match the index")
        query_norm = math.sqrt(sum(value * value for value in vector))
        scored = []
        for key, row in self._rows.items():
            if row["scope_key"] != scope_key:
                continue
            candidate = row["vector"]
            candidate_norm = math.sqrt(sum(value * value for value in candidate))
            score = sum(left * right for left, right in zip(vector, candidate)) / (query_norm * candidate_norm)
            scored.append((key, float(score)))
        return sorted(scored, key=lambda item: (-item[1], item[0]))[:count]

    def close(self) -> None:
        self._rows.clear()
        self.dimension = None
