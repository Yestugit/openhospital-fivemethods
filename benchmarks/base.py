from __future__ import annotations

import hashlib
import inspect
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Iterable, Mapping

from mematlas.core import evaluation


class BenchmarkAdapter(ABC):
    benchmark_id: str

    @abstractmethod
    def cases(self, split: str = "test") -> Iterable[evaluation.BenchmarkCase]:
        raise NotImplementedError

    def native_score(self, predictions: list[evaluation.Prediction]) -> Mapping[str, Any]:
        return {}

    async def native_score_async(self, predictions: list[evaluation.Prediction]) -> Mapping[str, Any]:
        result = self.native_score(predictions)
        if inspect.isawaitable(result):
            result = await result
        return result

    def describe(self) -> Mapping[str, Any]:
        return {"benchmark_id": self.benchmark_id}

    @staticmethod
    def file_sha256(path: str | Path) -> str:
        digest = hashlib.sha256()
        with Path(path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
