from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from mematlas.benchmarks.base import BenchmarkAdapter
from mematlas.core.evaluation import BenchmarkCase
from mematlas.core.models import MemoryContribution, MemoryEpisode, MemoryEvent, content_value
from mematlas.core.provenance import json_sha256
from mematlas.core.scope import normalize_scope


def _event(value: Mapping[str, Any], scope, *, identity: str) -> MemoryEvent:
    raw = dict(value)
    if "text" in raw:
        raw.setdefault("content", raw.pop("text"))
    raw.setdefault("content", "")
    raw.setdefault("kind", "message")
    raw.setdefault("event_id", identity)
    raw.setdefault("scope", scope)
    raw.setdefault("timestamp", datetime(1970, 1, 1, tzinfo=timezone.utc))
    return MemoryEvent.from_dict(raw)


class JsonDatasetBenchmark(BenchmarkAdapter):
    """JSON/JSONL cases using public events and optional complete episodes."""

    def __init__(self, path: str | Path, *, benchmark_id: str = "custom") -> None:
        self.path = Path(path).expanduser().resolve()
        self.benchmark_id = benchmark_id
        text = self.path.read_text(encoding="utf-8-sig")
        if self.path.suffix.lower() == ".jsonl":
            records = [json.loads(line) for line in text.splitlines() if line.strip()]
        else:
            payload = json.loads(text)
            records = payload.get("cases", []) if isinstance(payload, Mapping) else payload
        if not isinstance(records, list) or not all(isinstance(row, Mapping) for row in records):
            raise ValueError("Custom benchmark must contain a list of case objects")
        self._records = records

    def cases(self, split: str = "test") -> Iterable[BenchmarkCase]:
        for index, raw in enumerate(self._records):
            record_split = str(raw.get("split", raw.get("metadata", {}).get("split", "test")))
            if split not in {"all", record_split}:
                continue
            if "outcome" in raw:
                raise ValueError("Use episode.success/reward/feedback instead of outcome")
            case_id = str(raw.get("case_id", raw.get("id", index)))
            scope = normalize_scope(raw.get("scope") or {"user_id": f"{self.benchmark_id}:{case_id}"})
            prefix = json_sha256({"file": str(self.path), "case": case_id, "split": record_split})
            history = tuple(_event(event, scope, identity=f"{prefix}:{i}")
                            for i, event in enumerate(raw.get("history", [])))
            episode = None
            if raw.get("episode") is not None:
                data = dict(raw["episode"])
                data.setdefault("episode_id", f"{prefix}:episode")
                data.setdefault("scope", scope)
                events = data.pop("events", None)
                data["events"] = list(history) if events is None else [
                    _event(event, scope, identity=f"{prefix}:episode:{i}") for i, event in enumerate(events)]
                data["applied_memory"] = [MemoryContribution.from_dict(row)
                                          for row in data.get("applied_memory", [])]
                episode = MemoryEpisode(**data)
            yield BenchmarkCase(case_id=case_id, scope=scope, history=history,
                query=content_value(raw.get("query", "")),
                reference=raw.get("reference", raw.get("answer")), episode=episode,
                metadata={**dict(raw.get("metadata", {})), "split": record_split})

    def describe(self) -> Mapping[str, Any]:
        return {"benchmark_id": self.benchmark_id, "path": str(self.path),
                "sha256": self.file_sha256(self.path), "record_count": len(self._records)}


class InMemoryBenchmark(BenchmarkAdapter):
    def __init__(self, benchmark_id: str, cases: Iterable[BenchmarkCase]) -> None:
        self.benchmark_id = benchmark_id
        self._cases = tuple(cases)

    def cases(self, split: str = "test") -> Iterable[BenchmarkCase]:
        return (case for case in self._cases
                if split == "all" or str(case.metadata.get("split", "test")) == split)

    def describe(self) -> Mapping[str, Any]:
        return {"benchmark_id": self.benchmark_id, "record_count": len(self._cases)}
