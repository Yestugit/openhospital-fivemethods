"""Experiment records live beside the Memory checkpoint, owned by evaluation."""
import json
from contextlib import asynccontextmanager
from copy import deepcopy
from pathlib import Path
from uuid import uuid4

from mematlas.core.files import atomic_write_json
from mematlas.core.provenance import file_sha256, json_sha256
from mematlas.core.episodes import EpisodeResult, episode_counts


class EvaluationHistory:
    """Evaluation progress and feedback timing; trajectories belong to Memory."""

    def __init__(self):
        self.results = {}
        self.pending_feedback = {}
        self.exposures = 0

    def record_result(self, result):
        previous = self.results.get(result.episode_id)
        if previous and previous.learning == "submitted" and previous != result:
            raise ValueError("Submitted learning cannot be replaced")
        self.results[result.episode_id] = result

    def result(self, episode_id, *, scope):
        from mematlas.runtime.history import EpisodeHistory
        identity = EpisodeHistory.identity(episode_id, scope)
        return self.results.get(identity, EpisodeResult(identity, case_id=episode_id, execution="pending"))

    def counts(self, *, exclude=()):
        return episode_counts(row for key, row in self.results.items() if key not in exclude)

    def snapshot(self):
        return deepcopy({"schema_version": 1, "results": {key: row.to_dict() for key, row in self.results.items()},
                         "pending_feedback": self.pending_feedback, "exposures": self.exposures})

    def restore(self, data):
        if data.get("schema_version") != 1:
            raise ValueError("Unsupported evaluation history schema")
        results = {key: EpisodeResult.from_dict(row) for key, row in data["results"].items()}
        if any(key != row.episode_id for key, row in results.items()):
            raise ValueError("Invalid evaluation result identity")
        exposures = data["exposures"]
        if type(exposures) is not int or exposures < 0:
            raise ValueError("Invalid exposure count")
        pending = deepcopy(data["pending_feedback"])
        if any(key not in results or results[key].learning == "submitted" or
               type(row.get("due")) is not int or row["due"] < 0 for key, row in pending.items()):
            raise ValueError("Invalid pending feedback")
        self.results, self.pending_feedback, self.exposures = results, pending, exposures


class _MemoryExperience:
    def __init__(self, memory):
        self.core_memory = memory
        self.history = EvaluationHistory()
        self.metadata = {}

    async def start(self):
        await self.core_memory._setup()

    async def commit(self):
        await self.core_memory.flush()

    def record_completed(self, episode_id, *, scope, case_id=None):
        identity = self.core_memory._history.identity(episode_id, scope)
        if self.history.result(episode_id, scope=scope).learning != "submitted":
            self.history.record_result(EpisodeResult(identity, case_id=case_id or episode_id,
                                                    metadata={"scope": scope}))

    def episode(self, identity):
        return self.core_memory._history.episode(identity)

    @asynccontextmanager
    async def frozen_operation(self, fingerprint):
        """Restore method-local read caches before accepting or caching a result."""
        memory = self.core_memory
        async with memory._lock:
            snapshot = await memory._capture_state() if memory.capabilities.read_only_mode == "snapshot" else None
        try:
            yield
        finally:
            # Individual reads restore native caches under Memory's lock. A
            # sibling case may currently be replacing its database handles;
            # snapshot/restore/fingerprint must use that same lock as well.
            async with memory._lock:
                if snapshot is not None and await memory._capture_state() != snapshot:
                    await memory._restore_state(snapshot)
                if await self._state_fingerprint_unlocked() != fingerprint:
                    raise RuntimeError("Operation changed frozen memory; result cannot be cached")

    def snapshot_metadata(self):
        return deepcopy(self.metadata)

    def restore_metadata(self, metadata):
        self.metadata = deepcopy(metadata)

    async def snapshot_to(self, path, *, learned_count):
        output = Path(path).expanduser().resolve()
        memory_path = output.parent / f"memory-{uuid4().hex}.json"
        # Saving captures pending work exactly; it never schedules evolution.
        await self.core_memory.save(memory_path)
        atomic_write_json(output, {"schema_version": 6, "method_id": self.core_memory.method_id,
            "memory_file": memory_path.name, "memory_sha256": file_sha256(memory_path),
            "learned_count": learned_count, "integration": self.snapshot_metadata(),
            "episodes": self.history.snapshot()})
        return output

    async def restore_from(self, path):
        path = Path(path).resolve()
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != 6 or payload["method_id"] != self.core_memory.method_id:
            raise ValueError("Experience snapshot requires current schema/method")
        count = payload["learned_count"]
        if type(count) is not int or count < 0:
            raise ValueError("Invalid learned_count")
        memory_path = path.parent / payload["memory_file"]
        if memory_path.parent.resolve() != path.parent or file_sha256(memory_path) != payload["memory_sha256"]:
            raise ValueError("Invalid experience memory file or checksum")
        history = EvaluationHistory()
        history.restore(payload["episodes"])
        saved_history = json.loads(memory_path.read_text(encoding="utf-8"))["history"]
        if set(history.pending_feedback) - saved_history["episodes"].keys():
            raise ValueError("Pending feedback episode records are missing")
        await self.core_memory.load(memory_path)
        self.restore_metadata(payload["integration"])
        self.history.restore(history.snapshot())
        return count

    async def state_fingerprint(self):
        async with self.core_memory._lock:
            return await self._state_fingerprint_unlocked()

    async def _state_fingerprint_unlocked(self):
        from mematlas.core.state_files import fingerprint_state
        memory = self.core_memory
        state = fingerprint_state(await memory._capture_state(),
                                  resolve_files=memory.config.runtime.executor != "service")
        return json_sha256({"state": state, "history": memory._history.snapshot(),
                            "integration": self.snapshot_metadata(), "episodes": self.history.snapshot()})

    async def close(self):
        await self.core_memory._close()
