"""Public-interface glue and immutable checkpoints; no EMem algorithm lives here."""
from __future__ import annotations

import asyncio
from copy import deepcopy
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
from types import ModuleType
from typing import Any, Mapping
from dataclasses import replace

from mematlas.core.models import MemoryContribution, MemoryEvent, content_text
from mematlas.core.plugin import BaseMemoryMethod
from mematlas.core.scope import normalize_scope
from mematlas.methods.storage import method_data_dir


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def storage_path(value):
    """Use Windows extended paths for deep native layouts, without relayout."""
    path = Path(value).resolve()
    raw = str(path)
    if os.name == "nt" and not raw.startswith("\\\\?\\"):
        return Path("\\\\?\\UNC\\" + raw[2:] if raw.startswith("\\\\") else "\\\\?\\" + raw)
    return path


def put_blob(directory: Path, data: bytes) -> dict:
    key = hashlib.sha256(data).hexdigest()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / key
    if not path.exists():
        # Concurrent writers of identical bytes may safely replace this address.
        import os
        import tempfile
        fd, name = tempfile.mkstemp(dir=directory, prefix=".pending-")
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(data)
            os.replace(name, path)
        finally:
            if Path(name).exists():
                Path(name).unlink()
    elif hashlib.sha256(path.read_bytes()).hexdigest() != key:
        raise ValueError(f"Corrupt immutable checkpoint: {path}")
    return {"path": str(path.resolve()), "sha256": key, "size": len(data)}


def get_blob(ref: Mapping) -> bytes:
    if set(ref) != {"path", "sha256", "size"}:
        raise ValueError("Invalid checkpoint file reference")
    data = Path(ref["path"]).read_bytes()
    if len(data) != ref["size"] or hashlib.sha256(data).hexdigest() != ref["sha256"]:
        raise ValueError("Checkpoint size or SHA-256 mismatch")
    return data


def checked_source(services, method_id, revision):
    from mematlas.core.errors import ConfigurationError
    root = services.source_root(method_id)
    if root is None or not (root / "src" / method_id).is_dir():
        raise ConfigurationError(f"{method_id}: configure runtime.source_roots.{method_id} to the pinned official source checkout")
    spec = json.loads((Path(__file__).parent.parent / method_id / "source.json").read_text(encoding="utf-8"))
    if spec["revision"] != revision:
        raise ConfigurationError(f"{method_id}: source specification revision mismatch")
    for name, expected in spec["sha256"].items():
        path = root / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ConfigurationError(f"{method_id}: pinned upstream source differs: {name}")
    return root


def namespace(name: str, path: Path):
    """Select only used official modules, avoiding optional facade imports."""
    if name in sys.modules:
        current = sys.modules[name]
        paths = [str(Path(p).resolve()) for p in getattr(current, "__path__", [])]
        if str(path.resolve()) not in paths:
            raise RuntimeError(f"Conflicting upstream Python package {name}; use an isolated executor")
        return current
    result = ModuleType(name)
    result.__path__ = [str(path.resolve())]
    result.__package__ = name
    sys.modules[name] = result
    return result


def run_service(coroutine, loop=None):
    # Native algorithms run in asyncio.to_thread(), so this never nests the
    # public Memory loop; native EMem also invokes this from its own workers.
    if loop is not None:
        return asyncio.run_coroutine_threadsafe(coroutine, loop).result()
    return asyncio.run(coroutine)


def evidence(unit, episode_ref):
    """Attribute supplemental caller evidence without replaying old events.

    Retrieved/applied memory and extension payloads are never ingested as new
    observations. Exact duplicate event text is kept only once for fields that
    the public runtime has already materialized as events.
    """
    rows = list(unit.events)
    existing = {content_text(event.content) for event in rows}
    for field, value, role in (("task", unit.task, "user"), ("output", unit.output, "assistant"),
                               ("feedback", unit.feedback, "system")):
        if value is None or value == "" or value == {}:
            continue
        text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True)
        # content_text uses JSON for non-string content; compare parsed content
        # too so harmless key order cannot duplicate the runtime's own event.
        def equivalent(event):
            try:
                return json.loads(content_text(event.content)) == value
            except (ValueError, TypeError):
                return False
        if text in existing or any(equivalent(event) for event in rows):
            continue
        stamp = {"timestamp": rows[0].timestamp} if rows else {}
        rows.append(MemoryEvent("feedback" if field == "feedback" else "message", text,
            role=role, event_id=unit.write_ref + ":" + field, episode_id=episode_ref,
            scope=unit.scope, extensions={"source_field": field}, **stamp))
        existing.add(text)
    return [replace(event, scope=unit.scope, episode_id=episode_ref) for event in rows]


class EpisodeNativeAdapter(BaseMemoryMethod):
    """Accumulate complete native conversations without changing Agent calls.

    A task can contain many writes. Boundaries may arrive before the complete
    experience (OpenHospital). New events after a boundary form a new native
    session; already ingested events are never resubmitted as growing prefixes.
    Feedback-only submissions retain provenance and do not invent RL learning.
    """
    method_id: str
    level_ref: str
    native_revision: str
    default_top_k: int = 10

    def __init__(self, config, services):
        self.params, self.services = dict(config.params), services
        self.root = storage_path(method_data_dir(services, self.method_id, self.params.get("storage_path")))
        # Immutable checkpoint storage is deliberately outside live DB accounting.
        self.snapshots = storage_path(Path(services.artifact_dir) / "adapter_snapshots" / self.method_id)
        self.scopes = {}
        self.backends = {}

    def _scope(self, scope):
        scope = normalize_scope(scope)
        key = digest(scope)
        if key not in self.scopes:
            self.scopes[key] = {"scope": scope, "episodes": {}, "seen_events": [], "seen_writes": [], "provenance": []}
        return key, self.scopes[key]

    async def _backend(self, key):
        if key not in self.backends:
            self.backends[key] = await asyncio.to_thread(self._make_backend, self.root / key, self.scopes[key]["scope"])
        return self.backends[key]

    async def index(self, *, scope):
        return [{"level_ref": self.level_ref, "description": self.view_description,
                 "scope": dict(scope), "granularity": self.granularity}]

    async def add(self, unit, *, learning_episode=None):
        key, owner = self._scope(unit.scope or {})
        if unit.write_ref in owner["seen_writes"]:
            return
        ref = unit.episode_ref
        if ref is None:
            for related in unit.related_refs:
                match = next((p for p in owner["provenance"] if related == p["write_ref"]), None)
                if match:
                    ref = match["episode_ref"]
                    break
        ref = ref or unit.write_ref
        episode = owner["episodes"].setdefault(ref, {"events": [], "boundary": False, "sessions": 0})
        known = set(owner["seen_events"]) | {e["event_id"] for e in episode["events"]}
        events = evidence(unit, ref)
        for event in events:
            text = content_text(event.content)
            if text.strip() and event.event_id not in known:
                episode["events"].append(event.to_dict())
                known.add(event.event_id)
        # No reward/success is inferred; sidecar keeps caller-supplied evidence.
        provenance = {"write_ref": unit.write_ref, "episode_ref": ref,
                      "event_ids": [e.event_id for e in events],
                      "related_refs": list(unit.related_refs), "feedback": deepcopy(dict(unit.feedback))}
        if unit.type == "experience":
            episode["boundary"] = True
        if episode["boundary"] or unit.episode_ref is None:
            await self._commit(key, ref)
        owner["provenance"].append(provenance)
        owner["seen_writes"].append(unit.write_ref)

    async def _commit(self, key, ref):
        owner = self.scopes[key]
        episode = owner["episodes"][ref]
        if not episode["events"]:
            return
        events = [MemoryEvent.from_dict(row) for row in episode["events"]]
        backend = await self._backend(key)
        await asyncio.to_thread(backend.ingest, events, ref, episode["sessions"])
        owner["seen_events"].extend(e.event_id for e in events)
        episode["events"] = []
        episode["sessions"] += 1

    async def episode_end(self, episode_ref, *, scope, status, learning_episode=None):
        key, owner = self._scope(scope)
        episode = owner["episodes"].setdefault(episode_ref, {"events": [], "boundary": False, "sessions": 0})
        episode["boundary"] = True
        await self._commit(key, episode_ref)

    async def flush(self):
        """Submit unfinished buffered sessions, without assigning success/reward."""
        for key, owner in self.scopes.items():
            for ref in owner["episodes"]:
                await self._commit(key, ref)

    async def search(self, query, *, level_ref, top_k, scope, extensions):
        key, _ = self._scope(scope)
        if level_ref not in (None, self.level_ref):
            raise ValueError(f"Unknown {self.method_id} access view")
        if key not in self.backends:
            return MemoryContribution()
        rows, trace = await asyncio.to_thread(self.backends[key].search, content_text(query), top_k or self.params.get("top_k", self.default_top_k))
        for row in rows:
            row["level_ref"] = self.level_ref
        return MemoryContribution(items=rows, extensions={self.method_id: {"trace": trace}})

    async def detail(self, memory_id, *, scope):
        key, _ = self._scope(scope)
        if key not in self.backends:
            return MemoryContribution()
        rows = await asyncio.to_thread(self.backends[key].detail, memory_id)
        for row in rows:
            row["level_ref"] = self.level_ref
        return MemoryContribution(items=rows)

    async def save(self):
        native = {key: await asyncio.to_thread(backend.export_state, self.snapshots)
                  for key, backend in self.backends.items()}
        body = {"scopes": self.scopes, "native": native}
        return {"version": 1, "method_id": self.method_id, "native_revision": self.native_revision,
                "state": put_blob(self.snapshots, canonical(body))}

    async def load(self, state):
        if state.get("version") != 1 or state.get("method_id") != self.method_id or state.get("native_revision") != self.native_revision:
            raise ValueError(f"Invalid {self.method_id} checkpoint")
        body = json.loads(get_blob(state["state"]))
        for key, owner in body["scopes"].items():
            if key != digest(normalize_scope(owner["scope"])):
                raise ValueError("Checkpoint scope identity mismatch")
        if set(body["native"]) - set(body["scopes"]):
            raise ValueError("Native checkpoint has unknown scope")
        # Validate every file before changing a running backend.
        for native in body["native"].values():
            self.backend_type.validate_state(native)
        for key, backend in self.backends.items():
            await asyncio.to_thread(backend.close if key in body["native"] else backend.reset)
        self.backends = {}
        self.scopes = body["scopes"]
        for key, native in body["native"].items():
            backend = await asyncio.to_thread(self._make_backend, self.root / key, self.scopes[key]["scope"], native)
            self.backends[key] = backend

    async def reset(self, scope):
        key, _ = self._scope(scope)
        if key in self.backends:
            await asyncio.to_thread(self.backends.pop(key).reset)
        self.scopes.pop(key, None)

    async def close(self):
        # Close does not ingest unfinished tasks. Call flush explicitly if wanted.
        for backend in self.backends.values():
            await asyncio.to_thread(backend.close)
