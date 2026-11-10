"""Shared-shape ingestion helpers, private to this native adapter."""
from __future__ import annotations
import asyncio
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
from mematlas.core.models import MemoryEvent, MemoryUnit, content_text
from mematlas.core.scope import scope_key


def digest(value):
    return sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def evidence(unit):
    """Preserve actual events; supplemental fields are explicitly attributed, never retrieved context."""
    rows = list(unit.events)
    existing = {content_text(e.content) for e in rows}
    additions = [("task", unit.task, "user"), ("output", unit.output, "assistant"),
                 ("feedback", unit.feedback, "system")]
    for kind, value, role in additions:
        if value is None or value == {} or value == "":
            continue
        text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True)
        if text in existing:
            continue
        kwargs = {"timestamp": rows[0].timestamp} if rows else {}
        rows.append(MemoryEvent("feedback" if kind == "feedback" else "message", text,
            role=role, event_id=unit.write_ref + ":" + kind, episode_id=unit.episode_ref,
            scope=unit.scope, extensions={"source_field": kind}, **kwargs))
        existing.add(text)
    return [replace(e, scope=unit.scope, episode_id=unit.episode_ref or unit.write_ref) for e in rows
            if content_text(e.content).strip()]


def view_rows(levels, scope):
    return [{"level_ref": key, "description": description, "scope": dict(scope),
             "granularity": granularity} for key, description, granularity in levels]


class NativeAdapter:
    """Transaction-facing v3 semantics; native backends own all memory algorithms."""
    method_id = ""
    native_revision = "d326a6419467c2205041b58533b0ebfdf4f38173"
    levels = ()

    def _initialize(self, config, services):
        self.config, self.services = config, services
        self.params = {k: v.get("default") for k, v in self.descriptor.config_schema["properties"].items()
                       if "default" in v}
        self.params.update(config.params)
        self.backend = services.object(self.method_id + ".backend")
        self._owns_backend = self.backend is None
        self._seen = set()
        self._writes = {}
        self._boundaries = {}
        self._frozen = 0

    async def index(self, *, scope):
        return view_rows(self.levels, scope)

    async def add(self, unit, *, learning_episode=None):
        if self._frozen:
            raise RuntimeError("Native backend is frozen")
        owner = scope_key(unit.scope)
        key = owner + "\n" + unit.write_ref
        signature = digest(unit.to_dict())
        if key in self._writes:
            if self._writes[key] != signature:
                raise ValueError("write_ref was already accepted with different content")
            return
        pending = [event for event in evidence(unit) if (owner, event.event_id) not in self._seen]
        if pending:
            await self.backend.add(pending, unit.scope, unit.episode_ref or unit.write_ref,
                {"write_ref": unit.write_ref, "type": unit.type, "related_refs": list(unit.related_refs)})
            self._seen.update((owner, e.event_id) for e in pending)
        self._writes[key] = signature

    async def _learn(self, episode, *, observed_event_ids):
        await self.add(MemoryUnit.from_episode(episode))

    async def episode_end(self, episode_ref, *, scope, status, learning_episode=None):
        # OpenHospital can deliver this marker before its delayed experience.
        # A boundary never invents feedback or prohibits the later add().
        self._boundaries[scope_key(scope) + "\n" + episode_ref] = status
        if learning_episode is not None:
            await self._learn(learning_episode, observed_event_ids=frozenset())

    async def search(self, query, *, level_ref, top_k, scope, extensions):
        from mematlas.core.models import MemoryContribution
        text = content_text(query)
        if not text.strip():
            raise ValueError("Native memory search requires a nonempty text query")
        view = level_ref or "default"
        if view not in {row[0] for row in self.levels}:
            raise ValueError("Unknown memory view: " + view)
        items = await self.backend.search(text, scope, view, top_k if top_k is not None else self.params.get("top_k"))
        return MemoryContribution(items=items, extensions={self.method_id: {"level_ref": view}})

    async def flush(self):
        await self.backend.flush()

    async def _freeze(self):
        if self._frozen == 0:
            await self.backend.freeze()
        self._frozen += 1

    async def _thaw(self):
        if not self._frozen:
            raise RuntimeError("Unbalanced native thaw")
        if self._frozen == 1:
            await self.backend.thaw()
        self._frozen -= 1

    async def save(self):
        return deepcopy({"version": 1, "method_id": self.method_id, "native_revision": self.native_revision, "native": await self.backend.export_state(),
                         "seen": sorted(self._seen), "writes": self._writes, "boundaries": self._boundaries})

    async def load(self, state):
        if (state.get("version") != 1 or state.get("method_id") != self.method_id
                or state.get("native_revision") != self.native_revision):
            raise ValueError("Unsupported native checkpoint version")
        seen = {tuple(row) for row in state["seen"]}
        if any(len(row) != 2 or not all(isinstance(x, str) for x in row) for row in seen):
            raise ValueError("Invalid event identities in checkpoint")
        await self.backend.import_state(deepcopy(state["native"]))
        self._seen, self._writes, self._boundaries = seen, dict(state["writes"]), dict(state["boundaries"])
        # Lifecycle flags belong to the running transaction, never the checkpoint.

    async def reset(self, scope):
        owner = scope_key(scope)
        await self.backend.reset(scope)
        self._seen = {row for row in self._seen if row[0] != owner}
        self._writes = {k:v for k,v in self._writes.items() if not k.startswith(owner + "\n")}
        self._boundaries = {k:v for k,v in self._boundaries.items() if not k.startswith(owner + "\n")}

    async def close(self):
        if self._owns_backend and self.backend is not None:
            await self.backend.close()


def inspect_native_source(method_id, package, sources):
    """Read-only compatibility check of installed algorithm files against the pin."""
    from importlib.util import find_spec
    manifest = json.loads((Path(__file__).parent / "UPSTREAM.json").read_text(encoding="utf-8"))
    errors = []
    roots = {}
    for module in sources:
        spec = find_spec(module)
        if spec is None or not spec.submodule_search_locations:
            errors.append("Missing native package: " + module)
        else:
            roots[module] = Path(next(iter(spec.submodule_search_locations)))
    for relative, expected in manifest["source_files"].items():
        matched = False
        for module, prefix in sources.items():
            if relative.startswith(prefix):
                matched = True
                if module not in roots:
                    break
                source = roots[module] / relative[len(prefix):]
                if not source.is_file() or sha256(source.read_bytes()).hexdigest() != expected:
                    errors.append("Native source differs from pinned revision: " + relative)
                break
        if not matched:
            errors.append("Unmapped native source: " + relative)
    return {"revision": manifest["revision"], "checked_files": len(manifest["source_files"]),
            "errors": errors, "compatible": not errors}
