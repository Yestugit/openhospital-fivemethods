"""Adapter contracts only. The injected recorder is not a memory algorithm."""
from copy import deepcopy
from datetime import datetime, timezone
import importlib
import pytest
from mematlas.core.config import MethodConfig
from mematlas.core.models import MemoryEvent, MemoryUnit
from mematlas.core.plugin import MethodServices


class Recorder:
    def __init__(self):
        self.rows = []
        self.frozen = False
        self.last_query = None
    async def add(self, events, scope, episode_ref, metadata):
        assert not self.frozen
        self.rows.extend(e.to_dict() for e in events)
    async def search(self, query, scope, view, top_k):
        self.last_query = (query, scope, view, top_k)
        return [{"id": "native-result", "content": "native-result", "memory_type": view}]
    async def flush(self): pass
    async def freeze(self): self.frozen = True
    async def thaw(self): self.frozen = False
    async def export_state(self): return deepcopy(self.rows)
    async def import_state(self, state): self.rows = deepcopy(state)
    async def reset(self, scope): self.rows = [r for r in self.rows if r["scope"] != scope]


def adapter():
    module = importlib.import_module("mematlas.methods.memos.method")
    backend = Recorder()
    method = module.MemOSMemory(MethodConfig("memos"), MethodServices(objects={"memos.backend": backend}))
    return method, backend


@pytest.mark.asyncio
async def test_boundary_before_experience_deduplicates_actual_events_and_preserves_feedback():
    method, backend = adapter()
    scope = {"user_id": "alice"}
    event = MemoryEvent("message", "penicillin allergy", role="user", event_id="e1", scope=scope,
        timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc))
    first = MemoryUnit(events=[event], write_ref="w1", episode_ref="task1", scope=scope)
    await method.add(first)
    await method.episode_end("task1", scope=scope, status="completed")
    delayed = MemoryUnit(type="experience", events=[event], task="penicillin allergy",
        feedback={"success": False, "text": "Correction: verify allergy history"},
        write_ref="w2", episode_ref="task1", related_refs=["w1"], scope=scope)
    await method.add(delayed)
    await method.add(delayed)
    assert sum(row["event_id"] == "e1" for row in backend.rows) == 1
    assert len(backend.rows) == 2
    assert "Correction" in backend.rows[-1]["content"]
    assert all("native-result" not in row["content"] for row in backend.rows)
    other = MemoryUnit(events=[event], write_ref="w1", episode_ref="task1", scope={"user_id": "bob"})
    await method.add(other)
    assert len(backend.rows) == 3


@pytest.mark.asyncio
async def test_views_frozen_restoration_and_native_checkpoint_identity():
    method, backend = adapter()
    scope = {"user_id": "alice"}
    views = await method.index(scope=scope)
    assert views[0]["level_ref"] == "default"
    await method._freeze()
    state = await method.save()
    assert state == await method.save()
    await method.load(state)
    assert backend.frozen and method._frozen == 1
    await method.search("allergy", level_ref=views[-1]["level_ref"], top_k=None, scope=scope, extensions={})
    assert backend.last_query[2] == views[-1]["level_ref"]
    await method._thaw()
    assert not backend.frozen
    bad = deepcopy(state)
    bad["method_id"] = "different"
    with pytest.raises(ValueError): await method.load(bad)
    with pytest.raises(ValueError):
        await method.search("allergy", level_ref="unknown", top_k=None, scope=scope, extensions={})


@pytest.mark.asyncio
async def test_reused_write_reference_with_changed_payload_is_rejected():
    method, backend = adapter()
    await method.add(MemoryUnit(task="first", write_ref="one", scope={}))
    with pytest.raises(ValueError, match="different content"):
        await method.add(MemoryUnit(task="changed", write_ref="one", scope={}))
