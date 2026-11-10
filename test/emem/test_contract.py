"""Adapter boundary tests; these do not claim algorithm/native acceptance."""
import json
from copy import deepcopy

import pytest

from mematlas.core.config import MethodConfig
from mematlas.core.models import MemoryContribution, MemoryEvent, MemoryUnit
from mematlas.core.plugin import MethodServices
from mematlas.methods.emem.method import EMemMemory
from mematlas.methods.emem.utils import canonical, get_blob, put_blob
from mematlas.methods.memora.method import MemoraMemory


class BoundaryBackend:
    def __init__(self, directory, scope, params, services, state=None):
        self.records = json.loads(get_blob(state["records"])) if state else []
        self.last_top_k = None

    def ingest(self, events, ref, part):
        self.records.append({"events": [event.to_dict() for event in events], "ref": ref, "part": part})

    def search(self, query, top_k):
        self.last_top_k = top_k
        return [{"memory_id": str(i), "content": row["events"][0]["content"]} for i, row in enumerate(self.records)], []

    def detail(self, memory_id):
        return []

    def export_state(self, directory):
        return {"records": put_blob(directory, canonical(self.records))}

    @staticmethod
    def validate_state(state):
        get_blob(state["records"])

    def reset(self):
        self.records.clear()

    def close(self):
        pass


@pytest.fixture(params=[EMemMemory, MemoraMemory])
def adapter(request, tmp_path):
    method = request.param
    instance = method(MethodConfig(method.method_id), MethodServices(artifact_dir=tmp_path))
    instance.backend_type = BoundaryBackend
    return instance


def unit(write, episode, text, scope=None, **kwargs):
    return MemoryUnit(write_ref=write, episode_ref=episode,
        events=[MemoryEvent("message", text, event_id=text, role="user")],
        scope=scope or {"user_id": "a", "agent_id": "doctor"}, **kwargs)


@pytest.mark.asyncio
async def test_multiwrite_boundary_and_late_experience(adapter):
    owner = {"user_id": "a", "agent_id": "doctor"}
    await adapter.add(unit("w1", "e", "one"))
    await adapter.add(unit("w2", "e", "two"))
    assert not adapter.backends
    await adapter.episode_end("e", scope=owner, status="aborted")
    backend = next(iter(adapter.backends.values()))
    assert [[e["content"] for e in row["events"]] for row in backend.records] == [["one", "two"]]
    await adapter.add(MemoryUnit(write_ref="experience", episode_ref="e", type="experience",
        events=[MemoryEvent("message", text, event_id=text) for text in ["one", "two", "three"]], scope=owner))
    assert [e["content"] for e in backend.records[1]["events"]] == ["three"]
    assert all(not row.get("success") for row in backend.records)


@pytest.mark.asyncio
async def test_boundary_before_complete_experience_and_no_flush_on_save(adapter):
    owner = {"user_id": "a", "agent_id": "doctor"}
    await adapter.episode_end("later", scope=owner, status="completed")
    await adapter.add(unit("w", "later", "later-data", type="experience"))
    assert len(next(iter(adapter.backends.values())).records) == 1
    await adapter.add(unit("pending", "pending", "unfinished"))
    before = await adapter.save()
    assert before == await adapter.save()
    assert len(next(iter(adapter.backends.values())).records) == 1
    await adapter.flush()
    assert len(next(iter(adapter.backends.values())).records) == 2
    await adapter.load(before)
    assert before == await adapter.save()


@pytest.mark.asyncio
async def test_scope_reset_checkpoint_and_default_topk(adapter):
    one, two = {"user_id": "same", "session_id": "1"}, {"user_id": "same", "session_id": "2"}
    for scope in [one, two]:
        await adapter.add(unit("write", "episode", "event", scope, type="experience"))
    assert len(adapter.backends) == 2
    await adapter.search("q", level_ref=None, top_k=None, scope=one, extensions={})
    first_key = adapter._scope(one)[0]
    assert adapter.backends[first_key].last_top_k == (30 if adapter.method_id == "memora" else 10)
    before = await adapter.save()
    await adapter.reset(one)
    assert len(adapter.backends) == 1
    await adapter.load(before)
    assert len(adapter.backends) == 2
    assert before == await adapter.save()
    changed = deepcopy(before)
    changed["native_revision"] = "other-source"
    with pytest.raises(ValueError, match="checkpoint"):
        await adapter.load(changed)


@pytest.mark.asyncio
async def test_feedback_provenance_and_duplicate_write(adapter):
    one = unit("w", "ep", "one", type="experience")
    await adapter.add(one)
    await adapter.add(one)
    await adapter.add(MemoryUnit(write_ref="feedback", type="feedback", feedback={"success": False, "text": "real"},
        related_refs=("w",), scope=one.scope))
    key, owner = adapter._scope(one.scope)
    assert len(adapter.backends[key].records) == 2
    assert len(adapter.backends[key].records[1]["events"]) == 1
    assert json.loads(adapter.backends[key].records[1]["events"][0]["content"]) == {"success": False, "text": "real"}
    assert owner["provenance"][-1]["episode_ref"] == "ep"
    assert owner["provenance"][-1]["feedback"]["success"] is False


@pytest.mark.asyncio
async def test_full_tool_result_independent_output_and_late_feedback(adapter):
    scope = {"user_id": "a", "agent_id": "doctor"}
    tool = "real-tool-result:" + "evidence" * 9000
    await adapter.episode_end("hospital", scope=scope, status="completed")
    original = MemoryUnit(write_ref="clinical", episode_ref="hospital", type="experience", scope=scope,
        task="Evaluate medication safety", output={"diagnosis": "Observed allergy"},
        events=[MemoryEvent("tool_result", tool, event_id="actual-tool", role="tool")],
        applied_memory=[MemoryContribution(items=[{"memory_id": "context", "content": "retrieved context is not new evidence"}])])
    await adapter.add(original)
    backend = next(iter(adapter.backends.values()))
    received = backend.records[0]["events"]
    assert [row["event_id"] for row in received] == ["actual-tool", "clinical:task", "clinical:output"]
    assert received[0]["content"] == tool
    assert json.loads(received[2]["content"]) == {"diagnosis": "Observed allergy"}
    await adapter.add(MemoryUnit(write_ref="late-feedback", related_refs=("clinical",), type="feedback", scope=scope,
        events=original.events, feedback={"text": "Do not prescribe penicillin"}))
    later = backend.records[1]["events"]
    assert [row["event_id"] for row in later] == ["late-feedback:feedback"]
    assert json.loads(later[0]["content"])["text"] == "Do not prescribe penicillin"


@pytest.mark.asyncio
async def test_reject_corrupt_file_before_mutating_backend(adapter):
    await adapter.add(unit("w", "ep", "one", type="experience"))
    checkpoint = await adapter.save()
    ref = checkpoint["state"]
    from pathlib import Path
    Path(ref["path"]).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="SHA-256"):
        await adapter.load(checkpoint)
    assert len(next(iter(adapter.backends.values())).records) == 1
