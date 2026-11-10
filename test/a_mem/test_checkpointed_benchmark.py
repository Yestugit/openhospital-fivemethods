"""Recover per-event history import through the existing experiment runner."""
from mematlas.evaluation.validation import submit_episode
import json

import pytest

from mematlas import ExperimentRunner, MemAtlasConfig, RuntimeConfig
from mematlas.core.files import atomic_write_json
from mematlas.methods.a_mem import AMemConfig
from mematlas.test.a_mem.check_benchmarks import FixtureReader, fixture
from mematlas.test.a_mem.fixtures import FixedDecisions, services


@pytest.mark.asyncio
async def test_history_ingestion_resumes_after_two_events_without_duplicate_notes(tmp_path):
    from mematlas.test.a_mem.run_locomo_validation import make_experiment
    data = fixture("locomo")
    data[0]["conversation"]["session_1"].append({"speaker": "Bob", "text": "third fact", "dia_id": "D1:3"})
    path = tmp_path / "synthetic.json"
    atomic_write_json(path, data)
    class FailOnThird(FixedDecisions):
        async def complete(self, messages, **options):
            if messages[-1]["content"].endswith("Bob: third fact"):
                raise RuntimeError("Controlled provider interruption")
            return await super().complete(messages, **options)
    supplied = services()
    supplied.llm = FailOnThird()
    supplied.objects["answer_llm"] = FixtureReader()
    memory = MemAtlasConfig(method=AMemConfig(), runtime=RuntimeConfig(artifact_dir=str(tmp_path / "memory")))
    first = make_experiment(memory, path, tmp_path / "first", test_limit=3)
    result = (await ExperimentRunner().evaluate(first, services=supplied))[0]
    assert result.failures
    snapshot_root = tmp_path / "first/runs/001-a-mem/snapshots"
    checkpoint = json.loads((snapshot_root / "learned_2.json").read_text(encoding="utf-8"))
    state = json.loads((snapshot_root / checkpoint["memory_file"]).read_text(encoding="utf-8"))["state"]
    assert len(state["core"]["memories"]) == 2
    clean = services()
    clean.objects["answer_llm"] = FixtureReader()
    resumed = make_experiment(memory, path, tmp_path / "resumed", test_limit=3)
    complete = (await ExperimentRunner().evaluate(resumed, services=clean, resume_from=tmp_path / "first"))[0]
    assert not complete.failures, complete.failures
    assert len(complete.predictions) == 3
    final_root = tmp_path / "resumed/runs/001-a-mem/snapshots"
    final = json.loads((final_root / "learned_3.json").read_text(encoding="utf-8"))
    final_state = json.loads((final_root / final["memory_file"]).read_text(encoding="utf-8"))["state"]
    assert len(final_state["core"]["memories"]) == 3
    assert set(final_state["seen_events"]) == {"D1:1", "D1:2", "D1:3"}
    from mematlas.test.a_mem.export_validation import export
    audit = export(tmp_path / "resumed", tmp_path / "audit", expected_history=3)
    assert audit["notes"] == 3 and audit["frozen_verified"]


@pytest.mark.asyncio
async def test_per_event_checkpointing_preserves_full_history_memory_semantics(tmp_path):
    from mematlas import Memory, MemoryEpisode
    from mematlas.benchmarks.locomo import LoCoMoBenchmark
    from mematlas.test.a_mem.run_locomo_validation import make_experiment
    path = tmp_path / "synthetic.json"
    atomic_write_json(path, fixture("locomo"))
    history = next(iter(LoCoMoBenchmark(path).cases()))
    memory = MemAtlasConfig(method=AMemConfig(), runtime=RuntimeConfig(artifact_dir=str(tmp_path / "memory")))
    async with Memory(memory, services=services(evolve=True)) as direct:
        await submit_episode(direct, MemoryEpisode(events=list(history.history)), scope=history.scope)
        expected = await direct._method.save()
    supplied = services(evolve=True)
    supplied.objects["answer_llm"] = FixtureReader()
    experiment = make_experiment(memory, path, tmp_path / "checkpointed")
    result = (await ExperimentRunner().evaluate(experiment, services=supplied))[0]
    assert not result.failures
    root = tmp_path / "checkpointed/runs/001-a-mem/snapshots"
    saved = json.loads((root / "learned_2.json").read_text(encoding="utf-8"))
    actual = json.loads((root / saved["memory_file"]).read_text(encoding="utf-8"))["state"]
    def semantic_notes(state):
        notes = state["core"]["memories"]
        return {tuple(note["source_event_ids"]): {
            "content": note["content"], "context": note["context"], "keywords": note["keywords"],
            "tags": note["tags"], "dependencies": sorted(note["evolution_source_event_ids"]),
            "links": sorted(tuple(notes[identity]["source_event_ids"]) for identity in note["links"]),
            "embedding": note["embedding"],
        } for note in notes.values()}
    assert semantic_notes(actual) == semantic_notes(expected)
    assert actual["core"]["evo_count"] == expected["core"]["evo_count"]
    assert actual["seen_events"] == expected["seen_events"]
