"""Public interface semantics, feedback attribution and persistent task boundaries."""
from copy import deepcopy
from dataclasses import replace

import pytest

from mematlas import (
    BaseMemoryMethod, Memory, MemoryContribution, MemoryEvent, MemoryUnit,
    MethodCapabilities, MethodDescriptor, MethodPlugin, Operation, PluginRegistry,
)
from mematlas.core.capabilities import IntegrationSlot, MemoryForm
from mematlas.core.errors import CapabilityMismatchError, MemoryStateError, MethodOperationError
from mematlas.core.models import Modality


class RecordingMethod(BaseMemoryMethod):
    def __init__(self):
        self.items, self.tasks, self.boundaries = {}, [], []

    async def _learn(self, episode, *, observed_event_ids):
        self.tasks.append(episode.to_dict())
        for event in episode.events:
            self.items[event.event_id] = {"content": event.content, "memory_id": event.event_id}
        if episode.extensions.get("fail"):
            raise ValueError("controlled write failure")

    async def search(self, query, *, level_ref, top_k, scope, extensions):
        return MemoryContribution(items=list(self.items.values())[:top_k],
            extensions={"recording": {"query": query}})

    async def episode_end(self, episode_ref, *, scope, status, learning_episode=None):
        await super().episode_end(episode_ref, scope=scope, status=status, learning_episode=learning_episode)
        self.boundaries.append((episode_ref, status))

    async def detail(self, memory_id, *, scope):
        return MemoryContribution(items=[self.items[memory_id]])

    async def update(self, memory_id, content, *, scope):
        self.items[memory_id]["content"] = content

    async def delete(self, memory_id, *, scope):
        del self.items[memory_id]

    async def save(self):
        return deepcopy({"items": self.items, "tasks": self.tasks, "boundaries": self.boundaries})

    async def load(self, state):
        self.items, self.tasks, self.boundaries = deepcopy((state["items"], state["tasks"], state["boundaries"]))


class RecordingPlugin(MethodPlugin):
    def __init__(self, *, task_learning=False, maintenance=True):
        self.options = {"task_learning": task_learning, "maintenance": maintenance}
        ops = {Operation.INDEX, Operation.ADD, Operation.SEARCH, Operation.EPISODE_END, Operation.SAVE, Operation.LOAD}
        if maintenance:
            ops.update({Operation.DETAIL, Operation.UPDATE, Operation.DELETE})
        self.descriptor = MethodDescriptor("recording", "Recording", "Controlled contract fixture",
            MethodCapabilities(frozenset({Modality.TEXT}), frozenset({MemoryForm.RECORD}), frozenset(ops),
                frozenset({IntegrationSlot.CONTEXT}), read_only_mode="native",
                required_episode_fields=frozenset({"task", "trajectory", "success"}) if task_learning else frozenset()))

    def create_method(self, config, services):
        return RecordingMethod()

    def construction_spec(self):
        return {"factory": "mematlas.test.test_memory_interface:RecordingPlugin", "kwargs": self.options}


def memory_at(path, *, executor="in_process", service_url=None, **options):
    registry = PluginRegistry()
    registry.register(RecordingPlugin(**options))
    return Memory({"method": {"method_id": "recording"}, "scope": {"user_id": "alice"},
                   "runtime": {"artifact_dir": str(path), "executor": executor,
                               "service_url": service_url}}, registry=registry)


@pytest.mark.asyncio
async def test_units_share_a_task_and_retry_does_not_repeat_writes_or_boundaries(tmp_path):
    first = MemoryUnit(events=[MemoryEvent("message", "question", role="user")])
    second = MemoryUnit(events=[MemoryEvent("message", "answer", role="assistant")])
    async with memory_at(tmp_path) as memory:
        one, two = await memory.add(first), await memory.add(second)
        assert one.episode_ref == two.episode_ref and one.write_ref != two.write_ref
        assert await memory.add(first) == one
        assert len(memory._method.tasks) == 2
        await memory.episode_end()
        await memory.episode_end(one.episode_ref)
        assert memory._method.boundaries == [(one.episode_ref, "completed")]
        three = await memory.add(MemoryUnit(events=[MemoryEvent("message", "new task")]))
        assert three.episode_ref != one.episode_ref
        assert memory.stats()["submitted_units"] == 3
        assert not any(hasattr(memory, name) for name in ("observe", "learn", "retrieve", "evolve"))


@pytest.mark.asyncio
async def test_execution_record_with_only_output_can_be_retried_without_synthetic_event_conflict(tmp_path):
    from mematlas import MemoryEpisode
    from mematlas.evaluation.validation import submit_episode
    episode = MemoryEpisode("result-only", task="solve", events=[], output="actual answer")
    async with memory_at(tmp_path) as memory:
        first = await submit_episode(memory, episode)
        assert await submit_episode(memory, episode) == first
        assert len(memory._method.tasks) == 1


@pytest.mark.asyncio
async def test_task_learning_waits_for_boundary_and_preserves_used_contributions(tmp_path):
    used = MemoryContribution(items=[{"content": "rule", "memory_id": "rule-1"}],
                              extensions={"credit": {"decision": "actual"}})
    async with memory_at(tmp_path, task_learning=True) as memory:
        start = await memory.add(MemoryUnit(type="plan", task="solve",
            events=[MemoryEvent("action", "inspect")], applied_memory=[used]))
        await memory.add(MemoryUnit(type="experience", output="done", feedback={"success": False, "reward": 0.0}))
        assert memory._method.tasks == []
        await memory.episode_end()
        assert len(memory._method.tasks) == 1
        task = memory._method.tasks[0]
        assert task["episode_id"] == start.episode_ref and task["success"] is False
        assert task["reward"] == 0.0 and task["applied_memory"] == [used.to_dict()]
        assert len(task["events"]) == 2


@pytest.mark.asyncio
async def test_late_feedback_after_save_restore_learns_once_without_reending(tmp_path):
    snapshot = tmp_path / "memory.json"
    async with memory_at(tmp_path / "first", task_learning=True) as memory:
        receipt = await memory.add(MemoryUnit(task="solve", events=[MemoryEvent("action", "work")]))
        await memory.episode_end(status="aborted")
        assert memory._method.tasks == []
        await memory.save(snapshot)
    feedback = MemoryUnit(type="feedback", feedback={"success": False, "text": "failed check"},
                          related_refs=(receipt.write_ref,))
    async with memory_at(tmp_path / "restored", task_learning=True) as memory:
        await memory.load(snapshot)
        assert (await memory.add(feedback)).episode_ref == receipt.episode_ref
        await memory.add(feedback)
        assert len(memory._method.tasks) == 1
        assert len(memory._method.boundaries) == 1
        assert memory._method.tasks[0]["success"] is False
        assert memory.stats()["episodes"]["learned_exposures"] == 1


@pytest.mark.asyncio
async def test_feedback_cannot_attribute_another_tasks_submission(tmp_path):
    async with memory_at(tmp_path, task_learning=True) as memory:
        first = await memory.add(MemoryUnit(task="first", events=[MemoryEvent("action", "work")]))
        await memory.episode_end()
        second = await memory.add(MemoryUnit(task="second", events=[MemoryEvent("action", "work again")]))
        before = memory._history.snapshot()
        with pytest.raises(ValueError, match="different task"):
            await memory.add(MemoryUnit(type="feedback", episode_ref=second.episode_ref,
                related_refs=(first.write_ref,), feedback={"success": True}))
        assert memory._history.snapshot() == before
        assert not memory._method.tasks


@pytest.mark.asyncio
async def test_generated_event_times_are_current_and_stable_after_restore(tmp_path):
    from datetime import datetime, timezone
    unit = MemoryUnit(output="observed output")
    feedback = MemoryUnit(type="feedback", related_refs=(unit.write_ref,), feedback={"success": True})
    start = datetime.now(timezone.utc)
    async with memory_at(tmp_path / "source") as memory:
        receipt = await memory.add(unit)
        await memory.episode_end()
        await memory.add(feedback)
        assert all(MemoryEvent.from_dict(row).timestamp >= start for row in memory._history.events.values())
        await memory.save(tmp_path / "state.json")
    async with memory_at(tmp_path / "restored") as memory:
        await memory.load(tmp_path / "state.json")
        before = await memory._method.save()
        assert await memory.add(unit) == receipt
        await memory.add(feedback)
        assert await memory._method.save() == before


@pytest.mark.asyncio
async def test_task_description_and_feedback_do_not_invent_an_execution_trajectory(tmp_path):
    async with memory_at(tmp_path, task_learning=True) as memory:
        receipt = await memory.add(MemoryUnit(task="solve", feedback={"success": True}))
        await memory.episode_end()
        assert not memory._method.tasks
        await memory.add(MemoryUnit(type="experience", episode_ref=receipt.episode_ref,
            events=[MemoryEvent("action", "actually executed")]))
        assert len(memory._method.tasks) == 1


@pytest.mark.asyncio
async def test_reads_maintenance_scope_and_frozen_protection(tmp_path):
    async with memory_at(tmp_path) as memory:
        await memory.add(MemoryUnit(events=[MemoryEvent("message", "old content")]))
        assert (await memory.index())[0]["level_ref"] == "default"
        result = await memory.search("content", level_ref="default")
        assert result.extensions["recording"]["query"] == "content"
        ref = result.items[0]["memory_ref"]
        await memory.update(ref, "new content")
        assert (await memory.detail(ref)).items[0]["content"] == "new content"
        with pytest.raises(ValueError, match="scope"):
            await memory.detail(ref, scope={"user_id": "bob"})
        with pytest.raises(CapabilityMismatchError):
            await memory.search("content", level_ref="invented")
        before = await memory._method.save()
        async with memory.read_only():
            await memory.search("content")
            await memory.detail(ref)
            for operation in (lambda: memory.add(MemoryUnit(task="forbidden")),
                              memory.episode_end, lambda: memory.update(ref, "forbidden"), lambda: memory.delete(ref)):
                with pytest.raises(MemoryStateError):
                    await operation()
        assert await memory._method.save() == before
        await memory.delete(ref)
        assert not (await memory.search("content")).items


@pytest.mark.asyncio
async def test_failed_native_write_rolls_back_submission_and_native_state(tmp_path):
    async with memory_at(tmp_path) as memory:
        before = memory._history.snapshot()
        with pytest.raises(MethodOperationError):
            await memory.add(MemoryUnit(task="failure", extensions={"fail": True}))
        assert memory._history.snapshot() == before
        assert memory._method.tasks == []
        assert memory.state.value == "started"


@pytest.mark.asyncio
async def test_unsupported_maintenance_and_conflicting_submission_are_explicit(tmp_path):
    async with memory_at(tmp_path, maintenance=False) as memory:
        unit = MemoryUnit(task="original", write_ref="same")
        await memory.add(unit)
        before = memory._history.snapshot()
        with pytest.raises(ValueError, match="write_ref"):
            await memory.add(MemoryUnit(task="changed", write_ref="same"))
        assert memory._history.snapshot() == before
        with pytest.raises(CapabilityMismatchError):
            await memory.delete("unknown")


@pytest.mark.asyncio
async def test_event_extensions_are_validated_and_preserved(tmp_path):
    plugin = RecordingPlugin()
    plugin.descriptor = replace(plugin.descriptor, input_schemas={"event": {
        "type": "object",
        "properties": {"fixture": {"type": "object", "properties": {
            "source": {"type": "string"}}, "required": ["source"], "additionalProperties": False}},
        "required": [], "additionalProperties": False,
    }})
    registry = PluginRegistry()
    registry.register(plugin)
    config = {"method": {"method_id": "recording"}, "scope": {"user_id": "alice"},
              "runtime": {"artifact_dir": str(tmp_path)}}

    async with Memory(config, registry=registry) as memory:
        event = MemoryEvent("message", "recorded", extensions={"fixture": {"source": "agent"}})
        await memory.add(MemoryUnit(task="task", events=[event]))
        assert memory._method.tasks[-1]["events"][0]["extensions"] == {"fixture": {"source": "agent"}}

        before = memory._history.snapshot()
        invalid = MemoryUnit(task="invalid", events=[MemoryEvent("message", "bad", extensions={
            "fixture": {"source": 3}})])
        with pytest.raises(CapabilityMismatchError, match="event.extensions.fixture.source"):
            await memory.add(invalid)
        assert memory._history.snapshot() == before

        unknown = MemoryUnit(task="unknown", events=[MemoryEvent("message", "bad", extensions={
            "fixture": {"source": "agent", "other": True}})])
        with pytest.raises(CapabilityMismatchError, match="unknown fields: other"):
            await memory.add(unknown)
        assert memory._history.snapshot() == before


@pytest.mark.asyncio
async def test_subprocess_transfers_units_outcomes_references_and_boundaries(tmp_path):
    async with memory_at(tmp_path, executor="subprocess", task_learning=True) as memory:
        await memory.add(MemoryUnit(task="solve", events=[MemoryEvent("action", "work")],
                                    feedback={"success": True}))
        assert not (await memory.search("work")).items
        await memory.episode_end()
        assert (await memory.index())[0]["level_ref"] == "default"
        result = await memory.search("work")
        ref = result.items[0]["memory_ref"]
        await memory.update(ref, "corrected")
        assert (await memory.detail(ref)).items[0]["content"] == "corrected"
        await memory.delete(ref)
        assert not (await memory.search("work")).items


@pytest.mark.asyncio
async def test_http_executor_transfers_new_operations_and_preserves_run_identity(tmp_path):
    import asyncio
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    import json
    from threading import Thread
    from mematlas.runtime.execution.worker import Worker

    worker = Worker()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            self.respond(asyncio.run(worker.dispatch("stats", {})))

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            operation = self.path.lstrip("/")
            result = asyncio.run(worker.dispatch(operation, payload))
            if operation == "setup":
                result = {"run_id": payload["context"]["run_id"]}
            self.respond(result)

        def respond(self, result):
            body = json.dumps({"result": result}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        async with memory_at(tmp_path, executor="service",
                             service_url=f"http://127.0.0.1:{server.server_port}") as memory:
            await memory.add(MemoryUnit(events=[MemoryEvent("message", "source")]))
            await memory.episode_end()
            assert (await memory.index())[0]["level_ref"] == "default"
            ref = (await memory.search("source")).items[0]["memory_ref"]
            await memory.update(ref, "corrected")
            assert (await memory.detail(ref)).items[0]["content"] == "corrected"
            await memory.delete(ref)
            assert not (await memory.search("source")).items
    finally:
        await asyncio.to_thread(server.shutdown)
        server.server_close()
        thread.join(timeout=5)


@pytest.mark.asyncio
async def test_native_mem0_unit_and_explicit_content_maintenance(tmp_path):
    from mematlas.test.mem0.native_support import require_native, config_for, services
    require_native()
    async with Memory(config_for(tmp_path, infer=False), services=services()) as memory:
        await memory.add(MemoryUnit(events=[MemoryEvent("message", "Helios port 8443", role="user")]))
        await memory.episode_end()
        result = await memory.search("Helios")
        ref = result.items[0]["memory_ref"]
        await memory.update(ref, "Helios port 9443")
        assert (await memory.detail(ref)).items[0]["content"] == "Helios port 9443"
        await memory.delete(ref)
        assert not (await memory.search("Helios")).items


@pytest.mark.asyncio
async def test_memoryos_exposes_real_levels_and_filters_context_with_sources(tmp_path):
    config = {"method": {"method_id": "memoryos", "params": {"implementation_mode": "offline"}},
              "scope": {"user_id": "alice"}, "runtime": {"artifact_dir": str(tmp_path)}}
    async with Memory(config) as memory:
        receipt = await memory.add(MemoryUnit(events=[MemoryEvent("message", "Helios port 8443", role="user"),
                                            MemoryEvent("message", "Recorded", role="assistant")]))
        await memory.episode_end()
        assert {row["level_ref"] for row in await memory.index()} == {
            "default", "short_term", "mid_term", "long_term"}
        result = await memory.search("Helios", level_ref="short_term")
        assert result.items and all(row["level_ref"] == "short_term" for row in result.items)
        assert all(row["source_item_indices"] for row in result.context)
        assert not (await memory.search("Helios", level_ref="long_term")).items
        await memory.add(MemoryUnit(type="feedback", related_refs=(receipt.write_ref,),
            feedback={"success": True, "reward": 1.0}))


@pytest.mark.asyncio
@pytest.mark.parametrize("method_id", ["a-mem", "expel", "awm", "reasoningbank", "memcon", "ael"])
async def test_native_method_adapters_accept_the_same_submission_flow(tmp_path, method_id):
    from mematlas import MethodServices
    params, extensions = {}, {}
    supplied = MethodServices()
    if method_id == "a-mem":
        pytest.importorskip("chromadb")
        from mematlas.test.a_mem.fixtures import services
        supplied = services()
    elif method_id == "expel":
        pytest.importorskip("tiktoken")
        pytest.importorskip("faiss")
        from mematlas.test.expel.test_expel_integration import services
        supplied = services()
        params = {"benchmark": "hotpotqa"}
        extensions = {"expel": {"end_of_training": True}}
    elif method_id == "awm":
        from mematlas.test.awm.awm_fixtures import services
        supplied = services()
        params = {"profile": "mind2web"}
        extensions = {"awm": {"website": "shop", "domain": "Shopping", "subdomain": "General",
                              "template_id": 1, "end_of_training": True}}
    elif method_id == "reasoningbank":
        from mematlas.providers import CallableLLM
        supplied = MethodServices(llm=CallableLLM(lambda *args, **kwargs:
            "# Memory Item 1\n## Title Verify evidence\n## Description When answering\n## Content Check the observed result."))
        params = {"implementation_mode": "offline"}
    config = {"method": {"method_id": method_id, "params": params}, "scope": {"user_id": "alice"},
              "runtime": {"artifact_dir": str(tmp_path)}}
    async with Memory(config, services=supplied) as memory:
        query_extensions = {"awm": {"website": "shop"}} if method_id == "awm" else {}
        used = await memory.search("Find the verified value", extensions=query_extensions)
        action_extensions = {"awm": {"parsed_action": "fill('1', 'item')"}} if method_id == "awm" else {}
        receipt = await memory.add(MemoryUnit(type="experience", task="Find the verified value",
            events=[MemoryEvent("observation", "[textbox 1] Search", event_id="seen"),
                    MemoryEvent("action", "fill('1', 'item')", event_id="act", extensions=action_extensions)],
            output="done", feedback={"success": True, "reward": 0.8},
            applied_memory=[used], extensions=extensions))
        await memory.episode_end(receipt.episode_ref)
        assert memory.stats()["episodes"]["learned_exposures"] == 1
        await memory.search("A new task", extensions=query_extensions)
        await memory.save(tmp_path / "state.json")
        async with memory.read_only():
            await memory.search("A new task", extensions=query_extensions)


@pytest.mark.asyncio
async def test_memcon_keeps_task_context_across_step_queries_and_multiple_units(tmp_path):
    config = {"method": {"method_id": "memcon"}, "scope": {"user_id": "alice"},
              "runtime": {"artifact_dir": str(tmp_path)}}
    async with Memory(config) as memory:
        first = await memory.search("Organize the room")
        receipt = await memory.add(MemoryUnit(task="Organize the room", events=[
            MemoryEvent("episode_start", "Organize the room", extensions={"task": "Organize the room"}),
            MemoryEvent("action", "inspect table", extensions={"action": "inspect table", "reward": 0.0}),
            MemoryEvent("observation", "a cup", extensions={"done": False})], applied_memory=[first]))
        second = await memory.search("Where should the cup go?")
        await memory.add(MemoryUnit(events=[MemoryEvent("action", "move cup"),
            MemoryEvent("observation", "cup on shelf")], feedback={"success": True}, applied_memory=[second]))
        native = next(iter(memory._method._instances.values()))
        assert native._task_index == 1
        assert len(native._mdp.get_episode_actions()) == 2
        await memory.episode_end(receipt.episode_ref)
        assert native._task_index == 1
        assert native.records[0]["trajectory"] == "> inspect table\na cup\n> move cup\ncup on shelf"
        assert sum(native.policy["other"]["visits"].values()) == 2


@pytest.mark.asyncio
async def test_memcon_restores_pending_task_boundary_and_delayed_retrieval_credit(tmp_path):
    def config(path):
        return {"method": {"method_id": "memcon"}, "scope": {"user_id": "alice"},
                "runtime": {"artifact_dir": str(path)}}
    async with Memory(config(tmp_path / "source")) as memory:
        used = await memory.search("Organize the room")
        receipt = await memory.add(MemoryUnit(task="Organize the room", events=[
            MemoryEvent("action", "inspect"), MemoryEvent("observation", "clear")], applied_memory=[used]))
        again = await memory.search("Organize the room")
        await memory.add(MemoryUnit(type="plan", events=[MemoryEvent("message", "consider the shelf")],
                                    applied_memory=[again]))
        await memory.episode_end()
        await memory.save(tmp_path / "state.json")
    async with Memory(config(tmp_path / "restored")) as memory:
        await memory.load(tmp_path / "state.json")
        await memory.search("Arrange the desk")
        native = next(iter(memory._method._instances.values()))
        assert native.task == "Arrange the desk"
        await memory.add(MemoryUnit(type="feedback", related_refs=(receipt.write_ref,),
                                    feedback={"success": True}))
        assert native.records[0]["task"] == "Organize the room"
        assert native.records[0]["trajectory"] == "> inspect\nclear"
        assert sum(native.policy["other"]["visits"].values()) == 2


@pytest.mark.asyncio
async def test_memrl_native_feedback_batching_and_snapshots_use_new_interface(tmp_path, monkeypatch):
    import tempfile
    from mematlas import MethodServices
    # The upstream fixture creates import-time logs and temporary native cubes.
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    # A source directory named memos is not the legacy MemoryOS dependency.
    pytest.importorskip("memos.configs.mem_cube")
    from mematlas.test.memrl.fixtures import NativeHarness
    harness = NativeHarness(tmp_path / "native")
    services = MethodServices(objects={"memrl.service_factory": harness.factory})
    config = {"method": {"method_id": "memrl", "params": {"batch_size": 2}},
              "scope": {"user_id": "alice"}, "runtime": {"artifact_dir": str(tmp_path / "run")}}
    async with Memory(config, services=services) as memory:
        receipt = await memory.add(MemoryUnit(type="experience", task="Verify the value",
            events=[MemoryEvent("action", "Check the observed result")], feedback={"success": True},
            extensions={"memrl": {"end_of_training": True}}))
        await memory.episode_end(receipt.episode_ref)
        assert memory.stats()["episodes"]["learned_exposures"] == 1
        await memory.search("Verify the value")
        await memory.save(tmp_path / "state.json")
        async with memory.read_only():
            await memory.search("Verify the value")


@pytest.mark.asyncio
@pytest.mark.parametrize("timing,delay", [("immediate", 0), ("batch", 0), ("delayed", 1)])
async def test_hospital_ends_each_case_and_delivers_feedback_at_original_schedule(tmp_path, timing, delay):
    from mematlas.integrations.openhospital.bridge import OpenHospitalMemoryIntegration

    core = memory_at(tmp_path, task_learning=True)
    bridge = OpenHospitalMemoryIntegration(core, feedback_timing=timing, feedback_delay=delay)
    report = {"diagnosisAccuracy": 0.8, "examinationPrecision": 0.7, "treatmentOverallScore": 0.9}
    state = {"timeline": [{"kind": "message", "role": "user", "text": "symptoms"}]}
    try:
        await bridge.start()
        bridge.configure_training_batch(["p1", "p2"], batch_index=1)
        for patient in ("p1", "p2"):
            await bridge.record_execution(patient_id=patient, case_state=state, final_result={"diagnosis": "result"})
            await bridge.learn_case(patient_id=patient, case_state=state, final_result={"diagnosis": "result"},
                                    reflection={"future_strategy": "check evidence"}, evaluation_report=report)
        assert len(core._method.boundaries) == 2
        assert len(core._method.tasks) == {"immediate": 2, "batch": 0, "delayed": 1}[timing]
        if timing == "batch":
            await bridge.deliver_feedback(batch_end=True)
            assert len(core._method.tasks) == 2
        assert all(task["success"] is True and task["reward"] > 0 for task in core._method.tasks)
        assert len(core._method.boundaries) == 2
    finally:
        await bridge.close()


@pytest.mark.asyncio
async def test_hospital_cached_completion_does_not_release_or_erase_batch_feedback(tmp_path):
    from mematlas.integrations.openhospital.bridge import OpenHospitalMemoryIntegration
    report = {"diagnosisAccuracy": 0.8, "examinationPrecision": 0.7, "treatmentOverallScore": 0.9}
    state = {"timeline": [{"kind": "message", "role": "user", "text": "symptoms"}]}
    bridge = OpenHospitalMemoryIntegration(memory_at(tmp_path / "source", task_learning=True), feedback_timing="batch")
    try:
        await bridge.start()
        bridge.configure_training_batch(["patient"], batch_index=1)
        # A cached encounter can enter learn_case without a new doctor execution.
        await bridge.learn_case(patient_id="patient", case_state=state, final_result={"diagnosis": "result"},
            reflection={"future_strategy": "check evidence"}, evaluation_report=report)
        await bridge.record_execution(patient_id="patient", case_state=state, final_result={"diagnosis": "result"})
        assert len(bridge.core_memory._method.boundaries) == 1
        assert not bridge.core_memory._method.tasks
        assert bridge._bundles[bridge._occurrence("patient")]["evaluation_report"] == report
        await bridge.snapshot_to(tmp_path / "snapshot.json", learned_count=1)
    finally:
        await bridge.close()
    restored = OpenHospitalMemoryIntegration(memory_at(tmp_path / "restored", task_learning=True), feedback_timing="batch")
    try:
        await restored.start()
        await restored.restore_from(tmp_path / "snapshot.json")
        assert not restored.core_memory._method.tasks
        await restored.deliver_feedback(batch_end=True)
        assert len(restored.core_memory._method.tasks) == 1
        assert len(restored.core_memory._method.boundaries) == 1
    finally:
        await restored.close()
