"""Background quiescence, isolated probes and hospital native search budgets."""
import sys
import asyncio
from pathlib import Path
import subprocess

import pytest

from mematlas import Memory, MemoryEvent, MemoryUnit, PluginRegistry, MethodServices
from mematlas.core.errors import MemoryStateError, MethodOperationError
from mematlas.integrations.openhospital.config import HospitalOptions
from mematlas.test.native_adapters.fixtures import PausingPlugin
from mematlas.test.native_adapters.probe_fixture import ProbePlugin


def make_memory(path, *, executor="in_process", **settings):
    registry = PluginRegistry()
    registry.register(PausingPlugin(**settings))
    return Memory({"method": {"method_id": "recording"}, "scope": {"user_id": "alice"},
        "runtime": {"artifact_dir": str(path), "executor": executor}}, registry=registry)


@pytest.mark.asyncio
async def test_nested_freeze_restores_reads_and_resumes_pending_writes(tmp_path):
    async with make_memory(tmp_path, mutate=True) as memory:
        await memory.add(MemoryUnit(events=[MemoryEvent("message", "before")]))
        memory._method.calls.clear()
        async with memory.read_only():
            async with memory.read_only():
                assert memory._method.paused
                await memory.search("anything")
                assert "leak" not in memory._method.items
                await memory.save(tmp_path / "snapshot.json")
            assert memory._method.paused
        assert memory._method.calls == ["freeze", "thaw"]
        assert not memory._method.paused
        await memory.add(MemoryUnit(events=[MemoryEvent("message", "after")]))
        assert len(memory._method.tasks) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("executor", ["in_process", "subprocess"])
async def test_checkpoint_and_frozen_lifecycle_have_executor_parity(tmp_path, executor):
    checkpoint = tmp_path / "checkpoint.json"
    async with make_memory(tmp_path / "first", executor=executor) as memory:
        await memory.add(MemoryUnit(events=[MemoryEvent("message", "retained")]))
        await memory.save(checkpoint)
        async with memory.read_only():
            assert len((await memory.search("retained")).items) == 1
    async with make_memory(tmp_path / "restored", executor=executor) as restored:
        await restored.load(checkpoint)
        await restored.add(MemoryUnit(events=[MemoryEvent("message", "continued")]))
        assert len((await restored.search("all")).items) == 2


@pytest.mark.asyncio
async def test_entry_snapshot_failure_releases_pause(tmp_path):
    async with make_memory(tmp_path, failure="save") as memory:
        with pytest.raises(MethodOperationError, match="snapshot failed"):
            async with memory.read_only():
                pytest.fail("must not enter")
        assert memory._method.calls == ["freeze", "thaw"]
        assert not memory.is_read_only and not memory._method.paused


@pytest.mark.asyncio
@pytest.mark.parametrize("failure, error", [("save", MethodOperationError), ("save_state", MemoryStateError)])
async def test_exit_verification_failure_stops_learning_but_releases_pause(tmp_path, failure, error):
    async with make_memory(tmp_path) as memory:
        with pytest.raises(error, match="snapshot failed"):
            async with memory.read_only():
                memory._method.failure = failure
        assert memory._method.calls == ["freeze", "thaw"]
        assert not memory._method.paused and not memory.is_read_only
        with pytest.raises(MemoryStateError):
            await memory.add(MemoryUnit(task="cannot verify previous state"))


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["freeze", "thaw"])
async def test_scheduler_failure_prevents_future_writes(tmp_path, failure):
    async with make_memory(tmp_path, failure=failure) as memory:
        with pytest.raises(MethodOperationError, match=f"{failure} failed"):
            async with memory.read_only():
                pass
        assert not memory.is_read_only
        assert not memory.stats()["backend_status"]["can_continue"]
        with pytest.raises(MemoryStateError):
            await memory.add(MemoryUnit(events=[MemoryEvent("message", "unsafe")]))


def test_preflight_runs_in_target_python_without_backend_or_provider_construction(tmp_path):
    target = tmp_path / "python-env"
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(target)], check=True)
    python = target / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    (target / "dependency-marker").write_text("installed", encoding="utf-8")
    registry = PluginRegistry()
    registry.register(ProbePlugin())
    config = {"method": {"method_id": "probe"}, "scope": {"user_id": "alice"},
              "runtime": {"executor": "subprocess", "python_executable": str(python),
                          "artifact_dir": str(tmp_path / "must-not-create")}}
    services = MethodServices(factory_specs={"llm": {
        "factory": "mematlas.test.native_adapters.probe_fixture:forbidden_provider"},
        "backend": {"factory": "mematlas.test.native_adapters.probe_fixture:InspectableBackend"}})
    report = Memory.check(config, registry=registry, services=services)
    assert report["ready"], report
    assert Path(report["health"]["prefix"]).resolve() == target.resolve()
    assert report["health"]["provider_declared"]
    assert report["health"]["backend_inspectable"]
    assert not (tmp_path / "must-not-create").exists()
    (target / "dependency-marker").unlink()
    assert not Memory.check(config, registry=registry, services=services)["ready"]
    assert Memory.check(config, registry=registry, services=services, inspect_environment=False)["ready"]


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "5"])
def test_hospital_top_k_rejects_invalid_budget(value):
    with pytest.raises(ValueError, match="memory_top_k"):
        HospitalOptions(memory_top_k=value)


def test_hospital_defaults_respect_native_budget_and_legacy_suite_explicit_five(tmp_path):
    from mematlas.integrations.openhospital.suite import make_experiment
    from mematlas.integrations.openhospital import OpenHospitalMemoryIntegration
    assert HospitalOptions().memory_top_k is None
    assert OpenHospitalMemoryIntegration(make_memory(tmp_path)).top_k is None
    assert make_experiment(tmp_path, "repeated", embedding="api").benchmark["memory_top_k"] == 5


@pytest.mark.asyncio
async def test_subprocess_accounts_for_background_calls_between_requests(tmp_path):
    from mematlas.test.native_adapters.fixtures import BackgroundUsagePlugin
    registry = PluginRegistry()
    registry.register(BackgroundUsagePlugin())
    config = {"method": {"method_id": "recording"}, "scope": {"user_id": "alice"},
              "runtime": {"executor": "subprocess", "artifact_dir": str(tmp_path)}}
    async with Memory(config, registry=registry) as memory:
        await memory.add(MemoryUnit(events=[MemoryEvent("message", "background extraction")]))
        # No worker request while the native provider completes.
        await asyncio.sleep(0.1)
        await memory.search("collect telemetry")
        rows = [row for row in memory.services.usage.all() if row.provider == "controlled-background"]
        assert len(rows) == 1 and rows[0].prompt_tokens == 7
        await memory.search("must not count twice")
        assert len(memory.services.usage.all()) == 1


@pytest.mark.asyncio
async def test_large_native_state_crosses_worker_pipe_by_checked_file(tmp_path):
    import json
    from mematlas.core.state_files import KEY
    from mematlas.test.native_adapters.fixtures import LargeStatePlugin
    registry = PluginRegistry()
    registry.register(LargeStatePlugin())
    def config(path):
        return {"method": {"method_id": "recording"}, "scope": {"user_id": "alice"},
                "runtime": {"executor": "subprocess", "artifact_dir": str(path)}}
    checkpoint = tmp_path / "checkpoint.json"
    async with Memory(config(tmp_path / "first"), registry=registry) as memory:
        await memory.add(MemoryUnit(task="large state"))
        await memory.save(checkpoint)
        assert checkpoint.stat().st_size < 100_000
        reference = json.loads(checkpoint.read_text(encoding="utf-8"))["state"][KEY]
        assert Path(reference["path"]).is_file()
        async with memory.read_only():
            await memory.search("stable large vectors")
    async with Memory(config(tmp_path / "second"), registry=registry) as memory:
        await memory.load(checkpoint)
        assert memory.stats()["submitted_units"] == 1
    snapshot = Path(reference["path"])
    snapshot.write_text("{}", encoding="utf-8")
    async with Memory(config(tmp_path / "tampered"), registry=registry) as memory:
        with pytest.raises(MethodOperationError, match="checksum"):
            await memory.load(checkpoint)


def test_fingerprints_follow_nested_snapshot_content_across_storage_locations(tmp_path):
    import hashlib
    import json
    from mematlas.core.state_files import fingerprint_state, pack_state

    def blob(directory, name, raw):
        path = directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "size": len(raw)}

    states = []
    for name in ("original", "new-worker"):
        directory = tmp_path / name
        binary = blob(directory, "vectors", b"native-vector-data")
        body = json.dumps({"graph": binary, "buffer": ["unfinished"]}).encode()
        states.append({"state": blob(directory, "state.json", body)})
    assert states[0] != states[1]
    assert fingerprint_state(states[0]) == fingerprint_state(states[1])
    packed = pack_state(states[1], tmp_path / "transport", limit=1)
    assert fingerprint_state(packed) == fingerprint_state(states[0])
    Path(states[1]["state"]["path"]).write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        fingerprint_state(states[1])


def test_remote_fingerprints_use_remote_checksums_without_opening_server_paths():
    from mematlas.core.state_files import KEY, fingerprint_state
    one = {KEY: {"path": "/remote/one.json", "sha256": "a" * 64, "canonical_bytes": 10}}
    two = {KEY: {**one[KEY], "path": "/remote/two.json"}}
    assert fingerprint_state(one, resolve_files=False) == fingerprint_state(two, resolve_files=False)


@pytest.mark.asyncio
async def test_worker_shutdown_drains_native_exit_output_before_reaping(tmp_path):
    from mematlas.test.native_adapters.fixtures import ShutdownNoisePlugin
    registry = PluginRegistry()
    registry.register(ShutdownNoisePlugin())
    config = {"method": {"method_id": "recording"}, "scope": {"user_id": "alice"},
              "runtime": {"executor": "subprocess", "artifact_dir": str(tmp_path), "operation_timeout_s": 15}}
    async with Memory(config, registry=registry) as memory:
        marker = memory.services.artifact_dir / "shutdown-complete"
    assert marker.read_text(encoding="utf-8") == "graceful"
