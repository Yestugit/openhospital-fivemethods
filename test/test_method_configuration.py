"""Public method settings must survive runtime allocation and transport as plain config."""
from dataclasses import replace
import json

import pytest

from mematlas import MemAtlasConfig, Memory, MethodConfig, RuntimeConfig
from mematlas.core.errors import ConfigurationError
from mematlas.core.registry import default_registry
from mematlas.methods.mem0 import Mem0Config
from mematlas.methods.memrl import MemRLConfig
from mematlas.methods.memoryos import MemoryOSConfig


@pytest.mark.parametrize("settings, method_id, params", [
    (Mem0Config(top_k=2, write_policy="episode"), "mem0", {"top_k": 2, "write_policy": "episode"}),
    (MemRLConfig(batch_size=4, top_k=2), "memrl", {"batch_size": 4, "topk": 2}),
    (MemoryOSConfig(short_term_capacity=2), "memoryos",
     {"short_term_capacity": 2, "retrieval_queue_capacity": 7, "mid_term_heat_threshold": 5.0}),
])
async def test_user_settings_normalize_before_workspace_serialization_and_restore(tmp_path, settings, method_id, params):
    config = MemAtlasConfig(method=settings, runtime=RuntimeConfig(artifact_dir=str(tmp_path)),
                            scope={"user_id": "alice"})
    assert type(config.method) is MethodConfig
    assert config.method == MethodConfig(method_id, params)
    encoded = json.loads(json.dumps(config.to_dict()))
    assert MemAtlasConfig.from_dict(encoded) == config
    assert MemAtlasConfig.from_dict({**encoded, "method": settings}) == config
    # Runtime allocates storage using dataclasses.replace(method, params=...).
    memory = Memory(config)
    try:
        assert type(memory.config.method) is MethodConfig
        assert memory.config.method.params == params
        assert replace(memory.config.method, params={}).method_id == method_id
        saved = json.loads((memory.services.artifact_dir / "memory-run.json").read_text(encoding="utf-8"))
        assert saved["requested_config"]["method"] == encoded["method"]
    finally:
        await memory._close()


@pytest.mark.parametrize("settings", [
    Mem0Config(top_k=0), Mem0Config(write_policy="unknown"), MemRLConfig(batch_size=False), MemRLConfig(top_k=-1),
    MemoryOSConfig(short_term_capacity=0), MemoryOSConfig(mid_term_heat_threshold=-1),
])
def test_invalid_user_settings_fail_before_allocating_storage(tmp_path, settings):
    with pytest.raises(ConfigurationError):
        Memory(MemAtlasConfig(method=settings, runtime=RuntimeConfig(artifact_dir=str(tmp_path))))
    assert not list(tmp_path.iterdir())


def test_method_entry_points_match_declared_capabilities_and_defaults():
    from mematlas.methods import mem0, memrl, memoryos
    for package, config_class in ((mem0, Mem0Config), (memrl, MemRLConfig), (memoryos, MemoryOSConfig)):
        method = MemAtlasConfig(method=config_class()).method
        descriptor = default_registry().get(method.method_id).descriptor
        assert package.SUPPORTED_OPERATIONS == tuple(sorted(op.value for op in descriptor.capabilities.operations))
        assert "evolve" not in package.SUPPORTED_OPERATIONS
        assert "evolve" not in descriptor.input_schemas
        assert not descriptor.commands
        assert not {"get", "update", "delete", "history"}.intersection(package.__all__)
    assert {"SearchOptions", "AddOptions"}.issubset(memrl.__all__)
    assert not {"SearchOptions", "AddOptions"}.intersection(mem0.__all__ + memoryos.__all__)
    assert Mem0Config().top_k == 20
    assert MemRLConfig().batch_size == 32
    assert MemRLConfig().top_k == 3
    assert MemoryOSConfig().short_term_capacity == 10
    assert MemoryOSConfig().retrieval_queue_capacity == 7
    assert MemoryOSConfig().mid_term_heat_threshold == 5
