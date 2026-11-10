"""User settings; model services and scope use the public configuration."""
from dataclasses import dataclass, asdict
from mematlas.core.config import MethodConfig
from mematlas.methods.manifest import load_method_descriptor
_DESCRIPTOR = load_method_descriptor(__file__)
_DEFAULT = _DESCRIPTOR.config_schema["properties"]
@dataclass(frozen=True, kw_only=True)
class MemMachineConfig:
    """Native settings. storage_path owns this method's state; embedding_dims must match the provider."""
    top_k: int | None = None
    embedding_dims: int = _DEFAULT["embedding_dims"]["default"]
    storage_path: str | None = None
    flush_timeout_s: float = _DEFAULT["flush_timeout_s"]["default"]

    def _to_method_config(self):
        return MethodConfig("memmachine", {k:v for k,v in asdict(self).items() if v is not None})
