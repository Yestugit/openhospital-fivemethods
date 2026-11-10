"""User settings; model services and scope use the public configuration."""
from dataclasses import dataclass, asdict
from mematlas.core.config import MethodConfig
from mematlas.methods.manifest import load_method_descriptor
_DESCRIPTOR = load_method_descriptor(__file__)
_DEFAULT = _DESCRIPTOR.config_schema["properties"]
@dataclass(frozen=True, kw_only=True)
class MemOSConfig:
    """Native settings. storage_path owns this method's state; embedding_dims must match the provider."""
    top_k: int = _DEFAULT["top_k"]["default"]
    embedding_dims: int = _DEFAULT["embedding_dims"]["default"]
    storage_path: str | None = None
    neo4j_uri: str = _DEFAULT["neo4j_uri"]["default"]
    neo4j_user: str = _DEFAULT["neo4j_user"]["default"]
    neo4j_password_env: str = _DEFAULT["neo4j_password_env"]["default"]
    neo4j_database: str = _DEFAULT["neo4j_database"]["default"]
    search_mode: str = _DEFAULT["search_mode"]["default"]

    def _to_method_config(self):
        return MethodConfig("memos", {k:v for k,v in asdict(self).items() if v is not None})
