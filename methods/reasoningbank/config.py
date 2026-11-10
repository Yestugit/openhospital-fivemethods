"""User-facing ReasoningBank configuration."""
from dataclasses import asdict, dataclass
from typing import Literal

from mematlas.core.config import MethodConfig
from mematlas.methods.manifest import load_method_descriptor


_DESCRIPTOR = load_method_descriptor(__file__)
_PARAMETERS = _DESCRIPTOR.config_schema["properties"]


@dataclass(frozen=True, kw_only=True)
class ReasoningBankConfig:
    implementation_mode: Literal["standard", "offline"] = _PARAMETERS["implementation_mode"]["default"]
    domain: str = _PARAMETERS["domain"]["default"]
    include_feedback: bool = _PARAMETERS["include_feedback"]["default"]
    top_k: int = _PARAMETERS["top_k"]["default"]
    extraction_temperature: float = _PARAMETERS["extraction_temperature"]["default"]
    llm_model: str | None = None
    llm_base_url: str | None = None
    llm_api_key_env: str | None = None
    llm_options: dict | None = None
    embedding_model: str | None = None
    embedding_base_url: str | None = None
    embedding_api_key_env: str | None = None
    embedding_dims: int | None = None
    storage_path: str | None = None

    def _to_method_config(self) -> MethodConfig:
        return MethodConfig("reasoningbank", {key: value for key, value in asdict(self).items() if value is not None})
