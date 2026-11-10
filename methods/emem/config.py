"""EMem configuration; LLM/Embedding services use public provider settings."""
from dataclasses import asdict, dataclass
from typing import Literal
from mematlas.core.config import MethodConfig
from mematlas.methods.manifest import load_method_descriptor

_DESCRIPTOR = load_method_descriptor(__file__)
_PARAMS = _DESCRIPTOR.config_schema["properties"]


@dataclass(frozen=True, kw_only=True)
class EMemConfig:
    """Official source adapter, using contextual conversation extraction.

    top_k: Default number of retrieved EDUs (overridable by search).
    variant: emem skips retrieval PPR; emem_g uses the original graph PPR.
        Both preserve upstream graph construction, extraction and LLM reranking.
    support_json_schema: Whether the configured LLM supports structured schemas.
    json_output_mode: Opt-in json_unicode_v1 uses JSON object transport, native
        schema validation and three bounded format repairs. Extraction/ranking
        remain native. Profile identity is checked when restoring snapshots.
    storage_path: Optional owned state directory; otherwise runtime artifact dir.

    Profile: RAW-RAG's EMem runner, linking_top_k=30, normalized embeddings,
    synonymy threshold=.9, contextual EDU extraction without EDU context gen.
    No feedback labels required. Buffers commit at episode_end, a complete
    experience, or explicit flush; save/read_only do not commit buffers.
    Public Memory close follows the runtime's automatic flush behavior.
    """
    top_k: int = _PARAMS["top_k"]["default"]
    variant: Literal["emem", "emem_g"] = _PARAMS["variant"]["default"]
    support_json_schema: bool = _PARAMS["support_json_schema"]["default"]
    json_output_mode: Literal["native", "json_unicode_v1"] = _PARAMS["json_output_mode"]["default"]
    storage_path: str | None = None

    def _to_method_config(self):
        return MethodConfig("emem", {key: value for key, value in asdict(self).items() if value is not None})
