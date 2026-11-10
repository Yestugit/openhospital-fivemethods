"""Memora's official LoCoMo application profile; no experimental RL training."""
from dataclasses import asdict, dataclass
from mematlas.core.config import MethodConfig
from mematlas.methods.manifest import load_method_descriptor

_DESCRIPTOR = load_method_descriptor(__file__)
_PARAMS = _DESCRIPTOR.config_schema["properties"]


@dataclass(frozen=True, kw_only=True)
class MemoraConfig:
    """Original Microsoft Memora chat memory and prompted retrieval policy.

    top_k: Initial native retrieval count (default30 from official application).
        Policy expansion and linked episodes can return more than this count.
    max_steps: Maximum prompted policy steps after mandatory initial retrieval.
    storage_path: Optional owned Chroma directory; default runtime artifact dir.

    Preserves segmentation, primary/cue anchors, add/update decisions, history,
    BM25 hybrid retrieval, episodic links, EXPAND/RE_QUERY/STOP policy (four
    steps). Public LLM must support original Pydantic structured JSON schemas.
    No reward labels required. Explicit flush processes unfinished sessions;
    save/read_only preserve buffers without model calls or forced learning.
    Public Memory close follows the runtime's automatic flush behavior.
    """
    top_k: int = _PARAMS["top_k"]["default"]
    max_steps: int = _PARAMS["max_steps"]["default"]
    storage_path: str | None = None

    def _to_method_config(self):
        return MethodConfig("memora", {key: value for key, value in asdict(self).items() if value is not None})
