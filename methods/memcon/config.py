"""User configuration for MemCon."""
from dataclasses import dataclass

from mematlas.core.config import MethodConfig
from mematlas.methods.manifest import load_method_descriptor

_DESCRIPTOR = load_method_descriptor(__file__)
_DEFAULTS = _DESCRIPTOR.config_schema["x-variants"]["default"]


@dataclass(frozen=True, kw_only=True)
class MemConConfig:
    """MemCon settings. The native policy learns from episode success feedback."""

    top_k: int = _DEFAULTS["top_k"]
    successful_topk: int = _DEFAULTS["successful_topk"]
    failed_topk: int = _DEFAULTS["failed_topk"]
    insight_topk: int = _DEFAULTS["insight_topk"]
    threshold: float = _DEFAULTS["threshold"]
    storage_path: str | None = None

    def _to_method_config(self) -> MethodConfig:
        values = {"top_k": self.top_k, "successful_topk": self.successful_topk,
                  "failed_topk": self.failed_topk, "insight_topk": self.insight_topk,
                  "threshold": self.threshold}
        if self.storage_path is not None:
            values["storage_path"] = self.storage_path
        return MethodConfig("memcon", values)
