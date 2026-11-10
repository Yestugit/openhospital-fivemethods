from dataclasses import dataclass
from mematlas.core.config import MethodConfig
from mematlas.methods.manifest import load_method_descriptor

_DESCRIPTOR = load_method_descriptor(__file__)
_DEFAULTS = _DESCRIPTOR.config_schema["x-variants"]["default"]


@dataclass(frozen=True, kw_only=True)
class AELConfig:
    profile: str = "default"
    runner_mode: str = _DEFAULTS["runner_mode"]
    retrieval_top_k: int = _DEFAULTS["retrieval_top_k"]
    bars_per_day: int = _DEFAULTS["bars_per_day"]
    planner_names: tuple[str, ...] = tuple(_DEFAULTS["planner_names"])
    tool_config_names: tuple[str, ...] = tuple(_DEFAULTS["tool_config_names"])
    memory_policy_names: tuple[str, ...] = tuple(_DEFAULTS["memory_policy_names"])
    credit_method: str = _DEFAULTS["credit_method"]
    evolve_planner: bool = _DEFAULTS["evolve_planner"]
    evolve_tools: bool = _DEFAULTS["evolve_tools"]
    per_tool_selection: bool = _DEFAULTS["per_tool_selection"]
    storage_path: str | None = None

    def _to_method_config(self) -> MethodConfig:
        values = {"profile": self.profile, "runner_mode": self.runner_mode, "retrieval_top_k": self.retrieval_top_k, "bars_per_day": self.bars_per_day,
                  "planner_names": list(self.planner_names), "tool_config_names": list(self.tool_config_names),
                  "memory_policy_names": list(self.memory_policy_names), "credit_method": self.credit_method,
                  "evolve_planner": self.evolve_planner, "evolve_tools": self.evolve_tools,
                  "per_tool_selection": self.per_tool_selection}
        if self.storage_path is not None: values["storage_path"] = self.storage_path
        return MethodConfig("ael", values)
