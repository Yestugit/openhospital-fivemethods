"""One provider configuration resolver for checks, compilation and execution."""
from dataclasses import dataclass, field, asdict
from typing import Any, Mapping

from mematlas.core.errors import ConfigurationError


@dataclass(frozen=True)
class ProviderConfig:
    model: str | None = None
    base_url: str | None = None
    api_key_env: str = "OPENAI_API_KEY"
    options: Mapping[str, Any] = field(default_factory=dict)
    dimensions: int | None = None

    def __post_init__(self):
        if self.dimensions is not None and (isinstance(self.dimensions, bool) or not isinstance(self.dimensions, int) or self.dimensions < 1):
            raise ConfigurationError("Provider dimensions must be a positive integer")
        if not isinstance(self.options, Mapping) or not isinstance(self.api_key_env, str) or not self.api_key_env:
            raise ConfigurationError("Provider options and api_key_env are invalid")

    @classmethod
    def from_dict(cls, data):
        unknown = set(data) - {"model", "base_url", "api_key_env", "options", "dimensions"}
        if unknown:
            raise ConfigurationError(f"Unknown provider fields: {sorted(unknown)}")
        return cls(**dict(data))

    def to_dict(self):
        return asdict(self)


ALIASES = {
    "llm": {"model": ("llm_model", "model"),
        "base_url": ("llm_base_url", "openai_base_url", "model_base_url", "base_url"),
        "api_key_env": ("llm_api_key_env", "openai_api_key_env", "model_api_key_env", "api_key_env"),
        "options": ("llm_options",)},
    "embedding": {"model": ("embedding_model",), "base_url": ("embedding_base_url",),
        "api_key_env": ("embedding_api_key_env",), "dimensions": ("embedding_dims", "embedding_dim")},
}


def resolve_providers(params, configured=None):
    configured = dict(configured or {})
    if set(configured) - set(ALIASES):
        raise ConfigurationError("Public providers support llm and embedding; inject special services explicitly")
    resolved = {}
    for role, fields in ALIASES.items():
        raw = configured.get(role, {})
        values = raw.to_dict() if isinstance(raw, ProviderConfig) else dict(raw)
        for field_name, names in fields.items():
            supplied = [(name, params[name]) for name in names if params.get(name) is not None and params.get(name) != ""]
            if any(value != supplied[0][1] for _, value in supplied[1:]):
                raise ConfigurationError(f"Conflicting {role}.{field_name} aliases: {[name for name, _ in supplied]}")
            if supplied:
                value = supplied[0][1]
                if field_name in values and values[field_name] != value:
                    raise ConfigurationError(f"providers.{role}.{field_name} conflicts with method.params.{supplied[0][0]}")
                values[field_name] = value
        if role == "embedding":
            values.setdefault("base_url", resolved["llm"].base_url)
            values.setdefault("api_key_env", resolved["llm"].api_key_env)
        resolved[role] = ProviderConfig.from_dict(values)
    return resolved
