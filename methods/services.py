"""Provider construction and explicit offline mode for built-in methods."""

from __future__ import annotations

from typing import Any, Mapping

from mematlas.core.errors import ConfigurationError
from mematlas.core.plugin import MethodServices


def prepare_services(
    params: Mapping[str, Any], services: MethodServices, *, method_id: str,
    llm: bool = True, embedding: bool = True,
) -> None:
    """Configure providers once; never silently substitute a research algorithm."""
    mode = params.get("implementation_mode", "standard")
    if mode not in {"standard", "offline"}:
        raise ConfigurationError("implementation_mode must be standard or offline")
    if mode == "offline":
        return
    from mematlas.providers import OpenAICompatibleEmbedder, OpenAICompatibleLLM

    from mematlas.core.provider_config import resolve_providers
    providers = resolve_providers(params, services.provider_configs)
    llm_config, embedding_config = providers["llm"], providers["embedding"]
    if services.llm is None and llm_config.model:
        services.llm = OpenAICompatibleLLM(model=llm_config.model, base_url=llm_config.base_url,
            api_key_env=llm_config.api_key_env, default_options=llm_config.options, usage=services.usage)
    if services.embedder is None and embedding_config.model:
        services.embedder = OpenAICompatibleEmbedder(
            model=embedding_config.model, base_url=embedding_config.base_url,
            api_key_env=embedding_config.api_key_env, dimensions=embedding_config.dimensions,
            usage=services.usage,
        )
    missing = [name for name, required, service in (
        ("llm", llm, services.llm), ("embedding", embedding, services.embedder),
    ) if required and service is None]
    if missing:
        raise ConfigurationError(
            f"{method_id} standard mode requires {', '.join(missing)} services. "
            "Inject providers or configure llm_model/embedding_model; "
            "implementation_mode=offline explicitly enables simplified local smoke tests."
        )
