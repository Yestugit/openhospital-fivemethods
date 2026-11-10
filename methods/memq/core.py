from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib
import json
import re
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

from mematlas.core.errors import ConfigurationError
from mematlas.core.plugin import MethodServices
from mematlas.core.provider_config import ALIASES, resolve_providers


_IMPORT_LOCK = threading.RLock()
_UPSTREAM: dict[str, dict[str, Any]] = {}


@dataclass(frozen=True)
class MemQSettings:
    profile: str
    values: Mapping[str, Any]


def configured_providers(provider_configs: Mapping[str, Any] | None) -> dict[str, dict[str, Any]]:
    result = {}
    for role, raw in (provider_configs or {}).items():
        values = raw.to_dict() if hasattr(raw, "to_dict") else dict(raw)
        result[role] = {name: value for name, value in values.items() if value is not None}
    return result


def resolve_settings(
    params: Mapping[str, Any], variants: Mapping[str, Mapping[str, Any]],
    provider_configs: Mapping[str, Any] | None = None,
) -> MemQSettings:
    profile = str(params.get("profile", "default"))
    if profile not in variants:
        raise ValueError(f"Unknown MemQ profile {profile!r}; choose from {sorted(variants)}")
    values = dict(variants.get("default", {}))
    values.update(dict(variants[profile]))
    values.update(dict(params))
    configured = configured_providers(provider_configs)
    for role, fields in ALIASES.items():
        for field, names in fields.items():
            explicit = [name for name in names if params.get(name) is not None and params.get(name) != ""]
            if explicit or configured.get(role, {}).get(field) is not None:
                for name in names:
                    if name not in explicit:
                        values.pop(name, None)
    if params.get("vector_dim") is not None and not any(
        params.get(name) is not None for name in ALIASES["embedding"]["dimensions"]
    ) and configured.get("embedding", {}).get("dimensions") is None:
        values["embedding_dims"] = int(params["vector_dim"])
    providers = resolve_providers(values, configured)
    embedding_dimensions = providers["embedding"].dimensions
    vector_dimension = params.get("vector_dim")
    if embedding_dimensions is not None:
        if vector_dimension is not None and int(vector_dimension) != embedding_dimensions:
            raise ConfigurationError("MemQ vector_dim must match embedding_dims/provider dimensions")
        values["vector_dim"] = embedding_dimensions
        values["embedding_dims"] = embedding_dimensions
    elif vector_dimension is not None:
        values["embedding_dims"] = int(vector_dimension)
    values["profile"] = profile
    return MemQSettings(profile=profile, values=values)


def bundled_source_root() -> Path:
    return Path(__file__).resolve().parent / "_vendor"


def scope_key(scope: Any) -> str:
    payload = json.dumps(
        scope.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def upstream_user_id(scope: Any, prefix: str) -> str:
    safe_prefix = re.sub(r"[^A-Za-z0-9_.-]+", "_", prefix).strip("._-")[:48]
    return f"{safe_prefix or 'memq'}_{scope_key(scope)[:16]}"


def _run_async(awaitable: Any) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(awaitable)
    with ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(asyncio.run, awaitable).result()


def load_upstream(source_root: str | Path) -> Mapping[str, Any]:
    root = Path(source_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Bundled MemQ source was not found: {root}")
    key = str(root)
    with _IMPORT_LOCK:
        if key in _UPSTREAM:
            return _UPSTREAM[key]
        services_root = root / "memq"
        if not services_root.is_dir():
            raise FileNotFoundError(f"MemQ Python package was not found under {root}")
        root_text = str(root)
        if root_text not in sys.path:
            sys.path.insert(0, root_text)
        package = sys.modules.get("memq")
        package_file = getattr(package, "__file__", None) if package else None
        if package_file and not Path(package_file).resolve().is_relative_to(services_root):
            raise ImportError(
                f"A different 'memq' package is already imported from {package_file}"
            )
        memory_service = importlib.import_module("memq.service.memory_service")
        strategies = importlib.import_module("memq.service.strategies")
        public_config = importlib.import_module("memq.configs.config")
        provider_base = importlib.import_module("memq.providers.base")
        provider_llm = importlib.import_module("memq.providers.llm")
        provider_embedding = importlib.import_module("memq.providers.embedding")
        circuit_breaker = importlib.import_module("memq.providers.circuit_breaker")
        result = {
            "MemoryService": memory_service.MemoryService,
            "StrategyConfiguration": strategies.StrategyConfiguration,
            "RLConfig": public_config.RLConfig,
            "BaseEmbedder": provider_base.BaseEmbedder,
            "OpenAILLM": provider_llm.OpenAILLM,
            "OpenAIEmbedder": provider_embedding.OpenAIEmbedder,
            "LLMCircuitBreaker": circuit_breaker.LLMCircuitBreaker,
        }
        _UPSTREAM[key] = result
        return result


def make_provider_adapters(
    source_root: str | Path,
    services: MethodServices,
    values: Mapping[str, Any],
) -> tuple[Any, Any]:
    upstream = load_upstream(source_root)
    if services.llm is None:
        raise ValueError("MemQ requires a framework LLM service")
    if services.embedder is None:
        raise ValueError("MemQ requires a framework embedding service")

    source_llm = upstream["OpenAILLM"]
    source_embedder = upstream["OpenAIEmbedder"]
    providers = resolve_providers(values, configured_providers(services.provider_configs))
    log_directory = services.register_resource(services.artifact_dir / "memq" / "provider_logs")
    log_directory.mkdir(parents=True, exist_ok=True)

    class CompletionTransport:
        def create(self, *, messages, model=None, **kwargs):
            # MethodServices exposes final text, with usage recorded by the
            # framework. It has no SDK response/finish-reason/stream protocol.
            content = _run_async(services.llm.complete(messages, **kwargs))
            return SimpleNamespace(model=model, usage=None, choices=[
                SimpleNamespace(message=SimpleNamespace(content=str(content)), finish_reason=None),
            ])

    class EmbeddingTransport:
        def create(self, *, input, model=None):
            vectors = _run_async(services.embedder.embed(list(input)))
            if len(vectors) != len(input):
                raise ValueError("MemQ embedding provider must return one vector per text")
            return SimpleNamespace(model=model, usage=None,
                                   data=[SimpleNamespace(embedding=vector) for vector in vectors])

    class FrameworkLLM(source_llm):
        def __init__(self) -> None:
            self.services = services
            self.model = str(providers["llm"].model or "")
            self.base_url = providers["llm"].base_url
            self.default_temperature = float(values.get("llm_temperature", 0.0))
            self.default_max_tokens = values.get("llm_max_tokens", 4096)
            self.stream = False
            self.circuit_breaker = upstream["LLMCircuitBreaker"](
                **dict(values.get("llm_circuit_breaker", {})))
            self._token_log_lock = threading.Lock()
            self._token_log_path = log_directory / "llm.jsonl"
            self.total_prompt_tokens = 0
            self.total_completion_tokens = 0
            self.client = SimpleNamespace(chat=SimpleNamespace(completions=CompletionTransport()))

    class FrameworkEmbedder(source_embedder):
        def __init__(self) -> None:
            upstream["BaseEmbedder"].__init__(self,
                max_text_len=int(values.get("embedding_max_text_len", 8196)),
                max_token_len=int(values.get("embedding_max_token_len", 0)),
            )
            self.services = services
            self.model = str(providers["embedding"].model or "")
            self.base_url = providers["embedding"].base_url
            self._token_log_lock = threading.Lock()
            self._token_log_path = log_directory / "embedding.jsonl"
            self.total_tokens = 0
            self.client = SimpleNamespace(embeddings=EmbeddingTransport())

    return FrameworkLLM(), FrameworkEmbedder()


def create_upstream_service(
    settings: MemQSettings,
    services: MethodServices,
    scope: Any,
    *,
    source_root: str | Path | None = None,
) -> Any:
    root = source_root or bundled_source_root()
    upstream = load_upstream(root)
    values = settings.values
    llm, embedder = make_provider_adapters(root, services, values)
    strategies = upstream["StrategyConfiguration"].from_strings(
        str(values["build_strategy"]),
        str(values["retrieve_strategy"]),
        str(values["update_strategy"]),
    )
    rl_config = upstream["RLConfig"](
        epsilon=float(values["epsilon"]),
        tau=float(values["tau"]),
        alpha=float(values["alpha"]),
        gamma=float(values["gamma"]),
        lambd=float(values["lambd"]),
        q_init_pos=float(values["q_init_pos"]),
        q_init_neg=float(values["q_init_neg"]),
        q_floor=values.get("q_floor"),
        q_ceiling=values.get("q_ceiling"),
        success_reward=float(values["success_reward"]),
        failure_reward=float(values["failure_reward"]),
        sim_threshold=float(values["sim_threshold"]),
        topk=int(values["topk"]),
        novelty_threshold=float(values["novelty_threshold"]),
        weight_sim=float(values["weight_sim"]),
        weight_q=float(values["weight_q"]),
        structural_max_depth=int(values["structural_max_depth"]),
        normalize_dq=bool(values["normalize_dq"]),
        q_init_adaptive=bool(values["q_init_adaptive"]),
        alpha_min=float(values.get("alpha_min", 0.0)),
        alpha_schedule=str(values.get("alpha_schedule", "constant")),
        q_min_threshold=values.get("q_min_threshold", -0.8),
    )
    service = upstream["MemoryService"](
        llm_provider=llm,
        embedding_provider=embedder,
        strategy_config=strategies,
        user_id=upstream_user_id(scope, str(values.get("user_id", "memq_user"))),
        num_workers=int(values.get("num_workers", 32)),
        max_keywords=int(values["max_keywords"]),
        add_similarity_threshold=float(values["add_similarity_threshold"]),
        enable_value_driven=bool(values.get("enable_value_driven", True)),
        rl_config=rl_config,
        normalize_q=bool(values.get("normalize_q", True)),
        vector_dim=int(values.get("vector_dim", 3072)),
        dedup_by_task_id=bool(values.get("dedup_by_task_id", False)),
    )
    service.adjustment_mode = str(values.get("adjustment_mode", "append"))
    return service
