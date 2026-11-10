from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from mematlas.core.plugin import MethodServices
from mematlas.core.provider_config import ALIASES, resolve_providers

from .prompts import keyword_prompt, script_prompt


@dataclass(frozen=True)
class MemRLSettings:
    profile: str
    values: Mapping[str, Any]


def configured_providers(provider_configs: Mapping[str, Any] | None) -> dict[str, dict[str, Any]]:
    # Runtime resolves absent provider fields into ProviderConfig(model=None,
    # ...). Treat these nulls as unspecified when applying profile defaults.
    result = {}
    for role, raw in (provider_configs or {}).items():
        values = raw.to_dict() if hasattr(raw, "to_dict") else dict(raw)
        result[role] = {key: value for key, value in values.items() if value is not None}
    return result


def resolve_settings(
    params: Mapping[str, Any], variants: Mapping[str, Mapping[str, Any]],
    provider_configs: Mapping[str, Any] | None = None,
) -> MemRLSettings:
    provider_configs = configured_providers(provider_configs)
    profile = str(params.get("profile", "default"))
    if profile not in variants:
        raise ValueError(f"Unknown MemRL profile {profile!r}; choose from {sorted(variants)}")
    values = dict(variants.get("default", {}))
    values.update(dict(variants[profile]))
    values.update(dict(params))
    for role, fields in ALIASES.items():
        raw = (provider_configs or {}).get(role, {})
        configured = raw.to_dict() if hasattr(raw, "to_dict") else dict(raw)
        for field, names in fields.items():
            explicit = [name for name in names if params.get(name) is not None and params.get(name) != ""]
            if explicit or configured.get(field) is not None:
                for name in names:
                    if name not in explicit:
                        values.pop(name, None)
    # Validate explicit conflicts while allowing configured providers to replace
    # profile transport defaults. The native collection must use the same dims.
    providers = resolve_providers(values, provider_configs)
    if providers["embedding"].dimensions is not None:
        values["embedding_dims"] = providers["embedding"].dimensions
    values["profile"] = profile
    return MemRLSettings(profile=profile, values=values)


def bundled_source_root() -> Path:
    return Path(__file__).resolve().parent / "_vendor"


def upstream_user_id(scope: Any, user_id_prefix: str) -> str:
    payload = json.dumps(dict(scope), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    prefix = re.sub(r"[^A-Za-z0-9_.-]+", "_", user_id_prefix).strip("._-")[:48]
    return f"{prefix or 'memrl'}_{digest}"


def _run_async(awaitable: Any) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(awaitable)
    with ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(copy_context().run, asyncio.run, awaitable).result()


def _mos_config(
    settings: MemRLSettings,
    services: MethodServices,
    user_id: str,
) -> tuple[dict[str, Any], str | None]:
    values = settings.values
    configured_path = values.get("mos_config_path")
    if configured_path:
        path = Path(str(configured_path)).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"MemRL mos_config_path does not exist: {path}")
        return {}, str(path)

    providers = resolve_providers(values, configured_providers(services.provider_configs))
    llm, embedding = providers["llm"], providers["embedding"]
    llm_key = os.environ.get(llm.api_key_env)
    embedding_key = os.environ.get(embedding.api_key_env)
    missing = [name for name, key in ((llm.api_key_env, llm_key), (embedding.api_key_env, embedding_key)) if not key]
    if missing:
        raise ValueError(
            "MemRL's MemOS text-memory backend requires API credentials in its config; "
            f"set {', '.join(sorted(set(missing)))} or provide mos_config_path"
        )

    users_dir = services.artifact_dir / "memrl" / "users"
    users_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "chat_model": {
            "backend": "openai",
            "config": {
                "model_name_or_path": llm.model,
                "api_key": llm_key,
                "api_base": llm.base_url,
            },
        },
        "mem_reader": {
            "backend": "simple_struct",
            "config": {
                "llm": {
                    "backend": "openai",
                    "config": {
                        "model_name_or_path": llm.model,
                        "api_key": llm_key,
                        "api_base": llm.base_url,
                    },
                },
                "embedder": {
                    "backend": "universal_api",
                    "config": {
                        "provider": "openai",
                        "model_name_or_path": embedding.model,
                        "api_key": embedding_key,
                        "base_url": embedding.base_url,
                    },
                },
                "chunker": {"backend": "sentence", "config": {"chunk_size": 500}},
            },
        },
        "user_manager": {
            "backend": "sqlite",
            "config": {"db_path": str(services.register_resource(users_dir / f"{user_id}.db"))},
        },
        "top_k": int(values.get("k_retrieve", 1)),
    }
    return config, None


def _provider_adapters(source_root: Path, services: MethodServices, values: Mapping[str, Any]):
    services.add_import_root(source_root)
    from memrl.providers.base import BaseEmbedder, BaseLLM
    from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

    class FrameworkLLMAdapter(BaseLLM):
        def __init__(self) -> None:
            super().__init__()
            self.services = services
            self.default_temperature = float(values.get("llm_temperature", 0.7))
            self.default_max_tokens = values.get("llm_max_tokens")

        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=4, max=10),
            retry=retry_if_exception_type(Exception),
        )
        def generate(self, messages, **kwargs):
            options = {"temperature": self.default_temperature}
            if self.default_max_tokens is not None:
                options["max_tokens"] = self.default_max_tokens
            if "max_tokens" not in kwargs and "max_completion_tokens" in kwargs:
                options["max_tokens"] = kwargs["max_completion_tokens"]
            options.update(kwargs)
            return _run_async(self.services.llm.complete(messages, **options))

        @retry(
            stop=stop_after_attempt(2),
            wait=wait_exponential(multiplier=1, min=2, max=5),
            retry=retry_if_exception_type(Exception),
        )
        def extract_keywords(self, text: str, max_keywords: int = 8):
            response = self.generate(
                [{"role": "user", "content": keyword_prompt(text, max_keywords)}],
                temperature=0,
                max_tokens=100,
            )
            keywords = []
            for keyword in response.strip().split(","):
                keyword = keyword.strip().lower()
                keyword = re.sub(r"^[\"']|[\"']$", "", keyword)
                keyword = re.sub(r"\s+", " ", keyword)
                if keyword and len(keyword) > 1:
                    keywords.append(keyword)
            return keywords[:max_keywords]

        def generate_script(self, trajectory: str) -> str:
            kwargs = {"temperature": self.default_temperature}
            if self.default_max_tokens is not None:
                kwargs["max_tokens"] = self.default_max_tokens
            return self.generate(
                [{"role": "user", "content": script_prompt(trajectory)}],
                **kwargs,
            )

    class FrameworkEmbedderAdapter(BaseEmbedder):
        def __init__(self) -> None:
            super().__init__(max_text_len=int(values.get("embedding_max_text_len", 8196)))
            self.services = services

        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=4, max=10),
            retry=retry_if_exception_type(Exception),
        )
        def embed(self, texts):
            if not texts:
                return []
            chunked, counts = self._chunk_texts(texts)
            vectors = _run_async(self.services.embedder.embed(chunked))
            return self._merge_chunk_embeddings(vectors, counts)

    return FrameworkLLMAdapter(), FrameworkEmbedderAdapter()


def create_upstream_service(
    settings: MemRLSettings,
    services: MethodServices,
    *,
    user_id: str,
    source_root: Path | None = None,
) -> Any:
    source_root = source_root or bundled_source_root()
    if not source_root.is_dir():
        raise FileNotFoundError(f"Bundled MemRL source was not found: {source_root}")
    services.add_import_root(source_root)

    from memrl.configs.config import RLConfig
    from memrl.service.memory_service import MemoryService
    from memrl.service.strategies import StrategyConfiguration

    values = settings.values
    config, config_path = _mos_config(settings, services, user_id)
    temporary_path = None
    if config_path is None:
        with tempfile.NamedTemporaryFile("w", suffix=".json", encoding="utf-8", delete=False) as handle:
            json.dump(config, handle)
            temporary_path = Path(handle.name)
        config_path = str(temporary_path)

    rl_fields = (
        "epsilon", "tau", "alpha", "gamma", "q_init_pos", "q_init_neg", "q_floor",
        "success_reward", "failure_reward", "sim_threshold", "sim_threshold_by_task",
        "topk", "novelty_threshold", "recency_boost", "reward_merge_gain",
        "q_min_threshold", "weight_sim", "weight_q",
    )
    try:
        llm, embedder = _provider_adapters(source_root, services, values)
        kwargs = {
            "mos_config_path": config_path,
            "llm_provider": llm,
            "embedding_provider": embedder,
            "strategy_config": StrategyConfiguration.from_strings(
                str(values.get("build_strategy", "proceduralization")),
                str(values.get("retrieve_strategy", "query")),
                str(values.get("update_strategy", "adjustment")),
            ),
            "user_id": user_id,
            "vector_dimension": int(values.get("embedding_dims") or 3072),
            "num_workers": int(values.get("num_workers", 32)),
            "max_keywords": int(values.get("max_keywords", 8)),
            "memory_confidence": float(values.get("memory_confidence", 100.0)),
            "add_similarity_threshold": float(values.get("add_similarity_threshold", 0.9)),
            "enable_value_driven": bool(values.get("enable_value_driven", True)),
            "rl_config": RLConfig(**{name: values[name] for name in rl_fields if name in values}),
            "sim_norm_mean": float(values.get("sim_norm_mean", 0.0)),
            "sim_norm_std": float(values.get("sim_norm_std", 0.0)),
            "use_z_score_normalization": bool(values.get("use_z_score_normalization", True)),
            "dedup_by_task_id": bool(values.get("dedup_by_task_id", False)),
            "adjustment_mode": str(values.get("adjustment_mode", "append")),
        }
        return MemoryService(**kwargs)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
