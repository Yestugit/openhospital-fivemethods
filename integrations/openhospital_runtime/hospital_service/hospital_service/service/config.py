from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

SERVICE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = SERVICE_DIR / "config.yaml"


@dataclass(frozen=True)
class ModelSettings:
    base_url: str
    models: tuple[str, ...]
    api_key: str | None
    timeout_seconds: int
    request_retries: int

    @property
    def model(self) -> str:
        return self.models[0]


@dataclass(frozen=True)
class ExamModelSettings:
    channels: tuple[ModelSettings, ...]


@dataclass(frozen=True)
class ServiceSettings:
    host: str
    port: int
    data_dir: Path
    pg_dsn: str
    db_pool_max_size: int
    test_dataset: str
    model: ModelSettings
    exam_model: ExamModelSettings
    invoke_max_concurrency: int
    chat_max_concurrency: int
    evaluate_max_concurrency: int
    llm_max_concurrency: int
    llm_http_max_connections: int
    llm_http_max_keepalive: int


def load_settings(path: Path = CONFIG_PATH) -> ServiceSettings:
    config = _load_yaml(path)
    server = _mapping(config.get("server"))
    model = _mapping(config.get("model"))
    configured_exam_model = _mapping(config.get("exam_model"))
    database = _mapping(config.get("database"))
    limits = _mapping(config.get("limits"))
    llm = _mapping(config.get("llm"))
    pg_dsn = _env_text("PG_DSN", config.get("pg_dsn"))
    test_dataset = _env_text("TEST_DATASET", config.get("test_dataset"))
    model_settings = _model_settings(model, api_key_env="API_KEY")
    exam_model_settings = _exam_model_settings(configured_exam_model, model_settings)

    return ServiceSettings(
        host=_text(server.get("host"), "127.0.0.1"),
        port=_int(server.get("port"), 8001),
        data_dir=_path(config.get("data_dir"), SERVICE_DIR / "data"),
        pg_dsn=_text(pg_dsn, "postgresql://postgres:postgres@127.0.0.1:5432/contest_web"),
        db_pool_max_size=_env_int("DB_POOL_MAX_SIZE", database.get("pool_max_size"), 25),
        test_dataset=normalize_dataset_key(_text(test_dataset, "primary_test")),
        model=model_settings,
        exam_model=exam_model_settings,
        invoke_max_concurrency=_env_int("INVOKE_MAX_CONCURRENCY", limits.get("invoke_max_concurrency"), 64),
        chat_max_concurrency=_env_int("CHAT_MAX_CONCURRENCY", limits.get("chat_max_concurrency"), 64),
        evaluate_max_concurrency=_env_int("EVALUATE_MAX_CONCURRENCY", limits.get("evaluate_max_concurrency"), 16),
        llm_max_concurrency=_env_int("LLM_MAX_CONCURRENCY", llm.get("max_concurrency"), 64),
        llm_http_max_connections=_env_int("LLM_HTTP_MAX_CONNECTIONS", llm.get("http_max_connections"), 100),
        llm_http_max_keepalive=_env_int("LLM_HTTP_MAX_KEEPALIVE", llm.get("http_max_keepalive"), 20),
    )


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        payload = yaml.safe_load(file) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a YAML object.")
    return payload


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _path(value: Any, default: Path) -> Path:
    raw = _text(value, "")
    path = Path(raw) if raw else default
    if not path.is_absolute():
        path = SERVICE_DIR / path
    return path.resolve()


def _text(value: Any, default: str) -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def _optional_text(value: Any) -> str | None:
    text = _text(value, "")
    return text or None


def _env_text(name: str, fallback: Any) -> str:
    return _text(os.environ.get(name), _text(fallback, ""))


def _env_optional_text(name: str, fallback: Any) -> str | None:
    return _optional_text(os.environ.get(name)) or _optional_text(fallback)


def _model_names(model: dict[str, Any]) -> tuple[str, ...]:
    names = _configured_model_names(model)
    return names or ("qwen3.5-flash",)


def _model_settings(model: dict[str, Any], *, api_key_env: str) -> ModelSettings:
    return ModelSettings(
        base_url=_text(model.get("base_url"), "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        models=_model_names(model),
        api_key=_model_api_key(model, environment_variable=api_key_env),
        timeout_seconds=_int(model.get("timeout_seconds"), 120),
        request_retries=_int(model.get("request_retries"), 2),
    )


def _exam_model_settings(
    exam_model: dict[str, Any],
    fallback: ModelSettings,
) -> ExamModelSettings:
    if not exam_model:
        return ExamModelSettings(channels=(fallback,))

    raw_channels = exam_model.get("channels")
    if raw_channels is None:
        channel_configs = [exam_model]
    else:
        if not isinstance(raw_channels, list) or not raw_channels:
            raise ValueError("exam_model.channels must be a non-empty YAML list")
        if not all(isinstance(channel, dict) for channel in raw_channels):
            raise ValueError("each exam_model.channels entry must be a YAML object")
        channel_configs = raw_channels

    channels: list[ModelSettings] = []
    for index, channel_config in enumerate(channel_configs):
        if not _text(channel_config.get("base_url"), ""):
            raise ValueError(f"exam_model.channels[{index}].base_url is required")
        if not _configured_model_names(channel_config):
            raise ValueError(f"exam_model.channels[{index}].model is required")
        settings = _model_settings(
            channel_config,
            api_key_env=(
                "EXAM_MODEL_API_KEY"
                if index == 0
                else f"EXAM_MODEL_API_KEY_{index + 1}"
            ),
        )
        if not settings.api_key:
            raise ValueError(f"exam_model.channels[{index}].api_key is required")
        channels.append(settings)
    return ExamModelSettings(channels=tuple(channels))


def _model_api_key(model: dict[str, Any], *, environment_variable: str) -> str | None:
    environment_key = _env_optional_text(environment_variable, None)
    if environment_key:
        return environment_key

    configured = _optional_text(model.get("api_key"))
    if configured:
        return configured

    legacy_keys = model.get("api_keys")
    values = legacy_keys if isinstance(legacy_keys, (list, tuple)) else [legacy_keys]
    return next(
        (text for value in values if (text := _text(value, ""))),
        None,
    )


def _configured_model_names(model: dict[str, Any]) -> tuple[str, ...]:
    configured = model.get("model")
    values = configured if isinstance(configured, (list, tuple)) else [configured]
    return tuple(
        dict.fromkeys(
            text
            for value in values
            if (text := _text(value, ""))
        )
    )


def _env_int(name: str, fallback: Any, default: int) -> int:
    return _int(os.environ.get(name), _int(fallback, default))


def _int(value: Any, default: int) -> int:
    if value in (None, ""):
        return default
    return int(value)


def normalize_dataset_key(dataset_key: str) -> str:
    return {
        "preliminary_test": "primary_test",
        "final_test": "advanced_test",
    }.get(str(dataset_key or "").strip(), str(dataset_key or "").strip())
