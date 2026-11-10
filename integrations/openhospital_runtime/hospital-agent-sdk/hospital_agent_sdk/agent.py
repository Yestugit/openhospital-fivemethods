"""Base class for OpenHospital doctor agents."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml

from .actions import DoctorActions
from .client import ServiceClients, is_local_service_url
from .event_logger import EventLogger
from .llm import LLMCaller
from .runtime import AgentContext, AgentRuntimeConfig, PatientSelectionConfig, run_test, run_train


class BasicAgent(ABC):
    """Base class for participant doctor agents.

    Subclasses normally implement `train(patient_id)` and `test(patient_id)`.
    Runtime setup, local training, and Flask test endpoint wiring are handled
    by the SDK. Project-specific config and memory are built by the project and
    passed into the agent constructor.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        memory: Any = None,
    ):
        self.config = config or {}
        self.memory = memory

        self.context: Optional[AgentContext] = None
        self._actions: Optional[DoctorActions] = None
        self.logger: Optional[EventLogger] = None
        self.clients: Optional[ServiceClients] = None
        self.llm_caller: Optional[LLMCaller] = None
        self._llm: Optional[LLMCaller] = None
        self.runtime_settings: Optional[AgentRuntimeConfig] = None

    def bind_context(self, context: AgentContext) -> "BasicAgent":
        self.context = context
        self._actions = context.actions
        self.logger = context.logger
        self.clients = context.clients
        self.llm_caller = context.llm_caller
        self._llm = context.llm_caller
        self.runtime_settings = context.config
        return self

    def runtime_config(self, payload: Optional[Dict[str, Any]] = None) -> AgentRuntimeConfig:
        return runtime_config_from_project_config(self.config, payload)

    def run_train(self, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return run_train(
            agent_factory=self.bind_context,
            config=self.runtime_config(payload),
        )

    def run_test(self, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return run_test(
            agent_factory=self.bind_context,
            config=self.runtime_config(payload),
        )

    @property
    def actions(self) -> DoctorActions:
        if self._actions is None:
            raise RuntimeError("DoctorActions is not available before the SDK starts a train/test run.")
        return self._actions

    @actions.setter
    def actions(self, value: Optional[DoctorActions]) -> None:
        self._actions = value

    @property
    def llm(self) -> LLMCaller:
        if self._llm is None:
            raise RuntimeError("LLMCaller is not available before the SDK starts a train/test run.")
        return self._llm

    @llm.setter
    def llm(self, value: Optional[LLMCaller]) -> None:
        self._llm = value

    def require_actions(self) -> DoctorActions:
        return self.actions

    def require_llm_caller(self) -> LLMCaller:
        return self.llm

    @abstractmethod
    async def train(self, patient_id: str) -> Dict[str, Any]:
        """Run one training patient encounter."""

    @abstractmethod
    async def test(self, patient_id: str) -> Dict[str, Any]:
        """Run one test patient encounter."""


def load_config(path: Union[str, Path]) -> Dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    return data if isinstance(data, dict) else {}


def runtime_config_from_project_config(
    config: Dict[str, Any],
    request_overrides: Optional[Dict[str, Any]] = None,
) -> AgentRuntimeConfig:
    overrides = request_overrides or {}
    local_test = _config_bool(overrides.get("local_test"), False)
    team_id = _first_config_text(
        overrides,
        "team_id",
        "teamId",
        "account",
        default=_env_text("TEAM_ID", default="team_demo"),
    )
    service_token = _first_config_text(
        overrides,
        "service_token",
        "contestServiceToken",
        default=_env_text("SERVICE_TRAIN_TOKEN"),
    )
    service_base_url = _first_config_text(
        overrides,
        "service_base_url",
        "contestServiceBaseUrl",
        default=_env_text("SERVICE_BASE_URL", default="http://127.0.0.1:8001"),
    )
    if local_test and not service_token and not is_local_service_url(service_base_url):
        raise ValueError("SERVICE_TRAIN_TOKEN is required when /test uses a remote hospital service.")
    model_api_key = _first_config_text(
        overrides,
        "model_api_key",
        "modelApiKey",
        default=_env_text("MODEL_API_KEY"),
    )
    if local_test and not model_api_key:
        raise ValueError("MODEL_API_KEY is required when /test is called without contestServiceToken.")
    train_config = config_value(config, "train", {})
    train_config = train_config if isinstance(train_config, dict) else {}
    test_config = config_value(config, "test", {})
    test_config = test_config if isinstance(test_config, dict) else {}
    if overrides.get("patient_ids") is not None or overrides.get("patientIds") is not None:
        test_config = {
            **test_config,
            "patient_ids": overrides.get("patient_ids", overrides.get("patientIds")),
            "selection": "forward",
        }
    return AgentRuntimeConfig(
        service_base_url=service_base_url,
        service_token=service_token,
        model_api_key=model_api_key,
        team_id=team_id,
        output_dir=str(config_value(config, "output_dir", "outputs")),
        train=_patient_selection_config(train_config),
        test=_patient_selection_config(test_config),
        local_test=local_test,
    )


def config_value(config: Dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    current: Any = config
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current if current is not None else default


def _first_config_text(
    source: Dict[str, Any],
    *keys: str,
    default: str = "",
) -> str:
    for key in keys:
        value = source.get(key)
        if value is not None and str(value).strip():
            return _strip_bearer(value)
    return _strip_bearer(default)


def _env_text(name: str, default: str = "") -> str:
    value = os.environ.get(name)
    if value is not None and str(value).strip():
        return _strip_bearer(value)
    return _strip_bearer(default)


def _strip_bearer(value: Any) -> str:
    text = str(value or "").strip()
    if text.lower().startswith("bearer "):
        return text.split(None, 1)[1].strip()
    return text


def _optional_int(value: Any) -> Optional[int]:
    if value is None or str(value).strip() == "":
        return None
    return int(value)


def _config_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in {"0", "false", "no", "off", ""}


def _patient_selection_config(source: Dict[str, Any]) -> PatientSelectionConfig:
    return PatientSelectionConfig(
        patient_count=_optional_int(source.get("patient_count")),
        random_seed=int(source.get("random_seed", 42)),
        patient_ids=_config_text_list(source.get("patient_ids")),
        selection=_selection_text(source.get("selection", "random")),
    )


def _selection_text(value: Any) -> str:
    text = str(value or "random").strip().lower()
    if text not in {"random", "forward", "reverse"}:
        raise ValueError("selection must be one of: random, forward, reverse.")
    return text


def _config_text_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        raw_items = value.split(",")
    elif isinstance(value, (list, tuple, set)):
        raw_items = value
    else:
        raw_items = [value]
    return [str(item).strip() for item in raw_items if str(item).strip()]
