"""Train/test runner for OpenHospital doctor agents."""

from __future__ import annotations

import asyncio
import inspect
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Sequence, Union

from .actions import DoctorActions
from .client import ServiceClients, build_service_clients
from .event_logger import EventLogger
from .llm import LLMCaller

AgentMode = Literal["train", "test"]
AgentFactory = Callable[["AgentContext"], Any]


@dataclass
class PatientSelectionConfig:
    """Configuration for train/test patient selection."""

    patient_count: Optional[int] = None
    random_seed: int = 42
    patient_ids: List[str] = field(default_factory=list)
    selection: str = "random"


@dataclass
class AgentRuntimeConfig:
    """Configuration that is common to train and test runs."""

    service_base_url: str = "http://127.0.0.1:8001"
    service_token: str = ""
    model_api_key: str = ""
    team_id: str = "team_demo"
    output_dir: Union[str, Path] = "outputs"
    train: PatientSelectionConfig = field(default_factory=PatientSelectionConfig)
    test: PatientSelectionConfig = field(default_factory=PatientSelectionConfig)
    local_test: bool = False


@dataclass
class AgentContext:
    """Objects passed to user code when constructing an agent."""

    mode: AgentMode
    actions: DoctorActions
    logger: EventLogger
    clients: ServiceClients
    llm_caller: LLMCaller
    config: AgentRuntimeConfig
    phase: str = ""


def run_train(
    *,
    agent_factory: AgentFactory,
    config: AgentRuntimeConfig,
) -> Dict[str, Any]:
    return _run_sync(run_train_async(agent_factory=agent_factory, config=config))


def run_test(
    *,
    agent_factory: AgentFactory,
    config: AgentRuntimeConfig,
) -> Dict[str, Any]:
    return _run_sync(run_test_async(agent_factory=agent_factory, config=config))


async def run_train_async(
    *,
    agent_factory: AgentFactory,
    config: AgentRuntimeConfig,
) -> Dict[str, Any]:
    return await _run_train(agent_factory=agent_factory, config=config)


async def run_test_async(
    *,
    agent_factory: AgentFactory,
    config: AgentRuntimeConfig,
) -> Dict[str, Any]:
    return await _run_test(agent_factory=agent_factory, config=config)


async def _run_train(
    *,
    agent_factory: AgentFactory,
    config: AgentRuntimeConfig,
) -> Dict[str, Any]:
    runtime_config = _ensure_runtime_config(config)
    team_id = runtime_config.team_id
    run_id = _new_run_id("train")
    output_dir = Path(runtime_config.output_dir) / "train" / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    clients = build_service_clients(
        base_url=runtime_config.service_base_url,
        token=runtime_config.service_token,
        model_api_key=runtime_config.model_api_key,
        team_id=runtime_config.team_id,
        mode="train",
    )
    train_config = runtime_config.train
    patient_ids = await clients.dataset_client.list_patient_ids(
        patient_count=train_config.patient_count,
        random_seed=train_config.random_seed,
        patient_ids=train_config.patient_ids,
        selection=train_config.selection,
    )
    if not patient_ids:
        raise ValueError("No patient IDs returned by service /patients.")

    manifest = {
        "train_count": len(patient_ids),
        "train_ids": list(patient_ids),
        "config": {
            "patient_count": train_config.patient_count,
            "random_seed": train_config.random_seed,
            "patient_ids": list(train_config.patient_ids),
            "selection": train_config.selection,
        },
        "created_at": _timestamp(),
    }
    _write_json(output_dir / "train_manifest.json", manifest)

    train_logger = EventLogger(output_dir=output_dir)
    train_cases = await _run_patient_ids_with_context(
        agent_factory=agent_factory,
        runtime_config=runtime_config,
        clients=clients,
        patient_ids=patient_ids,
        mode="train",
        phase="train",
        logger=train_logger,
        team_id=team_id,
    )

    summary = {
        "mode": "train",
        "team_id": runtime_config.team_id,
        "run_id": run_id,
        "patient_count": len(patient_ids),
        "train_count": len(patient_ids),
        "output_dir": str(output_dir),
        "train_manifest_path": str(output_dir / "train_manifest.json"),
        "train_final_results_path": str(train_logger.final_results_path),
        "train_evaluation_results_path": str(train_logger.evaluation_results_path),
        "cases": train_cases,
        "completed_at": _timestamp(),
    }
    _write_json(output_dir / "run_summary.json", summary)
    return summary


async def _run_test(
    *,
    agent_factory: AgentFactory,
    config: AgentRuntimeConfig,
) -> Dict[str, Any]:
    runtime_config = _ensure_runtime_config(config)
    team_id = runtime_config.team_id
    run_id = _new_run_id("test")
    started_at = _timestamp()
    output_dir = Path(runtime_config.output_dir) / "test" / run_id
    logger = EventLogger(output_dir=output_dir)
    clients = build_service_clients(
        base_url=runtime_config.service_base_url,
        token=runtime_config.service_token,
        model_api_key=runtime_config.model_api_key,
        team_id=runtime_config.team_id,
        mode="test",
        send_model_api_key=runtime_config.local_test,
    )

    test_config = runtime_config.test
    if test_config.patient_ids:
        patient_ids = await clients.dataset_client.list_patient_ids(
            patient_ids=test_config.patient_ids,
            selection="forward",
        )
    elif runtime_config.local_test:
        patient_ids = await clients.dataset_client.list_patient_ids(
            patient_count=test_config.patient_count,
            random_seed=test_config.random_seed,
            patient_ids=test_config.patient_ids,
            selection=test_config.selection,
        )
    else:
        patient_ids = await clients.dataset_client.list_patient_ids()
    if not patient_ids:
        raise ValueError("No patient IDs returned by service /patients.")

    case_results = await _run_patient_ids_with_context(
        agent_factory=agent_factory,
        runtime_config=runtime_config,
        clients=clients,
        patient_ids=patient_ids,
        mode="test",
        phase="test",
        logger=logger,
        team_id=team_id,
    )
    final_results = _read_jsonl(logger.final_results_path)
    completed_at = _timestamp()
    elapsed_seconds = _elapsed_seconds(started_at, completed_at)

    summary = {
        "mode": "test",
        "team_id": runtime_config.team_id,
        "run_id": run_id,
        "patient_count": len(patient_ids),
        "local_test": runtime_config.local_test,
        "output_dir": str(output_dir),
        "final_results_path": str(logger.final_results_path),
        "events_path": str(logger.events_path),
        "final_result": final_results,
        "final_results": final_results,
        "final_result_count": len(final_results),
        "cases": case_results,
        "timing": {
            "started_at": started_at,
            "submitted_at": completed_at,
            "elapsed_seconds": elapsed_seconds,
        },
        "completed_at": completed_at,
    }
    _write_json(output_dir / "run_summary.json", summary)
    return summary


async def _run_patient_ids_with_context(
    *,
    agent_factory: AgentFactory,
    runtime_config: AgentRuntimeConfig,
    clients: ServiceClients,
    patient_ids: Sequence[str],
    mode: AgentMode,
    phase: str,
    logger: EventLogger,
    team_id: str,
) -> List[Dict[str, Any]]:
    actions = DoctorActions(
        patient_client=clients.patient_client,
        exam_client=clients.exam_client,
        evaluate_client=clients.evaluate_client,
        event_logger=logger,
        team_id=team_id,
    )
    context = AgentContext(
        mode=mode,
        phase="local_test" if mode == "test" and runtime_config.local_test else phase,
        actions=actions,
        logger=logger,
        clients=clients,
        llm_caller=LLMCaller(
            api_key=runtime_config.model_api_key,
            service_base_url=runtime_config.service_base_url,
            service_token=runtime_config.service_token,
            team_id=team_id,
            mode=mode,
            phase="local_test" if mode == "test" and runtime_config.local_test else phase,
        ),
        config=runtime_config,
    )
    agent = agent_factory(context)
    if agent is None:
        raise ValueError("agent_factory returned None.")

    logger.write_event(
        event_type="RUN_START",
        agent_id=team_id,
        patient_id="SYSTEM",
        payload={
            "mode": mode,
            "phase": phase,
            "patient_count": len(patient_ids),
            "patient_ids_source": "service:/patients",
            "service_base_url": runtime_config.service_base_url,
        },
    )
    return await _run_patient_ids(
        agent=agent,
        patient_ids=patient_ids,
        mode=mode,
        phase=phase,
        logger=logger,
        team_id=team_id,
    )


async def _run_patient_ids(
    *,
    agent: Any,
    patient_ids: Iterable[str],
    mode: AgentMode,
    phase: str,
    logger: EventLogger,
    team_id: str,
) -> List[Dict[str, Any]]:
    method = getattr(agent, mode, None)
    if method is None:
        raise AttributeError("Agent must define an async or sync %s(patient_id) method." % mode)

    case_results = []
    for index, patient_id in enumerate(patient_ids, start=1):
        patient_id = str(patient_id)
        if getattr(agent, "llm_caller", None) is not None:
            agent.llm_caller.usage_case_id = patient_id
        logger.write_event(
            event_type="CASE_START",
            agent_id=team_id,
            patient_id=patient_id,
            payload={"case_index": index, "mode": mode, "phase": phase},
        )
        try:
            result = method(patient_id=patient_id)
            if inspect.isawaitable(result):
                result = await result
            if not isinstance(result, dict):
                result = {"result": result}
            logger.write_event(
                event_type="CASE_END",
                agent_id=team_id,
                patient_id=patient_id,
                payload={
                    "case_index": index,
                    "mode": mode,
                    "phase": phase,
                    "finished": bool(result.get("finished")),
                },
            )
            case_results.append(
                {
                    "patient_id": patient_id,
                    "finished": bool(result.get("finished")),
                }
            )
        except Exception as exc:
            logger.write_event(
                event_type="AGENT_ERROR",
                agent_id=team_id,
                patient_id=patient_id,
                status="failed",
                payload={
                    "case_index": index,
                    "mode": mode,
                    "phase": phase,
                    "error": str(exc),
                },
            )
            raise
    return case_results


def _ensure_runtime_config(config: AgentRuntimeConfig) -> AgentRuntimeConfig:
    if config is None:
        raise ValueError("AgentRuntimeConfig is required; build it from project config and environment variables.")
    if not isinstance(config, AgentRuntimeConfig):
        raise TypeError("config must be an AgentRuntimeConfig instance.")
    return config


def _new_run_id(prefix: str) -> str:
    return "%s_%s" % (prefix, datetime.now(timezone.utc).astimezone().strftime("%Y%m%d_%H%M%S_%f"))


def _timestamp() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def _elapsed_seconds(started_at: str, completed_at: str) -> int:
    try:
        started = datetime.fromisoformat(str(started_at).replace("Z", "+00:00"))
        completed = datetime.fromisoformat(str(completed_at).replace("Z", "+00:00"))
    except ValueError:
        return 0
    return max(0, int((completed - started).total_seconds()))


def _run_sync(coro: Any) -> Dict[str, Any]:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    raise RuntimeError("A running event loop already exists; call the async SDK function instead.")


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
