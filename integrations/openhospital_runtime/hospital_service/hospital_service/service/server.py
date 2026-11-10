"""HTTP entrypoint for the unified train/test contest service."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

SERVICE_DIR = Path(__file__).resolve().parent

try:
    from .auth import ServiceAuth, ServiceAuthContext
    from .config import normalize_dataset_key
    from .doctor_llm import DoctorLLMService
    from .evaluation import EvaluationService
    from .exam import ExamService
    from .exam.catalog import ExaminationIndicatorCatalog
    from .exam.generator import ExaminationModelChannel, ExaminationResultGenerator
    from .exam.result_store import ExaminationResultStore, InMemoryExaminationResultStore
    from .limits import bounded_route
    from .patients import PatientDialogueService
    from .public_model_keys import PublicModelKeyPool
    from .repository import ServiceDataRepository
    from .token_usage import PostgresTokenUsageStore, TokenUsageStore
    from .usage_events import CallUsageStore
except ImportError:
    from auth import ServiceAuth, ServiceAuthContext
    from config import normalize_dataset_key
    from doctor_llm import DoctorLLMService
    from evaluation.service import EvaluationService
    from exam.service import ExamService
    from exam.catalog import ExaminationIndicatorCatalog
    from exam.generator import ExaminationModelChannel, ExaminationResultGenerator
    from exam.result_store import ExaminationResultStore, InMemoryExaminationResultStore
    from limits import bounded_route
    from patients.service import PatientDialogueService
    from public_model_keys import PublicModelKeyPool
    from repository import ServiceDataRepository
    from token_usage import PostgresTokenUsageStore, TokenUsageStore
    from usage_events import CallUsageStore


class ContestServiceApp:
    def __init__(
        self,
        *,
        base_dir: str | Path,
        test_dataset_key: str,
        usage_store: TokenUsageStore,
        auth: ServiceAuth | None = None,
        model_settings: Any | None = None,
        exam_model_settings: Any | None = None,
        model_key_pool: PublicModelKeyPool | None = None,
        result_store: ExaminationResultStore | None = None,
        usage_journal: str | Path | None = None,
    ):
        self.base_dir = Path(base_dir)
        self.test_dataset_key = normalize_dataset_key(str(test_dataset_key))
        self.usage_store = usage_store
        self.auth = auth or ServiceAuth()
        self.model_settings = model_settings
        self.exam_model_settings = exam_model_settings or model_settings
        self.model_key_pool = model_key_pool
        self.result_store = result_store or InMemoryExaminationResultStore()
        self._apps_by_dataset: dict[str, DatasetServices] = {}
        self.call_usage = CallUsageStore(usage_journal)

    def handle_get_with_query(self, raw_path: str, headers: Any | None = None) -> Any:
        parsed = urlparse(raw_path)
        path = _normalize_path(parsed.path)
        query = parse_qs(parsed.query)
        if path == "/health":
            return {"status": "ok"}
        if path not in _GET_ROUTES_REQUIRING_TEAM_ID:
            raise RouteNotFound(path)
        auth_context = self._auth_context_for_credentials(
            bearer_token=_bearer_token_from_headers(headers),
        )
        team_id = _authorized_team_id(auth_context, _query_team_id(query))
        services = self._services_for_auth_context(auth_context)
        if path == "/patients":
            return services.list_patients(
                patient_count=_query_optional_int(query, "patient_count", "patientCount", "count", "limit"),
                random_seed=_query_optional_int(query, "random_seed", "randomSeed", "seed"),
                requested_patient_ids=_query_patient_ids(query),
                selection=_query_selection(query),
            )
        if path == "/usage":
            return {**services.patient_service.usage(team_id=team_id),
                    "instance_id": self.call_usage.instance_id, "usage_schema_version": 2,
                    "usage_records": self.call_usage.records(dataset_key=services.repository.dataset_key, team_id=team_id)}
        raise RouteNotFound(path)

    async def handle_post_async(self, path: str, payload: dict[str, Any], headers: Any | None = None) -> Any:
        if path not in _POST_ROUTES_REQUIRING_TEAM_ID:
            raise RouteNotFound(path)
        with bounded_route(path):
            auth_context, team_id, services = self._post_context(payload, headers)
            component = {"/invoke": "patient", "/llm/chat": "doctor", "/exam/results": "examination",
                         "/evaluate": "judge", "/evaluate/case": "judge"}[path]
            if not hasattr(self, "call_usage"):
                self.call_usage = CallUsageStore()
            dataset_key = getattr(getattr(services, "repository", None), "dataset_key", getattr(auth_context, "dataset_key", "unknown"))
            with self.call_usage.scope(dataset_key=dataset_key, team_id=team_id,
                component=component, patient_id=payload.get("patient_id"),
                accounting_context=payload.get("_accounting_context")):
                return await self._execute_post(path, payload, auth_context, team_id, services)

    async def _execute_post(self, path, payload, auth_context, team_id, services):
        if path == "/invoke":
            _validate_model_key_policy(auth_context, payload)
            return await services.patient_service.invoke_async(
                _authorized_payload(auth_context, payload, team_id=team_id)
            )
        if path == "/llm/chat":
            _validate_model_key_policy(auth_context, payload)
            return await services.doctor_llm_service.chat_async(
                payload,
                team_id=team_id,
                allow_payload_api_key=not (
                    auth_context.bearer_authenticated and auth_context.mode == "test"
                ),
            )
        if path == "/exam/results":
            return await services.exam_service.get_results(payload)
        if path == "/evaluate":
            _validate_evaluation_token_policy(auth_context)
            _validate_model_key_policy(auth_context, payload)
            return await services.evaluation_service.evaluate_async(
                payload,
                include_reference_answers=_should_include_reference_answers(auth_context),
            )
        if path == "/evaluate/case":
            _validate_evaluation_token_policy(auth_context)
            _validate_model_key_policy(auth_context, payload)
            return await services.evaluation_service.evaluate_case_async(
                payload,
                include_reference_answers=_should_include_reference_answers(auth_context),
            )
        raise RouteNotFound(path)

    def _post_context(
        self,
        payload: dict[str, Any],
        headers: Any | None,
    ) -> tuple[ServiceAuthContext, str, "DatasetServices"]:
        auth_context = self._auth_context_for_credentials(
            bearer_token=_bearer_token_from_headers(headers),
        )
        team_id = _authorized_team_id(auth_context, _payload_team_id(payload))
        services = self._services_for_auth_context(auth_context)
        return auth_context, team_id, services

    def _auth_context_for_credentials(
        self,
        *,
        bearer_token: str | None = None,
    ) -> ServiceAuthContext:
        return self.auth.context_for_credentials(bearer_token=bearer_token)

    def _services_for_auth_context(self, auth_context: ServiceAuthContext) -> "DatasetServices":
        return self._services_for_dataset_key(auth_context.dataset_key)

    def _services_for_dataset_key(self, dataset_key: str) -> "DatasetServices":
        dataset_key = normalize_dataset_key(dataset_key)
        if dataset_key == "test":
            dataset_key = self.test_dataset_key
        if dataset_key not in self._apps_by_dataset:
            self._apps_by_dataset[dataset_key] = DatasetServices(
                repository=ServiceDataRepository(base_dir=self.base_dir, dataset_key=dataset_key),
                usage_store=self.usage_store,
                model_settings=self.model_settings,
                exam_model_settings=self.exam_model_settings,
                model_key_pool=self.model_key_pool,
                result_store=self.result_store,
            )
        return self._apps_by_dataset[dataset_key]


class DatasetServices:
    def __init__(
        self,
        repository: ServiceDataRepository,
        usage_store: TokenUsageStore,
        *,
        model_settings: Any | None = None,
        exam_model_settings: Any | None = None,
        model_key_pool: PublicModelKeyPool | None = None,
        result_store: ExaminationResultStore | None = None,
    ):
        self.repository = repository
        patient_kwargs: dict[str, Any] = {}
        if model_settings is not None:
            patient_kwargs = {
                "model_api": getattr(model_settings, "base_url", None),
                "model_name": getattr(model_settings, "model", None),
                "timeout_seconds": getattr(model_settings, "timeout_seconds", None),
                "request_retries": getattr(model_settings, "request_retries", None),
                "default_api_key": getattr(model_settings, "api_key", None),
            }
        self.patient_service = PatientDialogueService(
            repository,
            usage_store=usage_store,
            model_key_pool=model_key_pool,
            **patient_kwargs,
        )
        self.doctor_llm_service = DoctorLLMService(
            usage_store=usage_store,
            dataset_key=repository.dataset_key,
            model_key_pool=model_key_pool,
            **patient_kwargs,
        )
        examination_catalog = ExaminationIndicatorCatalog.load(
            repository.base_dir / "examination_indicators.json"
        )
        examination_settings = exam_model_settings or model_settings
        examination_generator = ExaminationResultGenerator(
            model_channels=_examination_model_channels(examination_settings),
        )
        self.exam_service = ExamService(
            repository,
            catalog=examination_catalog,
            result_store=result_store or InMemoryExaminationResultStore(),
            generator=examination_generator,
        )
        self.evaluation_service = EvaluationService(
            repository,
            usage_store=usage_store,
            model_settings=model_settings,
            model_key_pool=model_key_pool,
        )
    def list_patients(
        self,
        *,
        patient_count: int | None = None,
        random_seed: int | None = None,
        requested_patient_ids: list[str] | None = None,
        selection: str = "random",
    ) -> dict[str, Any]:
        all_patient_ids = self.repository.get_patient_ids()
        summaries = self.repository.get_patient_summaries()
        patients_by_department = None
        patients_without_department = None
        if str(selection or "").strip().lower() == "balanced_random":
            patients_by_department = self.repository.get_patients_by_department()
            patients_without_department = self.repository.get_patients_without_department()
        patient_ids = select_patient_ids(
            all_patient_ids,
            patient_count=patient_count,
            random_seed=random_seed,
            requested_patient_ids=requested_patient_ids,
            selection=selection,
            patients_by_department=patients_by_department,
            patients_without_department=patients_without_department,
        )
        summaries_by_id = {
            str(summary.get("patientId") or ""): summary
            for summary in summaries
        }
        return {
            "datasetKey": self.repository.dataset_key,
            "patient_ids": patient_ids,
            "patients": [
                summaries_by_id[patient_id]
                for patient_id in patient_ids
                if patient_id in summaries_by_id
            ],
            "count": len(patient_ids),
            "totalCount": len(all_patient_ids),
            "source": "service",
        }


def _examination_model_channels(settings: Any | None) -> tuple[ExaminationModelChannel, ...]:
    configured_channels = getattr(settings, "channels", None)
    channel_settings = configured_channels or ((settings,) if settings is not None else ())
    return tuple(
        ExaminationModelChannel(
            model_api=str(getattr(channel, "base_url", "") or "").strip(),
            api_key=str(getattr(channel, "api_key", "") or "").strip(),
            model_name=str(model_name).strip(),
            timeout_seconds=int(getattr(channel, "timeout_seconds", 120) or 120),
            request_retries=max(1, int(getattr(channel, "request_retries", 2) or 2)),
        )
        for channel in channel_settings
        for model_name in (getattr(channel, "models", ()) or ())
        if str(model_name).strip()
    )


class RouteNotFound(Exception):
    pass


def _path_only(raw_path: str) -> str:
    return _normalize_path(raw_path.split("?", 1)[0])


def _normalize_path(raw_path: str) -> str:
    path = raw_path.strip() or "/"
    if not path.startswith("/"):
        return f"/{path}"
    return path


def _query_value(query: dict[str, list[str]], key: str) -> str | None:
    values = query.get(key)
    if not values:
        return None
    return values[0]


def _query_team_id(query: dict[str, list[str]]) -> str | None:
    for key in _TEAM_ID_FIELDS:
        value = _query_value(query, key)
        if value is not None and str(value).strip():
            return value
    return None


def _query_optional_int(query: dict[str, list[str]], *keys: str) -> int | None:
    for key in keys:
        value = _query_value(query, key)
        if value is None or not str(value).strip():
            continue
        try:
            return max(0, int(value))
        except ValueError as exc:
            raise ValueError(f"{key} must be an integer.") from exc
    return None


def _query_patient_ids(query: dict[str, list[str]]) -> list[str]:
    values: list[str] = []
    for key in ("patient_ids", "patientIds", "ids"):
        for raw_value in query.get(key, []):
            values.extend(_parse_patient_id_value(raw_value))
    return _unique_preserve_order(values)


def _query_selection(query: dict[str, list[str]]) -> str:
    value = _query_value(query, "selection") or _query_value(query, "select")
    text = str(value or "random").strip().lower()
    if text not in {"random", "forward", "reverse", "balanced_random"}:
        raise ValueError("selection must be one of: random, balanced_random, forward, reverse.")
    return text


def _parse_patient_id_value(value: Any) -> list[str]:
    text = str(value or "").strip()
    if not text:
        return []
    if text.startswith("["):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError("patient_ids must be a comma-separated string or JSON array.") from exc
        if not isinstance(parsed, list):
            raise ValueError("patient_ids JSON value must be an array.")
        return [str(item).strip() for item in parsed if str(item).strip()]
    return [item.strip() for item in text.split(",") if item.strip()]


def select_patient_ids(
    patient_ids: list[str],
    *,
    patient_count: int | None = None,
    random_seed: int | None = None,
    requested_patient_ids: list[str] | None = None,
    selection: str = "random",
    patients_by_department: dict[str, list[str]] | None = None,
    patients_without_department: list[str] | None = None,
) -> list[str]:
    available = [str(patient_id).strip() for patient_id in patient_ids if str(patient_id).strip()]
    if requested_patient_ids:
        available_set = set(available)
        missing = [patient_id for patient_id in requested_patient_ids if patient_id not in available_set]
        if missing:
            raise ValueError("Requested patient_id not found: %s" % ", ".join(missing))
        return _unique_preserve_order(requested_patient_ids)
    if patient_count is None:
        return available
    if patient_count <= 0:
        return []
    selected = list(available)
    selection = str(selection or "random").strip().lower()
    if selection == "reverse":
        selected.reverse()
    elif selection == "balanced_random":
        selected = _balanced_random_patient_ids(
            selected,
            patient_count=min(patient_count, len(selected)),
            random_seed=random_seed,
            patients_by_department=patients_by_department or {},
            patients_without_department=patients_without_department or [],
        )
    elif selection == "random":
        rng = random.Random(int(random_seed or 0))
        rng.shuffle(selected)
    elif selection != "forward":
        raise ValueError("selection must be one of: random, balanced_random, forward, reverse.")
    return selected[: min(patient_count, len(selected))]


def _balanced_random_patient_ids(
    patient_ids: list[str],
    *,
    patient_count: int,
    random_seed: int | None,
    patients_by_department: dict[str, list[str]],
    patients_without_department: list[str],
) -> list[str]:
    rng = random.Random(int(random_seed or 0))
    available = set(patient_ids)
    buckets = {
        department: [patient_id for patient_id in department_patient_ids if patient_id in available]
        for department, department_patient_ids in patients_by_department.items()
    }
    buckets = {department: patient_ids for department, patient_ids in buckets.items() if patient_ids}
    unbucketed = [patient_id for patient_id in patients_without_department if patient_id in available]
    for bucket in buckets.values():
        rng.shuffle(bucket)
    rng.shuffle(unbucketed)

    selected = []
    selected_set: set[str] = set()
    ordered_departments = sorted(buckets)
    rng.shuffle(ordered_departments)

    while len(selected) < patient_count and ordered_departments:
        next_departments: list[str] = []
        for department in ordered_departments:
            bucket = buckets[department]
            if not bucket:
                continue
            patient_id = bucket.pop()
            selected.append(patient_id)
            selected_set.add(patient_id)
            if len(selected) >= patient_count:
                break
            if bucket:
                next_departments.append(department)
        ordered_departments = next_departments

    remaining = [patient_id for patient_id in unbucketed if patient_id not in selected_set]
    for bucket in buckets.values():
        remaining.extend(patient_id for patient_id in bucket if patient_id not in selected_set)
    rng.shuffle(remaining)
    for patient_id in remaining:
        if len(selected) >= patient_count:
            break
        selected.append(patient_id)
        selected_set.add(patient_id)
    return selected


def _unique_preserve_order(values: list[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        value = str(value).strip()
        if not value or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _payload_team_id(payload: dict[str, Any]) -> str | None:
    for key in _TEAM_ID_FIELDS:
        value = payload.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _bearer_token_from_headers(headers: Any | None) -> str | None:
    if headers is None:
        return None
    value = str(headers.get("X-Hospital-Service-Token") or "").strip()
    if not value:
        return None
    if value.lower().startswith("bearer "):
        return value.split(None, 1)[1].strip()
    return value


def _authorized_payload(
    auth_context: ServiceAuthContext,
    payload: dict[str, Any],
    *,
    team_id: str,
) -> dict[str, Any]:
    if not auth_context.bearer_authenticated:
        return payload
    protected_payload = dict(payload)
    protected_payload["team_id"] = team_id
    return protected_payload


def _authorized_team_id(auth_context: ServiceAuthContext, team_id: Any) -> str:
    requested = str(team_id or "").strip()
    if not requested:
        raise ValueError("team_id is required.")
    if not auth_context.bearer_authenticated:
        return requested
    account = str(auth_context.account or "").strip()
    if not account:
        raise PermissionError("Service token is missing account binding.")
    if requested == account:
        return requested
    raise PermissionError("Service token is not valid for the requested team_id.")


def _validate_model_key_policy(auth_context: ServiceAuthContext, payload: dict[str, Any]) -> None:
    if auth_context.bearer_authenticated and auth_context.mode == "evaluation":
        return
    if auth_context.bearer_authenticated and auth_context.mode == "test":
        if _payload_includes_model_key(payload):
            raise ValueError("Test requests must not include api_key/model_api_key/model_key or camelCase variants.")
        return
    if not auth_context.bearer_authenticated or auth_context.mode == "train":
        if not _payload_has_model_key_value(payload):
            raise ValueError("Train requests must include api_key/model_api_key/model_key or camelCase variants.")


def _validate_evaluation_token_policy(auth_context: ServiceAuthContext) -> None:
    if auth_context.mode == "local" and not auth_context.bearer_authenticated:
        return
    if auth_context.bearer_authenticated and auth_context.mode == "evaluation":
        return
    if (
        auth_context.bearer_authenticated
        and auth_context.mode == "train"
        and auth_context.dataset_key == "train"
    ):
        return
    raise PermissionError("Evaluation requires a train or internal evaluation service token.")


def _should_include_reference_answers(auth_context: ServiceAuthContext) -> bool:
    training_access = (
        auth_context.bearer_authenticated and auth_context.mode == "train"
    ) or (
        not auth_context.bearer_authenticated and auth_context.mode == "local"
    )
    return training_access and auth_context.dataset_key == "train"


def _payload_includes_model_key(payload: dict[str, Any]) -> bool:
    return _contains_model_key_field(payload)


def _payload_has_model_key_value(payload: dict[str, Any]) -> bool:
    for key in _MODEL_KEY_FIELDS:
        value = payload.get(key)
        if value is not None and str(value).strip():
            return True
    return False


def _contains_model_key_field(value: Any) -> bool:
    if isinstance(value, dict):
        for key, item in value.items():
            if key in _MODEL_KEY_FIELDS or _contains_model_key_field(item):
                return True
        return False
    if isinstance(value, list):
        return any(_contains_model_key_field(item) for item in value)
    return False


_MODEL_KEY_FIELDS = ("api_key", "apiKey", "model_api_key", "modelApiKey", "model_key", "modelKey")
_TEAM_ID_FIELDS = ("team_id", "teamId", "account")
_GET_ROUTES_REQUIRING_TEAM_ID = {"/patients", "/usage"}
_POST_ROUTES_REQUIRING_TEAM_ID = {"/invoke", "/llm/chat", "/exam/results", "/evaluate", "/evaluate/case"}


if __name__ == "__main__":
    main()
