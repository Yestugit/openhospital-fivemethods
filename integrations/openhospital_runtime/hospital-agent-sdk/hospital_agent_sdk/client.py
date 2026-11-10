"""HTTP clients for the OpenHospital contest service."""

from __future__ import annotations

import asyncio
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from .llm import LLMRequestPaused
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

DEFAULT_REQUEST_TIMEOUT_SECONDS = 120
DEFAULT_EVALUATION_TIMEOUT_SECONDS = 300
PATIENTS_PATH = "/patients"
INVOKE_PATH = "/invoke"
EXAM_RESULTS_PATH = "/exam/results"
EVALUATE_PATH = "/evaluate"
EVALUATE_CASE_PATH = "/evaluate/case"


class ServiceClientError(RuntimeError):
    """Raised when the contest service request fails."""


class ServiceHTTPClient:
    def __init__(
        self,
        base_url: str,
        timeout_seconds: int,
        token: str = "",
        request_retries: int = 8,
    ):
        self.base_url = str(base_url).rstrip("/")
        self.token = str(token or "").strip()
        self.timeout_seconds = int(timeout_seconds)
        self.request_context_provider = None
        self.request_retries = max(1, int(request_retries))

    async def get_json(self, path: str, query: Optional[Dict[str, Any]] = None) -> Any:
        query = dict(query or {})
        if not self.token and not is_local_service_url(self.base_url):
            raise ServiceClientError("contest service bearer token is required.")
        return await asyncio.to_thread(self._request_json, "GET", path, None, query)

    async def post_json(self, path: str, payload: Dict[str, Any]) -> Any:
        payload = dict(payload)
        from uuid import uuid4
        payload.setdefault("_request_id", uuid4().hex)
        if self.request_context_provider is not None:
            payload["_accounting_context"] = dict(self.request_context_provider())
        if not self.token and not is_local_service_url(self.base_url):
            raise ServiceClientError("contest service bearer token is required.")
        return await asyncio.to_thread(self._request_json, "POST", path, payload, None)

    def _request_json(
        self,
        method: str,
        path: str,
        payload: Optional[Dict[str, Any]],
        query: Optional[Dict[str, Any]],
    ) -> Any:
        url = "%s%s" % (self.base_url, _normalize_path(path))
        if query:
            encoded_query = urllib.parse.urlencode(
                {key: value for key, value in query.items() if value is not None}
            )
            if encoded_query:
                url = "%s?%s" % (url, encoded_query)

        body = None
        headers = {"Accept": "application/json"}
        if self.token:
            headers["X-Hospital-Service-Token"] = _strip_bearer(self.token)
        if payload is not None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"

        request = urllib.request.Request(url, data=body, headers=headers, method=method)
        raw = ""
        for attempt in range(1, self.request_retries + 1):
            retry_after = None
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    raw = response.read().decode("utf-8")
                break
            except urllib.error.HTTPError as exc:
                retry_after = exc.headers.get("Retry-After") if exc.headers else None
                detail = exc.read().decode("utf-8", errors="replace")
                error = ServiceClientError(
                    "%s %s failed with HTTP %s: %s" % (method, url, exc.code, detail)
                )
                error.status_code = exc.code
                try:
                    remote = json.loads(detail)
                except ValueError:
                    remote = {}
                if remote.get("provider_attempts_exhausted"):
                    raise LLMRequestPaused(str(error)) from exc
                retryable = exc.code in {408, 429} or 500 <= exc.code < 600
                if not retryable:
                    raise error from exc
                if attempt >= self.request_retries:
                    raise LLMRequestPaused(str(error)) from exc
            except urllib.error.URLError as exc:
                error = ServiceClientError("%s %s failed: %s" % (method, url, exc))
                if attempt >= self.request_retries:
                    raise LLMRequestPaused(str(error)) from exc
            except (TimeoutError, ConnectionError) as exc:
                if attempt >= self.request_retries:
                    raise LLMRequestPaused("%s %s timed out or disconnected after %s attempts" %
                                           (method, url, attempt)) from exc
            try:
                delay = max(min(2 ** attempt, 30), min(120, float(retry_after or 0)))
            except ValueError:
                delay = min(2 ** attempt, 30)
            time.sleep(delay)

        if not raw.strip():
            return {}
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ServiceClientError(
                "%s %s returned invalid JSON: %s" % (method, url, raw[:200])
            ) from exc


class DatasetClient:
    def __init__(self, base_url: str, token: str = "", team_id: str = ""):
        self.team_id = str(team_id or "").strip()
        self.http = ServiceHTTPClient(
            base_url=base_url,
            token=token,
            timeout_seconds=DEFAULT_REQUEST_TIMEOUT_SECONDS,
        )

    async def list_patient_ids(
        self,
        *,
        patient_count: Optional[int] = None,
        random_seed: Optional[int] = None,
        patient_ids: Optional[Iterable[str]] = None,
        selection: str = "random",
    ) -> List[str]:
        query = {
            "team_id": self.team_id,
            "patient_count": patient_count,
            "random_seed": random_seed,
            "patient_ids": ",".join(str(patient_id).strip() for patient_id in patient_ids or [] if str(patient_id).strip()),
            "selection": selection,
        }
        payload = await self.http.get_json(PATIENTS_PATH, query)
        return _parse_patient_ids(payload)


class PatientClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        team_id: str,
        token: str = "",
        send_api_key: bool = True,
    ):
        self.team_id = str(team_id).strip()
        self.api_key = str(api_key or "")
        self.send_api_key = bool(send_api_key)
        self.http = ServiceHTTPClient(
            base_url=base_url,
            token=token,
            timeout_seconds=DEFAULT_REQUEST_TIMEOUT_SECONDS,
        )
    async def invoke(self, patient_id: str, input_data: Dict[str, Any]) -> str:
        payload = {
            "team_id": self.team_id,
            "patient_id": patient_id,
            "input": input_data,
        }
        if self.send_api_key:
            if not self.api_key:
                raise ValueError("model_api_key must be configured before training patients.")
            payload["api_key"] = self.api_key
        data = await self.http.post_json(INVOKE_PATH, payload)
        if not isinstance(data, dict):
            raise ServiceClientError("Patient service returned non-object response: %r" % (data,))
        answer = data.get("answer")
        if not isinstance(answer, str):
            raise ServiceClientError("Patient service response missing string answer: %r" % (data,))
        return answer


class ExamClient:
    def __init__(self, base_url: str, token: str = "", team_id: str = ""):
        self.team_id = str(team_id or "").strip()
        self.http = ServiceHTTPClient(
            base_url=base_url,
            token=token,
            timeout_seconds=DEFAULT_REQUEST_TIMEOUT_SECONDS,
        )

    def get_results(self, patient_id: str, items: Iterable[str]) -> Any:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.get_results_async(patient_id=patient_id, items=items))
        return self.get_results_async(patient_id=patient_id, items=items)

    async def get_results_async(self, patient_id: str, items: Iterable[str]) -> Dict[str, Any]:
        requested = [str(item).strip() for item in items if str(item).strip()]
        payload = {
            "team_id": self.team_id,
            "patient_id": patient_id,
            "items": requested,
        }
        data = await self.http.post_json(EXAM_RESULTS_PATH, payload)
        return self._normalize_response(payload=data)

    @staticmethod
    def _normalize_response(payload: Any) -> Dict[str, Any]:
        if not isinstance(payload, dict):
            raise ServiceClientError("Exam service returned non-object response: %r" % (payload,))

        results = payload.get("results", {})
        if not isinstance(results, dict):
            raise ServiceClientError("Exam service response has invalid results field: %r" % (payload,))

        return {"results": results}


class EvaluateClient:
    def __init__(
        self,
        base_url: str,
        token: str = "",
        api_key: str = "",
        send_api_key: bool = True,
        team_id: str = "",
    ):
        self.api_key = str(api_key or "")
        self.send_api_key = bool(send_api_key)
        self.team_id = str(team_id or "").strip()
        self.http = ServiceHTTPClient(
            base_url=base_url,
            token=token,
            timeout_seconds=DEFAULT_EVALUATION_TIMEOUT_SECONDS,
        )

    def evaluate(self, output_log_path: Union[str, Path]) -> Any:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.evaluate_async(output_log_path=output_log_path))
        return self.evaluate_async(output_log_path=output_log_path)

    async def evaluate_async(self, output_log_path: Union[str, Path]) -> Dict[str, Any]:
        final_result_file = Path(output_log_path)
        if final_result_file.is_dir():
            final_result_file = final_result_file / "final_results.jsonl"

        request_payload = {"team_id": self.team_id, "final_result": _read_jsonl(final_result_file)}
        if self.send_api_key:
            if not self.api_key:
                raise ValueError("model_api_key must be configured before training evaluation.")
            request_payload["api_key"] = self.api_key
        response = await self.http.post_json(EVALUATE_PATH, request_payload)
        report = response if isinstance(response, dict) else {"response": response}
        report.setdefault("submitted_at", datetime.now(timezone.utc).astimezone().isoformat())
        _write_json(final_result_file.with_name("final_results_eval_report.json"), report)
        return report

    def evaluate_case(self, final_result: Dict[str, Any]) -> Any:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.evaluate_case_async(final_result=final_result))
        return self.evaluate_case_async(final_result=final_result)

    async def evaluate_case_async(self, final_result: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(final_result, dict):
            raise TypeError("final_result must be a JSON object/dict.")
        request_payload = {"team_id": self.team_id, "final_result": final_result}
        if self.send_api_key:
            if not self.api_key:
                raise ValueError("model_api_key must be configured before training evaluation.")
            request_payload["api_key"] = self.api_key
        response = await self.http.post_json(EVALUATE_CASE_PATH, request_payload)
        return response if isinstance(response, dict) else {"response": response}


@dataclass
class ServiceClients:
    dataset_client: DatasetClient
    patient_client: PatientClient
    exam_client: ExamClient
    evaluate_client: EvaluateClient


def build_service_clients(
    *,
    base_url: str,
    model_api_key: str,
    team_id: str,
    mode: str,
    token: str = "",
    send_model_api_key: Optional[bool] = None,
) -> ServiceClients:
    if send_model_api_key is None:
        send_model_api_key = str(mode) == "train"
    return ServiceClients(
        dataset_client=DatasetClient(base_url=base_url, token=token, team_id=team_id),
        patient_client=PatientClient(
            base_url=base_url,
            api_key=model_api_key,
            team_id=team_id,
            token=token,
            send_api_key=send_model_api_key,
        ),
        exam_client=ExamClient(base_url=base_url, token=token, team_id=team_id),
        evaluate_client=EvaluateClient(
            base_url=base_url,
            token=token,
            api_key=model_api_key,
            send_api_key=send_model_api_key,
            team_id=team_id,
        ),
    )


def _parse_patient_ids(payload: Any) -> List[str]:
    if isinstance(payload, list):
        raw_ids = payload
    elif isinstance(payload, dict):
        raw_ids = payload.get("patientIds") or payload.get("patient_ids") or payload.get("patients") or payload.get("ids")
    else:
        raw_ids = None

    if not isinstance(raw_ids, list):
        raise ServiceClientError("Remote /patients response missing patient ID list: %r" % (payload,))

    patient_ids = []
    for raw_id in raw_ids:
        if isinstance(raw_id, dict):
            raw_id = raw_id.get("patient_id") or raw_id.get("patientId") or raw_id.get("id")
        patient_id = str(raw_id).strip()
        if patient_id:
            patient_ids.append(patient_id)
    if not patient_ids:
        raise ServiceClientError("Remote /patients returned an empty patient ID list.")
    return patient_ids


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


def _normalize_path(path: str) -> str:
    normalized = str(path or "").strip()
    if not normalized:
        return "/"
    if normalized.startswith("/"):
        return normalized
    return "/%s" % normalized


def _strip_bearer(value: Any) -> str:
    text = str(value or "").strip()
    if text.lower().startswith("bearer "):
        return text.split(None, 1)[1].strip()
    return text


def is_local_service_url(base_url: str) -> bool:
    parsed = urllib.parse.urlsplit(str(base_url or ""))
    return parsed.scheme == "http" and parsed.hostname in {"127.0.0.1", "localhost", "::1"}
