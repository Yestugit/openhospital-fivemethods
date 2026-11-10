"""Run the bundled OpenHospital service on loopback without contest accounts.

Requires a model key for live patient dialogue and LLM-based scoring. The
hospital's committed service credentials are intentionally ignored.
"""

from __future__ import annotations

import argparse
import logging
from uuid import uuid4
import os
import sys
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from local_world import HospitalWorld, file_sha256

class LocalServiceAuth:
    def __init__(self, dataset: str):
        self.dataset = dataset

    def context_for_credentials(self, *, bearer_token: str | None = None):
        from service.auth import ServiceAuthContext
        return ServiceAuthContext(dataset_key=self.dataset, mode="local")


async def _fallback_exam_results(
    service: Any,
    *,
    dataset: str,
    payload: dict[str, Any],
    error: RuntimeError,
) -> dict[str, Any]:
    """Keep a local benchmark run alive when strict LLM result validation exhausts retries."""
    from service.exam.service import INVALID_EXAM_RESULT
    services = service._services_for_dataset_key(dataset)
    exam_service = services.exam_service
    raw_items = payload.get("items")
    if not isinstance(raw_items, list):
        raise ValueError("items must be a JSON array.")
    requested = exam_service.repository.canonicalize_examination_items(raw_items)
    patient_id = str(payload.get("patient_id") or "").strip()
    if not patient_id:
        raise ValueError("patient_id is required.")

    results: dict[str, dict[str, Any]] = {}
    failed_items: list[str] = []
    for examination in requested:
        try:
            results[examination] = await exam_service._get_one(patient_id, examination)
        except RuntimeError:
            failed_items.append(examination)
            results[examination] = {
                "result": INVALID_EXAM_RESULT,
                "status": "invalid",
                "abnormal_indicators": [],
            }
    logging.getLogger("openhospital.local").warning(
        "Exam generation fallback: patient_id=%s failed_items=%s reason=%s",
        patient_id,
        failed_items,
        error,
    )
    return {
        "results": results,
        "local_generation_fallback": {
            "failed_items": failed_items,
            "policy": "invalid_result",
        },
    }


def build_app(
    dataset: str = "train", *, world_dir: str | Path | None = None,
    hospital_root: str | Path | None = None,
) -> FastAPI:
    if dataset not in {"train", "primary_test", "advanced_test"}:
        raise ValueError("dataset must be train, primary_test, or advanced_test")

    root = Path(hospital_root or ROOT).expanduser().resolve()
    service_root = root / "hospital_service/hospital_service"
    if not (service_root / "service/server.py").is_file():
        raise FileNotFoundError(f"OpenHospital service is missing from {service_root}")
    load_dotenv(root / ".env.local", override=False)
    if str(service_root) not in sys.path:
        sys.path.insert(0, str(service_root))
    from service.config import ModelSettings
    from service.exam.result_store import ExaminationResultRecord
    from service.server import ContestServiceApp, RouteNotFound
    from service.token_usage import InMemoryTokenUsageStore

    config_path = service_root / "service" / "config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    model_config = config.get("model") or {}
    # Never read an API key or PostgreSQL DSN from the committed contest config.
    model = ModelSettings(
        base_url=os.environ.get("HOSPITAL_MODEL_BASE_URL") or model_config.get("base_url")
        or "https://dashscope.aliyuncs.com/compatible-mode/v1",
        models=(os.environ.get("HOSPITAL_PATIENT_MODEL") or "qwen3.5-flash",),
        api_key=os.environ.get("MODEL_API_KEY"),
        timeout_seconds=120,
        request_retries=8,
    )
    exam_model = ModelSettings(
        base_url=model.base_url,
        models=model.models,
        api_key=model.api_key,
        timeout_seconds=model.timeout_seconds,
        request_retries=int(os.environ.get("HOSPITAL_EXAM_REQUEST_RETRIES") or 8),
    )
    contract = {"dataset": dataset, "model": model.models, "base_url": model.base_url,
                "local_adapter": file_sha256(__file__),
                "world_adapter": file_sha256(Path(__file__).with_name("local_world.py")),
                "exam_retries": exam_model.request_retries, "failure_policy": os.environ.get("HOSPITAL_EXAM_FAILURE_POLICY", "raise"),
                "files": {str(p.relative_to(service_root)): file_sha256(p) for p in (service_root / "service").rglob("*")
                          if p.is_file() and (p.suffix == ".py" and "tests" not in p.parts or p.parent.name == dataset and p.suffix in {".json", ".jsonl"})}}
    # JSON canonicalization makes the same configuration comparable after restart.
    import json
    contract = json.loads(json.dumps(contract))
    world = HospitalWorld(Path(world_dir or os.environ.get("HOSPITAL_WORLD_DIR") or ".mematlas/hospital-world") / dataset,
                          contract, ExaminationResultRecord)
    instance_id = uuid4().hex
    service = ContestServiceApp(
        base_dir=service_root / "service" / "data",
        test_dataset_key="primary_test",
        usage_store=InMemoryTokenUsageStore(),
        auth=LocalServiceAuth(dataset),
        model_settings=model,
        exam_model_settings=exam_model,
        result_store=world,
        usage_journal=str(world.directory / "model_usage.jsonl"),
    )
    from service.llm_client import llm_http_pool
    app = FastAPI(title=f"OpenHospital local ({dataset})", lifespan=llm_http_pool)

    def local_only(request: Request) -> None:
        if request.client is None or request.client.host not in {"127.0.0.1", "::1"}:
            raise PermissionError("Local OpenHospital service accepts loopback clients only")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "dataset": dataset, "world_id": world.world_id,
                "instance_id": instance_id, "contract": contract}

    @app.get("/{path:path}")
    def get(path: str, request: Request) -> JSONResponse:
        try:
            local_only(request)
            if path == "usage":
                payload = service.handle_get_with_query(str(request.url))
                rows = payload["usage_records"]
                payload.update(world_id=world.world_id, instance_id=instance_id,
                    usage={"requests": len(rows), "input_tokens": sum(row["prompt_tokens"] for row in rows),
                           "output_tokens": sum(row["completion_tokens"] for row in rows),
                           "total_tokens": sum(row["prompt_tokens"] + row["completion_tokens"] for row in rows),
                           "unknown_calls": sum(not row["usage_known"] for row in rows)})
                return JSONResponse(payload)
            raw_path = request.url.path
            if request.url.query:
                raw_path += "?" + request.url.query
            return JSONResponse(service.handle_get_with_query(raw_path))
        except (PermissionError, ValueError, KeyError, RouteNotFound) as exc:
            return error_response(exc)

    import asyncio
    request_locks = {}

    @app.post("/{path:path}")
    async def post(path: str, request: Request) -> JSONResponse:
        try:
            local_only(request)
            payload = await request.json()
            if not isinstance(payload, dict):
                raise ValueError("Request body must be a JSON object")
            try:
                request_id = payload.get("_request_id")
                if request_id:
                    identity = str(payload.get("team_id", "")) + ":" + request.url.path + ":" + request_id
                    lock = request_locks.setdefault(identity, asyncio.Lock())
                    async with lock:
                        response = world.cached_response(identity, payload)
                        if response is None:
                            response = await service.handle_post_async(request.url.path, payload)
                            world.cache_response(identity, payload, response)
                else:
                    response = await service.handle_post_async(request.url.path, payload)
                world.record_request(str(payload.get("team_id", "")), request.url.path, payload, response)
                return JSONResponse(response)
            except BaseException as exc:
                world.record_request(str(payload.get("team_id", "")), request.url.path, payload,
                                     {"error_type": type(exc).__name__})
                raise
        except RuntimeError as exc:
            current, chain = exc, []
            while current is not None and id(current) not in {id(x) for x in chain}:
                chain.append(current)
                current = current.__cause__ or current.__context__
            statuses = [getattr(x, "status_code", getattr(getattr(x, "response", None), "status_code", getattr(x,"code",None))) for x in chain]
            if any(code in {401,403,404,422} for code in statuses):
                code = next(code for code in statuses if code in {401,403,404,422})
                return JSONResponse({"error": "Hospital provider configuration rejected", "provider_status": code}, status_code=code)
            if any("request failed after" in str(x) for x in chain):
                return JSONResponse({"error": "Hospital provider request attempts exhausted", "provider_attempts_exhausted": True}, status_code=503)
            policy = str(os.environ.get("HOSPITAL_EXAM_FAILURE_POLICY", "raise")).strip().lower()
            if request.url.path == "/exam/results" and policy == "invalid_result":
                with service.call_usage.scope(dataset_key=dataset, team_id=str(payload.get("team_id", "")),
                                               component="examination", patient_id=payload.get("patient_id")):
                    return JSONResponse(await _fallback_exam_results(
                        service, dataset=dataset, payload=payload, error=exc,
                    ))
            raise
        except (PermissionError, ValueError, KeyError, RouteNotFound) as exc:
            return error_response(exc)

    return app


def error_response(exc: Exception) -> JSONResponse:
    status = 401 if isinstance(exc, PermissionError) else 404 if isinstance(exc, (KeyError, RouteNotFound)) else 400
    return JSONResponse({"error": str(exc)}, status_code=status)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hospital-root", default=str(ROOT))
    parser.add_argument("--dataset", choices=("train", "primary_test", "advanced_test"), default="train")
    parser.add_argument("--port", type=int)
    parser.add_argument("--world-dir", help="Persistent shared case-world directory")
    args = parser.parse_args(argv)
    import uvicorn

    default_ports = {"train": 8001, "primary_test": 8002, "advanced_test": 8003}
    port = args.port if args.port is not None else default_ports[args.dataset]
    uvicorn.run(build_app(args.dataset, world_dir=args.world_dir, hospital_root=args.hospital_root),
                host="127.0.0.1", port=port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
