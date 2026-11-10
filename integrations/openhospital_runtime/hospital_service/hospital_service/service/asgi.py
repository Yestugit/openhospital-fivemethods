from __future__ import annotations

import atexit
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
import asyncio

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from .auth import ServiceAuth
from .config import load_settings
from .evaluation.evaluator import TRANSIENT_LLM_ERROR_CODE, is_transient_llm_error
from .exam.result_store import PostgresExaminationResultStore
from .public_model_keys import PublicModelKeyPool
from .server import ContestServiceApp, RouteNotFound, _path_only
from .token_usage import PostgresTokenUsageStore
from .llm_client import llm_http_pool


_settings = load_settings()
_usage_store = PostgresTokenUsageStore(_settings.pg_dsn)
_model_key_pool = PublicModelKeyPool(_settings.pg_dsn)
_examination_result_store = PostgresExaminationResultStore(
    _settings.pg_dsn,
    pool_max_size=_settings.db_pool_max_size,
)
_contest_app = ContestServiceApp(
    base_dir=Path(_settings.data_dir),
    test_dataset_key=_settings.test_dataset,
    usage_store=_usage_store,
    auth=ServiceAuth(pg_dsn=_settings.pg_dsn),
    model_settings=_settings.model,
    exam_model_settings=_settings.exam_model,
    model_key_pool=_model_key_pool,
    result_store=_examination_result_store,
)

@asynccontextmanager
async def lifespan(_app: FastAPI):
    async with llm_http_pool():
        try:
            yield
        finally:
            # Uvicorn drains requests before shutdown; persist buffered usage.
            await asyncio.to_thread(_usage_store.flush)


app = FastAPI(title="OpenHospital Contest Service", lifespan=lifespan)
REQUEST_COUNT = Counter("hospital_service_requests_total", "Total HTTP requests.", ["method", "path", "status"])
REQUEST_SECONDS = Histogram("hospital_service_request_seconds", "HTTP request duration.", ["method", "path"])


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/metrics")
async def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/{path:path}")
async def handle_get(path: str, request: Request) -> JSONResponse:
    raw_path = request.url.path + (f"?{request.url.query}" if request.url.query else "")
    headers = request.headers
    with REQUEST_SECONDS.labels("GET", f"/{path}").time():
        response = await _json_response_async(
            lambda: asyncio.to_thread(_contest_app.handle_get_with_query, raw_path, headers)
        )
    REQUEST_COUNT.labels("GET", f"/{path}", str(response.status_code)).inc()
    return response


@app.post("/{path:path}")
async def handle_post(path: str, request: Request) -> JSONResponse:
    payload = await request.json()
    if not isinstance(payload, dict):
        return JSONResponse({"error": "Request body must be a JSON object."}, status_code=400)
    with REQUEST_SECONDS.labels("POST", f"/{path}").time():
        response = await _json_response_async(
            lambda: _contest_app.handle_post_async(_path_only(request.url.path), payload, request.headers)
        )
    REQUEST_COUNT.labels("POST", f"/{path}", str(response.status_code)).inc()
    return response


def _json_response(callback: Any) -> JSONResponse:
    try:
        payload = callback()
    except RouteNotFound as exc:
        return JSONResponse({"error": str(exc) or "not found"}, status_code=404)
    except KeyError as exc:
        return JSONResponse({"error": str(exc).strip("'")}, status_code=404)
    except PermissionError as exc:
        return JSONResponse({"error": str(exc)}, status_code=401)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    except Exception as exc:
        return _server_error_response(exc)
    return JSONResponse(payload)


async def _json_response_async(callback: Any) -> JSONResponse:
    try:
        payload = await callback()
    except RouteNotFound as exc:
        return JSONResponse({"error": str(exc) or "not found"}, status_code=404)
    except KeyError as exc:
        return JSONResponse({"error": str(exc).strip("'")}, status_code=404)
    except PermissionError as exc:
        return JSONResponse({"error": str(exc)}, status_code=401)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    except Exception as exc:
        return _server_error_response(exc)
    return JSONResponse(payload)


def _server_error_response(error: Exception) -> JSONResponse:
    if is_transient_llm_error(error):
        return JSONResponse(
            {
                "error": str(error),
                "code": TRANSIENT_LLM_ERROR_CODE,
                "retryable": True,
            },
            status_code=424,
        )
    return JSONResponse({"error": str(error)}, status_code=500)


def _close_resources() -> None:
    _usage_store.close()
    _model_key_pool.close()
    _examination_result_store.close()


atexit.register(_close_resources)
