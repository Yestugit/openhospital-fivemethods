"""OpenAI-compatible chat client used by the patient service."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import json
import time
from dataclasses import dataclass
from typing import Any
from uuid import uuid4
from urllib.parse import urlparse
import urllib.error
import urllib.request

try:
    from .config import load_settings
    from .usage_events import record_model_call
except ImportError:
    from config import load_settings
    from usage_events import record_model_call

try:
    import httpx
except ModuleNotFoundError:
    httpx = None  # type: ignore[assignment]


def _retry_delay(attempt, error=None):
    headers = getattr(getattr(error, "response", None), "headers", None) or getattr(error, "headers", None)
    try:
        value = float(headers.get("Retry-After"))
        if value >= 0:
            return min(value, 120)
    except (TypeError, ValueError, AttributeError):
        pass
    return min(2 ** attempt, 30)


_llm_capacity: asyncio.Semaphore | None = None
_shared_http_client = None


@asynccontextmanager
async def llm_http_pool(_app=None):
    """Own one connection pool for a service lifespan, without shared credentials."""
    global _shared_http_client
    if httpx is None:
        yield
        return
    if _shared_http_client is not None:
        raise RuntimeError("LLM HTTP pool is already active")
    limits = httpx.Limits(max_connections=_settings().llm_http_max_connections,
                         max_keepalive_connections=_settings().llm_http_max_keepalive)
    async with httpx.AsyncClient(limits=limits) as client:
        _shared_http_client = (asyncio.get_running_loop(), client)
        try:
            yield
        finally:
            _shared_http_client = None


@asynccontextmanager
async def _http_client():
    if _shared_http_client is not None and _shared_http_client[0] is asyncio.get_running_loop():
        yield _shared_http_client[1]
    else:
        # Standalone client users retain scoped ownership outside ASGI.
        limits = httpx.Limits(max_connections=_settings().llm_http_max_connections,
                             max_keepalive_connections=_settings().llm_http_max_keepalive)
        async with httpx.AsyncClient(limits=limits) as client:
            yield client


@dataclass
class ChatResult:
    content: str
    usage: dict[str, Any]
    raw: dict[str, Any]


class OpenAICompatibleChatClient:
    """Small stdlib client for OpenAI-compatible chat/completions APIs."""

    def __init__(
        self,
        *,
        model_api: str,
        api_key: str,
        model: str,
        timeout_seconds: int,
        retries: int = 2,
    ):
        self.url = self._normalize_chat_completions_url(model_api)
        self.api_key = api_key
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.retries = max(1, retries)

    def _normalize_chat_completions_url(self, model_api: str) -> str:
        raw_url = str(model_api).rstrip("/")
        parsed = urlparse(raw_url)
        if not parsed.scheme or not parsed.netloc:
            raise ValueError("model_api must be an absolute URL")
        if raw_url.endswith("/chat/completions"):
            return raw_url
        if raw_url.endswith("/v1"):
            return f"{raw_url}/chat/completions"
        return f"{raw_url}/v1/chat/completions"

    async def chat(
        self,
        *,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        extra_body: dict[str, Any] | None = None,
    ) -> ChatResult:
        if httpx is not None:
            async with _capacity():
                return await self._chat_httpx(
                    messages=messages,
                    temperature=temperature,
                    extra_body=extra_body,
                )
        return await asyncio.to_thread(
            self._chat_sync,
            messages=messages,
            temperature=temperature,
            extra_body=extra_body,
        )

    async def _chat_httpx(
        self,
        *,
        messages: list[dict[str, str]],
        temperature: float,
        extra_body: dict[str, Any] | None,
    ) -> ChatResult:
        payload = self._request_payload(messages, temperature, extra_body)
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        request_id = uuid4().hex
        last_error: Exception | None = None
        timeout = httpx.Timeout(float(self.timeout_seconds))
        async with _http_client() as client:
            for attempt in range(1, self.retries + 1):
                recorded = False
                started = time.perf_counter()
                delay = _retry_delay(attempt)
                try:
                    response = await client.post(self.url, json=payload, headers=headers, timeout=timeout)
                    response.raise_for_status()
                    data = response.json()
                    record_model_call(data.get("usage"), model=self.model, endpoint=self.url, attempt=attempt, request_id=request_id, elapsed_s=time.perf_counter()-started, http_status=200)
                    recorded = True
                    choices = data.get("choices") or []
                    if not choices:
                        raise RuntimeError(f"Chat API returned no choices: {data}")
                    content = str(choices[0].get("message", {}).get("content", "")).strip()
                    usage = self._normalize_usage(data.get("usage"))
                    return ChatResult(content=content, usage=usage, raw=data)
                except Exception as exc:  # noqa: BLE001 - retry wrapper preserves context.
                    last_error = exc
                    delay = _retry_delay(attempt, exc)
                    if not recorded:
                        record_model_call(None, model=self.model, endpoint=self.url, status="failed", attempt=attempt, request_id=request_id, elapsed_s=time.perf_counter()-started, http_status=getattr(getattr(exc, "response", None), "status_code", getattr(exc, "code", None)))
                    status = getattr(getattr(exc, "response", None), "status_code", None)
                    if isinstance(status, int) and status not in {408,429} and status < 500:
                        raise
                    if status is None and not isinstance(exc, (httpx.TransportError, TimeoutError, ConnectionError)):
                        raise
                if attempt < self.retries:
                    await asyncio.sleep(delay)

        raise RuntimeError(
            f"Chat API request failed after {self.retries} attempt(s): {last_error}"
        ) from last_error

    def _chat_sync(
        self,
        *,
        messages: list[dict[str, str]],
        temperature: float,
        extra_body: dict[str, Any] | None,
    ) -> ChatResult:
        payload = self._request_payload(messages, temperature, extra_body)
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        request_id = uuid4().hex
        last_error: Exception | None = None
        for attempt in range(1, self.retries + 1):
            recorded = False
            started = time.perf_counter()
            delay = _retry_delay(attempt)
            try:
                request = urllib.request.Request(self.url, data=body, headers=headers, method="POST")
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    raw = response.read().decode("utf-8")
                data = json.loads(raw)
                record_model_call(data.get("usage"), model=self.model, endpoint=self.url, attempt=attempt, request_id=request_id, elapsed_s=time.perf_counter()-started, http_status=200)
                recorded = True
                choices = data.get("choices") or []
                if not choices:
                    raise RuntimeError(f"Chat API returned no choices: {data}")
                content = str(choices[0].get("message", {}).get("content", "")).strip()
                usage = self._normalize_usage(data.get("usage"))
                return ChatResult(content=content, usage=usage, raw=data)
            except urllib.error.HTTPError as exc:
                delay = _retry_delay(attempt, exc)
                detail = exc.read().decode("utf-8", errors="replace")
                last_error = exc
                record_model_call(None, model=self.model, endpoint=self.url, status="failed", attempt=attempt, request_id=request_id, elapsed_s=time.perf_counter()-started, http_status=getattr(getattr(exc, "response", None), "status_code", getattr(exc, "code", None)))
                if exc.code not in {408,429} and exc.code < 500:
                    raise
            except Exception as exc:  # noqa: BLE001 - retry wrapper preserves context.
                last_error = exc
                if not recorded:
                    record_model_call(None, model=self.model, endpoint=self.url, status="failed", attempt=attempt, request_id=request_id, elapsed_s=time.perf_counter()-started, http_status=getattr(getattr(exc, "response", None), "status_code", getattr(exc, "code", None)))
                if not isinstance(exc, (urllib.error.URLError, TimeoutError, ConnectionError)):
                    raise
            if attempt < self.retries:
                time.sleep(delay)

        raise RuntimeError(
            f"Chat API request failed after {self.retries} attempt(s): {last_error}"
        ) from last_error

    def _request_payload(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        extra_body: dict[str, Any] | None,
    ) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if extra_body:
            payload.update(extra_body)
        model_name = self.model.lower()
        if model_name.startswith(("deepseek-", "glm-", "kimi-")):
            payload.pop("enable_thinking", None)
            payload["thinking"] = {"type": "disabled"}
        else:
            payload["enable_thinking"] = False
        payload["stream"] = False
        return payload

    def _normalize_usage(self, usage: Any) -> dict[str, Any]:
        if not isinstance(usage, dict):
            return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        completion_tokens = int(usage.get("completion_tokens") or 0)
        total_tokens = int(usage.get("total_tokens") or prompt_tokens + completion_tokens)
        result = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }
        if isinstance(usage.get("prompt_tokens_details"), dict):
            result["prompt_tokens_details"] = dict(usage["prompt_tokens_details"])
        return result


def _capacity() -> asyncio.Semaphore:
    global _llm_capacity
    if _llm_capacity is None:
        _llm_capacity = asyncio.Semaphore(_settings().llm_max_concurrency)
    return _llm_capacity


def _settings() -> Any:
    return load_settings()
