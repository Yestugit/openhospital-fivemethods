"""Doctor-side LLM caller for participant agents."""

from __future__ import annotations

import asyncio
import json
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

DEFAULT_LLM_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_LLM_MODEL = "qwen3.5-flash"
DEFAULT_LLM_TIMEOUT_SECONDS = 120
DEFAULT_LLM_REQUEST_RETRIES = 8
SERVICE_LLM_CHAT_PATH = "/llm/chat"


class LLMRequestPaused(RuntimeError):
    """Transient API outage exhausted retries; retain the recovery checkpoint."""


def _retryable(exc):
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code in {408, 429} or 500 <= exc.code < 600
    return isinstance(exc, (urllib.error.URLError, TimeoutError, socket.timeout,
                            ConnectionError, json.JSONDecodeError))


def _retry_delay(attempt, error=None):
    headers = getattr(error, "headers", None)
    try:
        value = float(headers.get("Retry-After"))
        if value >= 0:
            return min(value, 120)
    except (TypeError, ValueError, AttributeError):
        pass
    return min(2 ** attempt, 30)


@dataclass(frozen=True)
class LLMResponse:
    content: str
    usage: dict[str, Any]
    raw: dict[str, Any]


class LLMCaller:
    """OpenAI-compatible caller for doctor decision logic.

    Train/debug calls use the participant project's `model_api_key` directly.
    Test calls use the contest service `/llm/chat` proxy when service token
    context is available.
    """

    def __init__(
        self,
        api_key: str,
        *,
        service_base_url: str = "",
        service_token: str = "",
        team_id: str = "",
        mode: str = "",
        phase: str = "",
        timeout_seconds: int = DEFAULT_LLM_TIMEOUT_SECONDS,
        request_retries: int = DEFAULT_LLM_REQUEST_RETRIES,
    ):
        self.api_key = str(api_key or "").strip()
        self.service_base_url = str(service_base_url or "").rstrip("/")
        self.service_token = _strip_bearer(service_token)
        self.team_id = str(team_id or "").strip()
        self.mode = str(mode or "").strip()
        self.phase = str(phase or "").strip()
        self.base_url = DEFAULT_LLM_BASE_URL
        self.model = DEFAULT_LLM_MODEL
        self.timeout_seconds = int(timeout_seconds)
        self.request_retries = max(1, int(request_retries))
        self.usage_observer = None
        self.request_context_provider = None

    def _notify_usage(self, usage, *, attempt, status="success", origin="direct", **metadata):
        if self.usage_observer is not None:
            self.usage_observer({"usage": usage, "model": self.model, "origin": origin,
                                 "attempt": attempt, "status": status, "base_url": self.base_url,
                                 "case_id": getattr(self, "usage_case_id", None), **metadata})

    async def call(
        self,
        prompt: str,
        *,
        system_prompt: str = "",
        temperature: float = 0.2,
        extra_body: dict[str, Any] | None = None,
    ) -> str:
        """Return only the assistant text for a simple prompt."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        response = await self.chat(
            messages=messages,
            temperature=temperature,
            extra_body=extra_body,
        )
        return response.content

    def call_sync(
        self,
        prompt: str,
        *,
        system_prompt: str = "",
        temperature: float = 0.2,
        extra_body: dict[str, Any] | None = None,
    ) -> str:
        return asyncio.run(
            self.call(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                extra_body=extra_body,
            )
        )

    async def chat(
        self,
        *,
        messages: list[dict[str, Any]],
        temperature: float = 0.2,
        extra_body: dict[str, Any] | None = None,
    ) -> LLMResponse:
        """Call chat/completions and return content, usage, and raw payload."""
        return await asyncio.to_thread(
            self._chat_sync,
            messages=messages,
            temperature=temperature,
            extra_body=extra_body,
        )

    def chat_sync(
        self,
        *,
        messages: list[dict[str, Any]],
        temperature: float = 0.2,
        extra_body: dict[str, Any] | None = None,
    ) -> LLMResponse:
        return asyncio.run(
            self.chat(
                messages=messages,
                temperature=temperature,
                extra_body=extra_body,
            )
        )

    def _chat_sync(
        self,
        *,
        messages: list[dict[str, Any]],
        temperature: float,
        extra_body: dict[str, Any] | None,
    ) -> LLMResponse:
        if self._should_use_service_proxy():
            return self._service_chat_sync(
                messages=messages,
                temperature=temperature,
                extra_body=extra_body,
            )
        if not self.api_key:
            raise ValueError("model_api_key is required before calling the doctor LLM.")

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "enable_thinking": False,
        }
        if extra_body:
            payload.update(extra_body)

        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        url = f"{self.base_url.rstrip('/')}/chat/completions"

        request_id = uuid4().hex
        def notify(usage, **details):
            self._notify_usage(usage, request_id=request_id, elapsed_s=time.perf_counter()-started, **details)
        last_error: Exception | None = None
        for attempt in range(1, self.request_retries + 1):
            recorded = False
            started = time.perf_counter()
            delay = _retry_delay(attempt)
            try:
                request = urllib.request.Request(url, data=body, headers=headers, method="POST")
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    raw = response.read().decode("utf-8")
                data = json.loads(raw)
                notify(data.get("usage"), http_status=200, attempt=attempt)
                recorded = True
                choices = data.get("choices") or []
                if not choices:
                    raise RuntimeError(f"LLM API returned no choices: {data}")
                content = str(choices[0].get("message", {}).get("content", "")).strip()
                return LLMResponse(
                    content=content,
                    usage=_normalize_usage(data.get("usage")),
                    raw=data,
                )
            except urllib.error.HTTPError as exc:
                delay = _retry_delay(attempt, exc)
                detail = exc.read().decode("utf-8", errors="replace")
                last_error = RuntimeError(f"LLM API HTTP {exc.code}: {detail}")
                notify(None, attempt=attempt, status="failed", http_status=exc.code)
                if not _retryable(exc):
                    raise last_error from exc
            except Exception as exc:  # noqa: BLE001 - retry wrapper preserves context.
                last_error = exc
                if not recorded:
                    notify(None, attempt=attempt, status="failed", http_status=getattr(last_error, "code", None))
                if not _retryable(exc):
                    raise
            if attempt < self.request_retries:
                time.sleep(delay)

        raise LLMRequestPaused(
            f"LLM API request failed after {self.request_retries} attempt(s): {last_error}"
        ) from last_error

    def _should_use_service_proxy(self) -> bool:
        if not self.service_base_url or not self.service_token:
            return False
        return self.phase == "test"

    def _service_chat_sync(
        self,
        *,
        messages: list[dict[str, Any]],
        temperature: float,
        extra_body: dict[str, Any] | None,
    ) -> LLMResponse:
        if not self.service_token:
            raise ValueError("service_token is required before calling the service doctor LLM.")
        if not self.team_id:
            raise ValueError("team_id is required before calling the service doctor LLM.")

        payload: dict[str, Any] = {
            "team_id": self.team_id,
            "messages": messages,
            "temperature": temperature,
        }
        if extra_body:
            payload["extra_body"] = extra_body
        if self.api_key and self.phase != "test":
            payload["api_key"] = self.api_key

        if self.request_context_provider is not None:
            payload["_accounting_context"] = dict(self.request_context_provider())

        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "X-Hospital-Service-Token": self.service_token,
        }
        url = "%s%s" % (self.service_base_url, SERVICE_LLM_CHAT_PATH)

        request_id = uuid4().hex
        def notify(usage, **details):
            self._notify_usage(usage, request_id=request_id, elapsed_s=time.perf_counter()-started, **details)
        last_error: Exception | None = None
        for attempt in range(1, self.request_retries + 1):
            recorded = False
            started = time.perf_counter()
            delay = _retry_delay(attempt)
            try:
                request = urllib.request.Request(url, data=body, headers=headers, method="POST")
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    raw = response.read().decode("utf-8")
                data = json.loads(raw)
                notify(data.get("usage"), http_status=200, attempt=attempt, origin="service_proxy")
                recorded = True
                content = str(data.get("content") or "").strip()
                if not content:
                    raise RuntimeError("Service LLM returned empty content: %r" % (data,))
                return LLMResponse(
                    content=content,
                    usage=_normalize_usage(data.get("usage")),
                    raw=data,
                )
            except urllib.error.HTTPError as exc:
                delay = _retry_delay(attempt, exc)
                detail = exc.read().decode("utf-8", errors="replace")
                last_error = RuntimeError("Service LLM HTTP %s: %s" % (exc.code, detail))
                notify(None, attempt=attempt, status="failed", origin="service_proxy")
                if not _retryable(exc):
                    raise last_error from exc
            except urllib.error.URLError as exc:
                last_error = RuntimeError("Service LLM request failed: %s" % (exc,))
                notify(None, attempt=attempt, status="failed", origin="service_proxy")
            except Exception as exc:  # noqa: BLE001 - retry wrapper preserves context.
                last_error = exc
                if not recorded:
                    notify(None, attempt=attempt, status="failed", origin="service_proxy")
                if not _retryable(exc):
                    raise
            if attempt < self.request_retries:
                time.sleep(delay)

        raise LLMRequestPaused(
            "Service LLM request failed after %d attempt(s): %s"
            % (self.request_retries, last_error)
        ) from last_error


def _normalize_usage(usage: Any) -> dict[str, Any]:
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


def _strip_bearer(value: Any) -> str:
    text = str(value or "").strip()
    if text.lower().startswith("bearer "):
        return text.split(None, 1)[1].strip()
    return text
