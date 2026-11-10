"""HTTP service executor for remote Memory implementations."""

from __future__ import annotations

import asyncio
import json
from typing import Any, Mapping
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from mematlas.core.config import MemAtlasConfig
from mematlas.core.errors import ExecutorError
from mematlas.core.models import MemoryContribution, MemoryEpisode, MemoryEvent, MemoryUnit
from mematlas.core.plugin import BaseMemoryMethod, MethodContext, MethodServices
from mematlas.core.telemetry import UsageRecord, RemoteUsageImporter, current_usage_context


class ServiceMemoryMethod(BaseMemoryMethod):
    """JSON-over-HTTP proxy for an external Memory method."""

    def __init__(self, config: MemAtlasConfig, services: MethodServices, plugin_spec=None):
        self.config = config
        self.services = services
        self.base_url = str(config.runtime.service_url or "").rstrip("/")
        self.headers = {"Content-Type": "application/json", **dict(config.runtime.service_headers)}
        self.timeout_s = float(config.runtime.service_timeout_s)
        self._started = False
        self.storage_bytes: int | None = None
        self.usage_error: str | None = None
        self.plugin_spec = plugin_spec
        self.run_id: str | None = None
        self._broken = False
        self._usage_importer = RemoteUsageImporter(services.usage)

    def _url(self, path: str) -> str:
        if not self.base_url:
            raise ExecutorError("service executor requires runtime.service_url")
        return f"{self.base_url}/{path.lstrip('/')}"

    def _request_sync(self, method: str, path: str, payload: Mapping[str, Any] | None = None) -> Any:
        body = None
        headers = dict(self.headers)
        if payload is not None:
            body = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
        request = Request(self._url(path), data=body, headers=headers, method=method)
        try:
            with urlopen(request, timeout=self.timeout_s) as response:
                raw = response.read().decode("utf-8")
        except (HTTPError, URLError, TimeoutError) as exc:
            raise ExecutorError(f"Memory service {method} {path} failed: {exc}") from exc
        if not raw:
            return None
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ExecutorError(f"Memory service returned invalid JSON: {raw[:300]}") from exc
        if isinstance(value, Mapping) and value.get("ok") is False:
            error = value.get("error", {})
            raise ExecutorError(str(error.get("message", error)))
        return value.get("result") if isinstance(value, Mapping) and "result" in value else value

    async def _request(self, method: str, path: str, payload: Mapping[str, Any] | None = None) -> Any:
        if self._broken and path not in {"close", "usage"}:
            raise ExecutorError("Service request was interrupted; create a new run before further operations")
        cancelled = False
        try:
            result = await asyncio.to_thread(self._request_sync, method, path, payload)
            return result
        except asyncio.CancelledError:
            cancelled = True
            self._broken = True
            self.services.usage.add(UsageRecord("service", "unknown", usage_known=False, status="unknown",
                metadata={"reason": "Client stopped waiting; the remote operation may still be executing", "path": path}))
            raise
        except Exception:
            if method == "POST":
                self._broken = True
                self.services.usage.add(UsageRecord("service", "unknown", usage_known=False, status="unknown",
                    metadata={"reason": "Remote operation failed without a confirmed completion boundary", "path": path}))
            raise
        finally:
            if method == "POST" and self._started and not cancelled:
                await self._sync_usage()

    async def _sync_usage(self) -> None:
        try:
            usage = await asyncio.to_thread(self._request_sync, "GET", "usage")
            if not isinstance(usage, Mapping) or not isinstance(usage.get("usage_records"), list):
                raise ExecutorError("Service usage response must contain cumulative usage_records")
            self._usage_importer.update(usage["usage_records"], run_id=self.run_id)
            self.storage_bytes = usage.get("storage_bytes")
        except Exception as exc:
            message = str(exc)
            if self.usage_error != message:
                self.services.usage.add(UsageRecord("service", "unknown", usage_known=False,
                    status="failed", metadata={"reason": message}))
            self.usage_error = message

    async def setup(self, context: MethodContext) -> None:
        self.run_id = context.run_id
        self.headers["X-MemAtlas-Run-Id"] = context.run_id
        result = await self._request("POST", "setup", {
            "config": self.config.to_dict(),
            "services": self.services.serializable_spec(),
            "plugin": self.plugin_spec,
            "context": {"run_id": context.run_id, "metadata": dict(context.metadata)},
            "accounting_context": current_usage_context(),
        })
        if not isinstance(result, Mapping) or result.get("run_id") != context.run_id:
            raise ExecutorError("Memory service must acknowledge the run_id and isolate requests using X-MemAtlas-Run-Id")
        self._started = True
        await self._sync_usage()

    async def index(self, *, scope):
        return await self._request("POST", "index", {"scope": scope})

    async def add(self, unit, *, learning_episode=None):
        await self._request("POST", "add", {"unit": unit.to_dict(),
            "learning_episode": learning_episode.to_dict() if learning_episode is not None else None})

    async def search(self, query, *, level_ref=None, top_k, scope, extensions):
        result = await self._request("POST", "search", {"query": query, "level_ref": level_ref,
            "top_k": top_k, "scope": scope, "extensions": extensions})
        return MemoryContribution.from_dict(result or {})

    async def episode_end(self, episode_ref, *, scope, status, learning_episode=None):
        await self._request("POST", "episode_end", {"episode_ref": episode_ref, "scope": scope,
            "status": status, "learning_episode": learning_episode.to_dict() if learning_episode is not None else None})

    async def detail(self, memory_id, *, scope):
        result = await self._request("POST", "detail", {"memory_id": memory_id, "scope": scope})
        return MemoryContribution.from_dict(result or {})

    async def update(self, memory_id, content, *, scope):
        await self._request("POST", "update", {"memory_id": memory_id, "content": content, "scope": scope})

    async def delete(self, memory_id, *, scope):
        await self._request("POST", "delete", {"memory_id": memory_id, "scope": scope})

    async def flush(self) -> None:
        await self._request("POST", "flush")

    async def _freeze(self) -> None:
        await self._request("POST", "_freeze")

    async def _thaw(self) -> None:
        await self._request("POST", "_thaw")

    async def reset(self, scope) -> None:
        await self._request("POST", "reset", {"scope": scope})

    async def save(self) -> Mapping[str, Any]:
        return await self._request("POST", "save") or {}

    async def load(self, state: Mapping[str, Any]) -> None:
        await self._request("POST", "load", {"state": dict(state)})

    async def command(self, name: str, payload: Mapping[str, Any]) -> Any:
        return await self._request("POST", "command", {"name": name, "payload": dict(payload)})

    async def health(self) -> Any:
        return await self._request("GET", "health")

    async def collect_usage(self) -> Any:
        return await self._request("GET", "usage")

    async def close(self) -> None:
        if self._started:
            try:
                await self._sync_usage()
            finally:
                await self._request("POST", "close")
                self._started = False


__all__ = ["ServiceMemoryMethod"]
