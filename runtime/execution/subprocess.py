from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4
from collections import deque

from mematlas.core.config import MemAtlasConfig
from mematlas.core.errors import ExecutorError
from mematlas.core.models import MemoryContribution, MemoryEpisode, MemoryEvent, MemoryUnit
from mematlas.core.plugin import BaseMemoryMethod, MethodContext, MethodServices
from mematlas.core.telemetry import UsageRecord, current_usage_context


_STREAM_LIMIT_BYTES = 16 * 1024 * 1024


def worker_command(config) -> list[str]:
    # This repository also supports a flat checkout named MemAtlas. Bootstrap
    # its package explicitly, without installing it or borrowing parent packages.
    root = Path(__file__).resolve().parents[2]
    bootstrap = (
        "import importlib.util,runpy,sys; "
        "root=sys.argv.pop(1); "
        "spec=importlib.util.spec_from_file_location('mematlas',root+'/__init__.py',submodule_search_locations=[root]); "
        "module=importlib.util.module_from_spec(spec); sys.modules['mematlas']=module; "
        "spec.loader.exec_module(module); "
        "runpy.run_module('mematlas.runtime.execution.worker',run_name='__main__')"
    )
    return [config.runtime.python_executable or sys.executable, "-c", bootstrap, str(root)]


def worker_environment(services: MethodServices) -> dict[str, str]:
    """One import/environment contract for readonly probes and execution."""
    env = os.environ.copy()
    source_dir = str(Path(__file__).resolve().parents[3])
    env["PYTHONPATH"] = os.pathsep.join(
        item for item in (source_dir, *services.source_roots.values(), env.get("PYTHONPATH", "")) if item
    )
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env.setdefault(name, "1")
    return env


def probe_environment(config, plugin, services):
    """Check imports/source in the selected Python without constructing a backend."""
    import subprocess
    from mematlas.core.serialization import plugin_spec
    payload = {"config": config.to_dict(), "services": services.serializable_spec(),
               "plugin": plugin_spec(plugin), "descriptor": plugin.descriptor.to_dict()}
    request = {"id": "preflight", "operation": "_check", "payload": payload}
    executable = config.runtime.python_executable or sys.executable
    try:
        completed = subprocess.run(
            worker_command(config),
            input=json.dumps(request) + "\n", capture_output=True, text=True,
            encoding="utf-8", env=worker_environment(services), timeout=30,
        )
        response = json.loads(completed.stdout)
        if completed.returncode or not response.get("ok"):
            raise ValueError(str(response.get("error", {})))
        return {**response["result"], "python_executable": executable, "inspection": "worker"}
    except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
        return {"ready": False, "status": "unavailable", "inspection": "worker",
                "python_executable": executable, "reason": str(exc)}


class SubprocessMemoryMethod(BaseMemoryMethod):
    def __init__(self, config: MemAtlasConfig, services: MethodServices, plugin_spec=None, descriptor=None):
        self.config = config
        self.services = services
        self.process: asyncio.subprocess.Process | None = None
        self._lock = asyncio.Lock()
        self._stderr_task: asyncio.Task | None = None
        self._stderr_lines = deque(maxlen=200)
        self.plugin_spec = plugin_spec
        self.descriptor = descriptor
        self._broken = False

    async def _drain_stderr(self) -> None:
        assert self.process and self.process.stderr
        stream = self.process.stderr
        while True:
            line = await stream.readline()
            if not line:
                return
            decoded = line.decode(errors="replace").rstrip()
            self._stderr_lines.append(decoded)
            path = Path(self.services.artifact_dir) / "worker-stderr.log"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as journal:
                journal.write(decoded + "\n")
            self.services.logger.debug("method-worker: %s", decoded)

    async def _start_process(self) -> None:
        if self.process is not None:
            return
        env = worker_environment(self.services)
        self.process = await asyncio.create_subprocess_exec(
            *worker_command(self.config),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
            limit=_STREAM_LIMIT_BYTES,
        )
        self._stderr_task = asyncio.create_task(self._drain_stderr())

    async def _request(self, operation: str, payload: Mapping[str, Any] | None = None) -> Any:
        if self._broken:
            raise ExecutorError("Worker is unavailable after interruption; create a new Memory or explicitly resume")
        request_id = uuid4().hex
        request = {"id": request_id, "operation": operation, "payload": dict(payload or {}),
                   "accounting_context": current_usage_context()}
        encoded = (json.dumps(request) + "\n").encode("utf-8")
        async with self._lock:
            if self._broken:
                raise ExecutorError("Worker is unavailable after interruption")
            await self._start_process()
            assert self.process and self.process.stdin and self.process.stdout
            try:
                self.process.stdin.write(encoded)
                await self.process.stdin.drain()
                line = await self.process.stdout.readline()
            except BaseException:
                await self._abort()
                raise
        if not line:
            code = await self.process.wait()
            detail = "\n".join(list(self._stderr_lines)[-20:])
            await self._abort()
            raise ExecutorError(
                f"Method worker exited without response, code={code}"
                + (f"; stderr={detail}" if detail else "")
            )
        try:
            response = json.loads(line.decode("utf-8"))
        except json.JSONDecodeError as exc:
            await self._abort()
            raise ExecutorError(f"Invalid worker response: {line.decode(errors='replace')}") from exc
        if response.get("id") != request_id:
            await self._abort()
            raise ExecutorError("Method worker response id mismatch")
        telemetry = response.get("telemetry", {})
        for record in telemetry.get("usage_records", []):
            self.services.usage.add(UsageRecord(**record), imported=True)
        for path in telemetry.get("resource_paths", []):
            self.services.register_resource(path)
        if not response.get("ok"):
            error = response.get("error", {})
            exc = ExecutorError(f"{error.get('type', 'WorkerError')}: {error.get('message', '')}")
            exc.remote_error = error
            raise exc
        return response.get("result")

    async def _abort(self) -> None:
        already_broken = self._broken
        self._broken = True
        if not already_broken:
            self.services.usage.add(UsageRecord("subprocess", "unknown", usage_known=False, status="unknown",
                metadata={"reason": "Worker interrupted before final usage could be collected"}))
        process, self.process = self.process, None
        if process is not None:
            await self._reap(process, abort=True)
        if self._stderr_task:
            self._stderr_task.cancel()
            await asyncio.gather(self._stderr_task, return_exceptions=True)
            self._stderr_task = None

    async def _reap(self, process, *, abort=False):
        # After the final protocol response native libraries may still print
        # during interpreter shutdown. Consume stdout so a full Windows pipe
        # cannot keep asyncio's exit waiter alive after the process has exited.
        async def drain():
            if process.stdout is not None:
                while await process.stdout.read(65536):
                    pass

        draining = asyncio.create_task(drain())
        if process.stdin is not None:
            process.stdin.close()
        try:
            if abort and process.returncode is None:
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
            try:
                await asyncio.wait_for(process.wait(), timeout=5)
            except TimeoutError:
                if process.returncode is None:
                    try:
                        process.kill()
                    except ProcessLookupError:
                        pass
                await asyncio.wait_for(process.wait(), timeout=5)
            await asyncio.wait_for(draining, timeout=5)
        finally:
            if not draining.done():
                draining.cancel()
            await asyncio.gather(draining, return_exceptions=True)

    async def setup(self, context: MethodContext) -> None:
        await self._request("setup", {
            "config": self.config.to_dict(),
            "services": self.services.serializable_spec(),
            "plugin": self.plugin_spec,
            "descriptor": self.descriptor,
            "context": {
                "run_id": context.run_id,
                "metadata": dict(context.metadata),
            },
        })

    async def index(self, *, scope):
        return await self._request("index", {"scope": scope})

    async def add(self, unit, *, learning_episode=None):
        await self._request("add", {"unit": unit.to_dict(),
            "learning_episode": learning_episode.to_dict() if learning_episode is not None else None})

    async def search(self, query, *, level_ref=None, top_k, scope, extensions):
        result = await self._request("search", {"query": query, "level_ref": level_ref,
            "top_k": top_k, "scope": scope, "extensions": extensions})
        return MemoryContribution.from_dict(result or {})

    async def episode_end(self, episode_ref, *, scope, status, learning_episode=None):
        await self._request("episode_end", {"episode_ref": episode_ref, "scope": scope,
            "status": status, "learning_episode": learning_episode.to_dict() if learning_episode is not None else None})

    async def detail(self, memory_id, *, scope):
        result = await self._request("detail", {"memory_id": memory_id, "scope": scope})
        return MemoryContribution.from_dict(result or {})

    async def update(self, memory_id, content, *, scope):
        await self._request("update", {"memory_id": memory_id, "content": content, "scope": scope})

    async def delete(self, memory_id, *, scope):
        await self._request("delete", {"memory_id": memory_id, "scope": scope})

    async def flush(self) -> None:
        await self._request("flush")

    async def _inspect(self, options):
        return await self._request("_inspect", options)

    async def _freeze(self) -> None:
        await self._request("_freeze")

    async def _thaw(self) -> None:
        await self._request("_thaw")

    async def reset(self, scope) -> None:
        await self._request("reset", {"scope": scope})

    async def save(self) -> Mapping[str, Any]:
        return await self._request("save")

    async def load(self, state: Mapping[str, Any]) -> None:
        await self._request("load", {"state": dict(state)})

    async def command(self, name: str, payload: Mapping[str, Any]) -> Any:
        return await self._request("command", {"name": name, "payload": dict(payload)})

    async def close(self) -> None:
        if self.process is None:
            return
        try:
            await self._request("close")
        finally:
            process, self.process = self.process, None
            try:
                if process is not None:
                    await self._reap(process)
            finally:
                if self._stderr_task:
                    try:
                        await asyncio.wait_for(self._stderr_task, timeout=5)
                    finally:
                        self._stderr_task = None
                self._stderr_lines.clear()
