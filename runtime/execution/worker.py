from __future__ import annotations

import asyncio
import contextlib
import json
import sys
import traceback
from typing import Any, Mapping

from mematlas.core.config import MemAtlasConfig
from mematlas.core.models import MemoryEpisode, MemoryEvent, MemoryUnit, content_value
from mematlas.core.scope import normalize_scope
from mematlas.core.plugin import MethodContext, MethodServices
from mematlas.core.registry import default_registry
from mematlas.core.telemetry import usage_context, usage_scope


class Worker:
    def __init__(self) -> None:
        self.session = None
        self.services = None

    async def dispatch(self, operation: str, payload: Mapping[str, Any]) -> Any:
        if operation == "_check":
            # Inspect dependency/source compatibility only. Provider constructors,
            # create_method(), setup(), databases and models are intentionally absent.
            from pathlib import Path
            from mematlas.core.serialization import construct, resolve_reference
            config = MemAtlasConfig.from_dict(payload["config"])
            spec = payload.get("services", {})
            services = MethodServices(
                source_roots=dict(spec.get("source_roots", {})),
                provider_configs=dict(spec.get("provider_configs", {})),
                artifact_dir=Path(spec.get("artifact_dir", ".mematlas/artifacts")),
                llm=object() if spec.get("llm") else None,
                embedder=object() if spec.get("embedder") else None,
                # Classes expose their protocol for inspection without running
                # constructors. Factory functions remain uninstantiated too.
                objects={name: resolve_reference(item["factory"])
                         for name, item in spec.get("objects", {}).items()},
            )
            for path in services.source_roots.values():
                services.add_import_root(path)
            plugin = construct(payload["plugin"]) if payload.get("plugin") else default_registry().get(config.method.method_id)
            if plugin.descriptor.method_id != config.method.method_id or plugin.descriptor.interface_version != 3:
                raise ValueError("Worker plugin identity/interface mismatch")
            if payload.get("descriptor") and json.dumps(plugin.descriptor.to_dict(), sort_keys=True) != json.dumps(payload["descriptor"], sort_keys=True):
                raise ValueError("Worker plugin descriptor differs from the registered plugin")
            return dict(plugin.healthcheck(config.method, services))
        if operation == "setup":
            config = MemAtlasConfig.from_dict(payload["config"])
            services = MethodServices.from_spec(payload.get("services", {}))
            self.services = services
            if payload.get("plugin"):
                from mematlas.core.serialization import construct
                plugin = construct(payload["plugin"])
            else:
                plugin = default_registry().get(config.method.method_id)
            if plugin.descriptor.method_id != config.method.method_id:
                raise ValueError("Worker plugin identity does not match requested method")
            if plugin.descriptor.interface_version != 3:
                raise ValueError("Worker requires Memory interface v3")
            if payload.get("descriptor") and json.dumps(plugin.descriptor.to_dict(), sort_keys=True) != json.dumps(payload["descriptor"], sort_keys=True):
                raise ValueError("Worker plugin descriptor differs from the registered plugin")
            self.session = plugin.create_method(config.method, services)
            ctx_data = payload.get("context", {})
            context = MethodContext(
                run_id=ctx_data.get("run_id", "worker"),
                method_config=config.method,
                runtime_config=config.runtime,
                metadata=ctx_data.get("metadata", {}),
            )
            from mematlas.core.telemetry import usage_scope
            from mematlas.providers.instrumentation import instrument_services
            instrument_services(services)
            with usage_scope(services.usage, "memory"):
                await self.session.setup(context)
            instrument_services(services)
            return None
        if self.session is None:
            raise RuntimeError("Worker session has not been set up")
        if operation == "_inspect":
            from mematlas.integrations.openhospital.inspection import inspect_native
            return await inspect_native(self.session, payload)
        if operation in {"_freeze", "_thaw"}:
            await getattr(self.session, operation)()
            return None
        if operation == "index":
            return await self.session.index(scope=normalize_scope(payload["scope"]))
        if operation == "add":
            episode = payload.get("learning_episode")
            await self.session.add(MemoryUnit.from_dict(payload["unit"]),
                learning_episode=MemoryEpisode.from_dict(episode) if episode is not None else None)
            return None
        if operation == "search":
            result = await self.session.search(content_value(payload["query"]), level_ref=payload.get("level_ref"),
                top_k=payload.get("top_k"), scope=normalize_scope(payload["scope"]), extensions=payload.get("extensions", {}))
            return result.to_dict()
        if operation == "episode_end":
            episode = payload.get("learning_episode")
            await self.session.episode_end(payload["episode_ref"], scope=normalize_scope(payload["scope"]),
                status=payload["status"], learning_episode=MemoryEpisode.from_dict(episode) if episode is not None else None)
            return None
        if operation == "detail":
            return (await self.session.detail(payload["memory_id"], scope=normalize_scope(payload["scope"]))).to_dict()
        if operation == "update":
            await self.session.update(payload["memory_id"], content_value(payload["content"]), scope=normalize_scope(payload["scope"]))
            return None
        if operation == "delete":
            await self.session.delete(payload["memory_id"], scope=normalize_scope(payload["scope"]))
            return None
        if operation == "flush":
            await self.session.flush()
            return None
        if operation == "reset":
            scope = normalize_scope(payload["scope"])
            await self.session.reset(scope)
            return None
        if operation == "save":
            from mematlas.core.state_files import pack_state
            return pack_state(await self.session.save(), self.services.artifact_dir)
        if operation == "load":
            from mematlas.core.state_files import unpack_state
            await self.session.load(unpack_state(payload["state"]))
            return None
        if operation == "command":
            return await self.session.command(payload["name"], payload.get("payload", {}))
        if operation == "stats":
            return {
                "usage_records": self.services.usage.to_dicts() if self.services else [],
                "resource_paths": [str(path) for path in self.services.resource_paths] if self.services else [],
            }
        if operation == "close":
            await self.session.close()
            return None
        raise ValueError(f"Unknown operation: {operation}")


async def main() -> None:
    worker = Worker()
    loop = asyncio.get_running_loop()
    telemetry_ledger = None
    sent_usage_cursor = 0
    async def heartbeat():
        from mematlas.core.activity import append
        while True:
            append("heartbeat")
            await asyncio.sleep(5)
    asyncio.create_task(heartbeat())
    while True:
        line = await loop.run_in_executor(None, sys.stdin.buffer.readline)
        if not line:
            return
        request = json.loads(line)
        response = {"id": request.get("id")}
        try:
            with contextlib.redirect_stdout(sys.stderr), usage_context(**request.get("accounting_context", {})):
                result = await worker.dispatch(request["operation"], request.get("payload", {}))
            response.update({"ok": True, "result": result})
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            response.update({
                "ok": False,
                "error": __import__("mematlas.core.recovery", fromlist=["error_info"]).error_info(exc),
            })
        if worker.services is not None:
            if telemetry_ledger is not worker.services.usage:
                telemetry_ledger = worker.services.usage
                sent_usage_cursor = 0
            # Background providers may finish between requests. A cursor taken
            # at request entry would silently discard all those calls.
            pending_usage = worker.services.usage.to_dicts()[sent_usage_cursor:]
            sent_usage_cursor += len(pending_usage)
            response["telemetry"] = {
                "usage_records": pending_usage,
                "resource_paths": [str(path) for path in worker.services.resource_paths],
            }
        sys.stdout.write(json.dumps(response, ensure_ascii=False, default=str) + "\n")
        sys.stdout.flush()
        if request.get("operation") == "close":
            return


if __name__ == "__main__":
    asyncio.run(main())
