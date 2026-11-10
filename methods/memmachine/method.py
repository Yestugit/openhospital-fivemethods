"""Public v3 contract around the pinned upstream native backend."""
from __future__ import annotations
import os
import sys
from importlib import metadata
from mematlas.core.errors import ConfigurationError
from mematlas.core.plugin import BaseMemoryMethod, MethodPlugin
from mematlas.methods.services import prepare_services
from mematlas.providers.instrumentation import instrument_services
from .config import _DESCRIPTOR as DESCRIPTOR
from .utils import NativeAdapter, inspect_native_source


class MemMachineMemory(NativeAdapter, BaseMemoryMethod):
    descriptor = DESCRIPTOR
    method_id = "memmachine"
    levels = (("default", "Native episodic and semantic results", "native episodes, summaries and features"), ("episodic", "Native long and short term episodic memory", "episodes and summaries"), ("semantic", "Native extracted semantic features", "semantic features"))

    def __init__(self, config, services):
        self._initialize(config, services)

    async def setup(self, context):
        report = MemMachinePlugin().healthcheck(self.config, self.services)
        if not report["ready"]:
            raise ConfigurationError(f"memmachine requirements are not satisfied: {report}")
        if self.backend is None:
            prepare_services(self.params, self.services, method_id=self.method_id)
            instrument_services(self.services)
            from .core import MemMachineCore
            self.backend = MemMachineCore(self.params, self.services, context.run_id)
            await self.backend.setup()


class MemMachinePlugin(MethodPlugin):
    descriptor = DESCRIPTOR

    def create_method(self, config, services):
        return MemMachineMemory(config, services)

    def healthcheck(self, config, services):
        backend = services.object("memmachine.backend")
        if backend is not None:
            required = ("add", "search", "flush", "freeze", "thaw", "export_state", "import_state", "reset")
            missing = [name for name in required if not callable(getattr(backend, name, None))]
            errors = self.validate_config(config)
            return {"method_id": "memmachine", "ready": not errors and not missing,
                    "status": "ready" if not errors and not missing else "unavailable",
                    "config_errors": errors, "missing_backend_operations": missing,
                    "injected_backend": True}
        report = dict(super().healthcheck(config, services))
        errors = list(report["config_errors"])
        from mematlas.core.provider_config import resolve_providers
        dimensions = resolve_providers(config.params, services.provider_configs)["embedding"].dimensions
        expected = config.params.get("embedding_dims", 1024)
        if dimensions is not None and dimensions != expected:
            errors.append(f"embedding_dims={expected} differs from providers.embedding.dimensions={dimensions}")
        if sys.version_info < (3, 12):
            errors.append("MemMachine requires Python >=3.12 in its own environment")
        source = inspect_native_source("memmachine", "memmachine", {"memmachine_server": "packages/server/src/memmachine_server/", "memmachine_common": "packages/common/src/memmachine_common/"})
        report["native_source"] = source
        errors.extend(source["errors"])
        report["config_errors"] = errors
        report["ready"] = report["ready"] and not errors
        report["status"] = "ready" if report["ready"] else "unavailable"
        return report


PLUGIN = MemMachinePlugin()
