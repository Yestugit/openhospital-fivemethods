"""Memory v3 adapter for the original EMem implementation."""
import asyncio
from mematlas.core.plugin import MethodPlugin
from mematlas.methods.manifest import load_method_descriptor
from mematlas.methods.services import prepare_services
from .native import NativeEMem, REVISION
from .utils import EpisodeNativeAdapter, checked_source
from mematlas.core.errors import ConfigurationError
from mematlas.core.provider_config import resolve_providers
import os

DESCRIPTOR = load_method_descriptor(__file__)


class EMemMemory(EpisodeNativeAdapter):
    method_id = "emem"
    level_ref = "edus"
    view_description = "Official contextual EDUs with source attribution, reranking and optional graph PPR"
    granularity = "elementary discourse units"
    backend_type = NativeEMem
    native_revision = REVISION

    async def setup(self, context):
        prepare_services(self.params, self.services, method_id=self.method_id)
        self.services._native_io_loop = asyncio.get_running_loop()
        checked_source(self.services, self.method_id, REVISION)

    def _make_backend(self, directory, scope, state=None):
        return self.backend_type(directory, scope, self.params, self.services, state=state)

    async def save(self):
        state = await super().save()
        profile = self.params.get("json_output_mode", "native")
        if profile != "native":
            state["adapter_profile"] = profile
        return state

    async def load(self, state):
        if state.get("adapter_profile", "native") != self.params.get("json_output_mode", "native"):
            raise ValueError("EMem checkpoint JSON compatibility profile mismatch; automatic migration is disabled")
        return await super().load(state)


class EMemPlugin(MethodPlugin):
    descriptor = DESCRIPTOR

    def create_method(self, config, services):
        return EMemMemory(config, services)

    def healthcheck(self, config, services):
        report = dict(super().healthcheck(config, services))
        try:
            checked_source(services, "emem", REVISION)
        except (ValueError, OSError, RuntimeError, ConfigurationError) as exc:
            report["config_errors"].append(str(exc))
        provider = resolve_providers(config.params, services.provider_configs)["llm"]
        if not (os.environ.get(provider.api_key_env) or getattr(services.llm, "api_key", None)):
            report["config_errors"].append(f"EMem native CacheOpenAI requires {provider.api_key_env} for its configured LLM endpoint")
        report["ready"] = not (report["config_errors"] or report["missing_packages"] or report["missing_services"])
        report["status"] = "ready" if report["ready"] else "unavailable"
        return report


PLUGIN = EMemPlugin()
