"""Memory v3 adapter for Microsoft's original Memora implementation."""
import asyncio
from mematlas.core.plugin import MethodPlugin
from mematlas.methods.manifest import load_method_descriptor
from mematlas.methods.services import prepare_services
from .native import NativeMemora, REVISION
from .utils import EpisodeNativeAdapter, checked_source
from mematlas.core.errors import ConfigurationError

DESCRIPTOR = load_method_descriptor(__file__)


class MemoraMemory(EpisodeNativeAdapter):
    method_id = "memora"
    level_ref = "memories"
    view_description = "Official primary/cue search and iterative policy expansion with linked episodic context"
    granularity = "native factual and episodic memories"
    backend_type = NativeMemora
    native_revision = REVISION
    default_top_k = 30

    async def setup(self, context):
        prepare_services(self.params, self.services, method_id=self.method_id)
        self.services._native_io_loop = asyncio.get_running_loop()
        checked_source(self.services, self.method_id, REVISION)

    def _make_backend(self, directory, scope, state=None):
        return self.backend_type(directory, scope, self.params, self.services, state=state)


class MemoraPlugin(MethodPlugin):
    descriptor = DESCRIPTOR

    def create_method(self, config, services):
        return MemoraMemory(config, services)

    def healthcheck(self, config, services):
        report = dict(super().healthcheck(config, services))
        try:
            checked_source(services, "memora", REVISION)
        except (ValueError, OSError, RuntimeError, ConfigurationError) as exc:
            report["config_errors"].append(str(exc))
        report["ready"] = not (report["config_errors"] or report["missing_packages"] or report["missing_services"])
        report["status"] = "ready" if report["ready"] else "unavailable"
        return report


PLUGIN = MemoraPlugin()
