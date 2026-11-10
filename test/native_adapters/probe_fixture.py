"""Import-only preflight fixture, usable in an interpreter without pytest."""
from pathlib import Path
import sys

from mematlas import MethodPlugin, MethodCapabilities, MethodDescriptor, Operation
from mematlas.core.capabilities import IntegrationSlot, MemoryForm
from mematlas.core.models import Modality


class ProbePlugin(MethodPlugin):
    descriptor = MethodDescriptor("probe", "Probe", "Isolated environment probe",
        MethodCapabilities(frozenset({Modality.TEXT}), frozenset({MemoryForm.RECORD}),
            frozenset({Operation.INDEX, Operation.ADD, Operation.SEARCH, Operation.EPISODE_END}),
            frozenset({IntegrationSlot.CONTEXT})))

    def healthcheck(self, config, services):
        return {"ready": (Path(sys.prefix) / "dependency-marker").is_file(), "prefix": sys.prefix,
                "provider_declared": services.llm is not None,
                "backend_inspectable": callable(getattr(services.object("backend"), "add", None))}

    def create_method(self, config, services):
        raise AssertionError("preflight must never construct a backend")


def forbidden_provider():
    raise AssertionError("preflight must never construct a model provider")


class InspectableBackend:
    async def add(self):
        pass

    def __init__(self):
        raise AssertionError("readonly probe constructed a database backend")
