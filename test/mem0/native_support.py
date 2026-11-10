"""Controlled providers for actual pinned-native regression tests (no offline engine)."""
import hashlib
import json
import math

from mematlas import MemAtlasConfig, MethodConfig, RuntimeConfig
from mematlas.core.plugin import MethodServices


def require_native():
    import pytest
    from mematlas.methods.mem0.provenance import verify_source
    errors = verify_source()
    if errors:
        pytest.skip("Pinned Mem0 native interpreter required: " + "; ".join(errors))


class Vectors:
    dimensions = 16

    def __init__(self):
        self.calls = []

    def construction_spec(self):
        return {"factory": "mematlas.test.mem0.native_support:Vectors"}

    async def embed(self, texts):
        self.calls.append(list(texts))
        vectors = []
        for text in texts:
            data = hashlib.sha256(text.lower().encode()).digest()
            row = [1 + data[index] / 255 for index in range(self.dimensions)]
            norm = math.sqrt(sum(value * value for value in row))
            vectors.append([value / norm for value in row])
        return vectors


class ExtractionLLM:
    model = "controlled-native-test"

    def __init__(self, facts=None):
        self.calls = []
        self.facts = facts or ["Helios port is 8443", "Alice prefers tea"]
        self.fail = False

    def construction_spec(self):
        return {"factory": "mematlas.test.mem0.native_support:ExtractionLLM", "kwargs": {"facts": self.facts}}

    async def complete(self, messages, **options):
        self.calls.append((messages, options))
        if self.fail:
            raise RuntimeError("controlled provider failure")
        return json.dumps({"memory": [{"event": "ADD", "text": text} for text in self.facts]})


def services(*, facts=None):
    return MethodServices(llm=ExtractionLLM(facts), embedder=Vectors())


def config_for(tmp_path, **params):
    return MemAtlasConfig(method=MethodConfig("mem0", {"embedding_dims": 16, **params}),
        runtime=RuntimeConfig(artifact_dir=str(tmp_path)), scope={"user_id": "alice"})
