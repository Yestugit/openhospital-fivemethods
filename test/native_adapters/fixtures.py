"""Controlled executor fixtures, with no third-party dependencies or providers."""
from dataclasses import replace
import asyncio
from mematlas.test.test_memory_interface import RecordingPlugin, RecordingMethod


class PausingMethod(RecordingMethod):
    def __init__(self, failure=None, mutate=False):
        super().__init__()
        self.paused = False
        self.calls = []
        self.failure = failure
        self.mutate = mutate

    async def _freeze(self):
        self.calls.append("freeze")
        self.paused = True
        if self.failure == "freeze":
            raise RuntimeError("freeze failed")

    async def _thaw(self):
        self.calls.append("thaw")
        if self.failure == "thaw":
            raise RuntimeError("thaw failed")
        self.paused = False

    async def save(self):
        if not self.paused:
            raise RuntimeError("snapshot raced scheduler")
        if self.failure == "save":
            raise RuntimeError("snapshot failed")
        if self.failure == "save_state":
            from mematlas.core.errors import MemoryStateError
            raise MemoryStateError("snapshot failed")
        return await super().save()

    async def load(self, state):
        assert self.paused, "restore must hold scheduler pause"
        await super().load(state)

    async def search(self, *args, **kwargs):
        if self.mutate:
            self.items["leak"] = {"memory_id": "leak", "content": "mutated read"}
        return await super().search(*args, **kwargs)


class PausingPlugin(RecordingPlugin):
    def __init__(self, failure=None, mutate=False):
        super().__init__()
        self.options = {"failure": failure, "mutate": mutate}
        self.descriptor = replace(self.descriptor,
            capabilities=replace(self.descriptor.capabilities, read_only_mode="snapshot"))

    def create_method(self, config, services):
        return PausingMethod(**self.options)

    def construction_spec(self):
        return {"factory": "mematlas.test.native_adapters.fixtures:PausingPlugin", "kwargs": self.options}


class LargeStateMethod(RecordingMethod):
    async def save(self):
        return {**await super().save(), "native_vectors": "v" * (17 * 1024 * 1024)}

    async def load(self, state):
        assert len(state["native_vectors"]) == 17 * 1024 * 1024
        await super().load(state)


class LargeStatePlugin(RecordingPlugin):
    def create_method(self, config, services):
        return LargeStateMethod()

    def construction_spec(self):
        return {"factory": "mematlas.test.native_adapters.fixtures:LargeStatePlugin"}


class BackgroundUsageMethod(RecordingMethod):
    def __init__(self, services):
        super().__init__()
        self.services = services
        self.pending = []

    async def _learn(self, *args, **kwargs):
        await super()._learn(*args, **kwargs)
        async def finished():
            from mematlas.core.telemetry import UsageRecord
            await asyncio.sleep(0.01)
            self.services.usage.add(UsageRecord("controlled-background", "fixture", prompt_tokens=7))
        self.pending.append(asyncio.create_task(finished()))

    async def close(self):
        await asyncio.gather(*self.pending)


class BackgroundUsagePlugin(RecordingPlugin):
    def create_method(self, config, services):
        return BackgroundUsageMethod(services)

    def construction_spec(self):
        return {"factory": "mematlas.test.native_adapters.fixtures:BackgroundUsagePlugin"}


class RecoverableFailureMethod(RecordingMethod):
    async def search(self, *args, **kwargs):
        from mematlas.providers.requests import LLMRequestPaused
        try:
            raise TimeoutError("controlled upstream timeout")
        except TimeoutError as exc:
            raise LLMRequestPaused("controlled eight attempts exhausted") from exc


class RecoverableFailurePlugin(RecordingPlugin):
    def create_method(self, config, services):
        return RecoverableFailureMethod()

    def construction_spec(self):
        return {"factory": "mematlas.test.native_adapters.fixtures:RecoverableFailurePlugin"}


class ShutdownNoiseMethod(RecordingMethod):
    def __init__(self, services):
        super().__init__()
        self.services = services

    async def close(self):
        import atexit
        import sys
        from pathlib import Path
        marker = Path(self.services.artifact_dir) / "shutdown-complete"

        def finish():
            # Greater than the worker reader limit: a caller that only waits
            # for process exit can deadlock without consuming the trailing pipe.
            sys.stdout.write("shutdown diagnostics\n" * 1_000_000)
            sys.stdout.flush()
            marker.write_text("graceful", encoding="utf-8")
        atexit.register(finish)


class ShutdownNoisePlugin(RecordingPlugin):
    def create_method(self, config, services):
        return ShutdownNoiseMethod(services)

    def construction_spec(self):
        return {"factory": "mematlas.test.native_adapters.fixtures:ShutdownNoisePlugin"}
