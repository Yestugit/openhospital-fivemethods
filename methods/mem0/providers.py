"""Transport-only native providers; native prompts, batches and ranking are untouched.

Imported only after native runtime paths and telemetry have been configured.
"""
import asyncio
from contextvars import Context, copy_context
from dataclasses import dataclass, field
from threading import Lock
from typing import Any

from mem0.configs.llms.base import BaseLlmConfig
from mem0.llms.base import LLMBase

_BINDINGS: dict[str, "Binding"] = {}


@dataclass
class Binding:
    services: Any
    loop: asyncio.AbstractEventLoop
    context: Context = field(default_factory=copy_context)
    failures: list[Exception] = field(default_factory=list)
    lock: Lock = field(default_factory=Lock)

    def call(self, coroutine):
        try:
            # Native entity extraction/search uses its own ThreadPoolExecutor.
            # Explicitly carry the enclosing operation's usage context into it.
            future = self.context.copy().run(asyncio.run_coroutine_threadsafe, coroutine, self.loop)
            return future.result()
        except Exception as exc:
            with self.lock:
                self.failures.append(exc)
            raise


class BridgeLlmConfig(BaseLlmConfig):
    def __init__(self, binding, **kwargs):
        super().__init__(**kwargs)
        self.binding = binding


class NativeLLM(LLMBase):
    def generate_response(self, messages, response_format=None, tools=None, tool_choice="auto", **kwargs):
        binding = _BINDINGS[self.config.binding]
        options = self._get_supported_params(**kwargs)
        if response_format:
            options["response_format"] = response_format
        if tools:
            options.update(tools=tools, tool_choice=tool_choice)
        return binding.call(binding.services.llm.complete(messages, **options))


class NativeEmbedder:
    def __init__(self, config):
        self.config = config
        # The embedding factory accepts only BaseEmbedderConfig fields; the
        # private model token locates a service and is never sent to a model.
        self.binding = _BINDINGS[config.model]

    def embed(self, text, memory_action="add"):
        return self.embed_batch([text], memory_action)[0]

    def embed_batch(self, texts, memory_action="add"):
        if not texts:
            return []
        vectors = self.binding.call(self.binding.services.embedder.embed(list(texts)))
        if len(vectors) != len(texts) or any(len(row) != self.config.embedding_dims for row in vectors):
            error = ValueError("Embedding provider returned incompatible batch shape/dimensions")
            with self.binding.lock:
                self.binding.failures.append(error)
            raise error
        return vectors
