from __future__ import annotations

import os
from dataclasses import dataclass

from mematlas.core.telemetry import UsageLedger, UsageRecord, active_usage


@dataclass
class OpenAICompatibleEmbedder:
    model: str
    api_key: str | None = None
    api_key_env: str = "OPENAI_API_KEY"
    base_url: str | None = None
    dimensions: int | None = None
    usage: UsageLedger | None = None

    def construction_spec(self):
        from dataclasses import fields
        values = {field.name: getattr(self, field.name) for field in fields(self) if field.init and field.name != "usage"}
        return {"factory": "mematlas.providers.embeddings:OpenAICompatibleEmbedder", "kwargs": values}

    def __post_init__(self) -> None:
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise ImportError("Install mematlas[openai] to use OpenAICompatibleEmbedder") from exc
        self._client = AsyncOpenAI(
            api_key=self.api_key or os.environ.get(self.api_key_env),
            base_url=self.base_url,
            max_retries=0,
        )

    async def embed(self, texts: list[str]) -> list[list[float]]:
        # The owned Qwen endpoint accepts at most 32 texts per HTTP request.
        # Keep the upstream logical batch and ordering intact while respecting
        # that transport limit. Meter every actual request, including retries.
        from urllib.parse import urlparse
        local_qwen = self.model == "Qwen/Qwen3-Embedding-0.6B" and urlparse(self.base_url or "").hostname in {"127.0.0.1", "localhost", "::1"}
        if local_qwen and len(texts) > 32:
            vectors = []
            for offset in range(0, len(texts), 32):
                vectors.extend(await self.embed(texts[offset:offset + 32]))
            return vectors
        kwargs = {"model": self.model, "input": texts}
        if self.dimensions is not None:
            kwargs["dimensions"] = self.dimensions
        ledger = self.usage or (active_usage()[0] if active_usage() else None)
        from .requests import metered_request
        response = await metered_request(lambda: self._client.embeddings.create(**kwargs), ledger=ledger,
                                         provider="openai_compatible_embedding", model=self.model, kind="embedding")
        return [list(item.embedding) for item in response.data]
