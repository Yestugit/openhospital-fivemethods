"""SigLIP-compatible chat embedding endpoint used by M2A's visual store."""

import base64
import mimetypes
import os
from pathlib import Path

from mematlas.core.telemetry import UsageRecord, active_usage


class MultimodalEmbedder:
    def __init__(self, model: str, base_url: str, api_key_env: str = "OPENAI_API_KEY"):
        from openai import AsyncOpenAI
        self.model = model
        self.base_url = base_url
        self.api_key_env = api_key_env
        self.client = AsyncOpenAI(base_url=base_url, api_key=os.environ.get(api_key_env, "EMPTY"))

    def construction_spec(self):
        return {"factory": "mematlas.methods.m2a.embeddings:MultimodalEmbedder", "kwargs": {
            "model": self.model, "base_url": self.base_url, "api_key_env": self.api_key_env,
        }}

    async def _embed(self, content):
        from openai.types import CreateEmbeddingResponse
        response = await self.client.post("/embeddings", cast_to=CreateEmbeddingResponse, body={
            "model": self.model, "messages": [{"role": "user", "content": content}],
            "encoding_format": "float",
        })
        context = active_usage()
        if context:
            usage = getattr(response, "usage", None)
            context[0].add(UsageRecord(provider="m2a_multimodal_embedding", model=self.model,
                prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
                usage_known=usage is not None))
        return list(response.data[0].embedding)

    async def embed_text(self, text: str):
        return await self._embed([{"type": "text", "text": text}])

    async def encode_image(self, image: str):
        if not image.startswith(("http://", "https://", "data:")):
            mime = mimetypes.guess_type(image)[0] or "image/png"
            image = f"data:{mime};base64," + base64.b64encode(Path(image).read_bytes()).decode("ascii")
        return await self._embed([{"type": "image_url", "image_url": {"url": image}}])
