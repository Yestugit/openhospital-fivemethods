from __future__ import annotations

import inspect
import os
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Mapping

from mematlas.core.telemetry import UsageLedger, UsageRecord, active_usage


@dataclass
class CallableLLM:
    function: Callable[..., str | Awaitable[str]]

    async def complete(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        result = self.function(messages, **kwargs)
        if inspect.isawaitable(result):
            result = await result
        return str(result)


@dataclass
class OpenAICompatibleLLM:
    model: str
    api_key: str | None = None
    api_key_env: str = "OPENAI_API_KEY"
    base_url: str | None = None
    default_options: Mapping[str, Any] = field(default_factory=dict)
    usage: UsageLedger | None = None
    input_cost_per_million: float | None = None
    output_cost_per_million: float | None = None

    def construction_spec(self):
        from dataclasses import fields
        values = {field.name: getattr(self, field.name) for field in fields(self) if field.init and field.name != "usage"}
        return {"factory": "mematlas.providers.llm:OpenAICompatibleLLM", "kwargs": values}

    def __post_init__(self) -> None:
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise ImportError("Install mematlas[openai] to use OpenAICompatibleLLM") from exc
        self._client = AsyncOpenAI(
            api_key=self.api_key or os.environ.get(self.api_key_env),
            base_url=self.base_url,
            max_retries=0,
            timeout=120.0,
        )

    async def complete(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        options = {**dict(self.default_options), **kwargs}
        if "extra_body" in self.default_options and "extra_body" in kwargs:
            options["extra_body"] = {**self.default_options["extra_body"], **kwargs["extra_body"]}
        model = str(options.pop("model", self.model))
        prompt_cache = options.pop("prompt_cache", False)
        journal = options.pop("response_journal", None)
        if prompt_cache and messages and messages[0].get("role") == "system":
            # Stable extraction instructions are a separate message boundary.
            import copy
            messages = copy.deepcopy(messages)
            content = messages[0].get("content")
            if isinstance(content, str) and len(content) >= 4096:
                messages[0]["content"] = [{"type": "text", "text": content,
                    "cache_control": {"type": "ephemeral"}}]
        ledger = self.usage or (active_usage()[0] if active_usage() else None)
        if ledger is not None and self.input_cost_per_million is not None and self.output_cost_per_million is not None:
            ledger.model_pricing.setdefault(model, {"input_cost_per_million": self.input_cost_per_million,
                                                    "output_cost_per_million": self.output_cost_per_million})
        from .requests import metered_request
        response = await metered_request(lambda: self._client.chat.completions.create(model=model, messages=messages, **options),
                                         ledger=ledger, provider="openai_compatible", model=model, kind="llm")
        if journal:
            import json
            from pathlib import Path
            from datetime import datetime, timezone
            from mematlas.core.telemetry import current_usage_context
            path = Path(journal)
            path.parent.mkdir(parents=True, exist_ok=True)
            usage = getattr(response, "usage", None)
            with path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"timestamp": datetime.now(timezone.utc).isoformat(),
                    "model": model, "context": current_usage_context(), "messages": messages,
                    "response": response.choices[0].message.content,
                    "usage": usage.model_dump() if hasattr(usage, "model_dump") else None}, ensure_ascii=False) + "\n")
        return response.choices[0].message.content or ""
