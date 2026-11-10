"""Service-hosted doctor LLM proxy for participant agents."""

from __future__ import annotations

from typing import Any

try:
    from .llm_client import OpenAICompatibleChatClient
    from .patients.service import estimate_tokens
    from .token_usage import InMemoryTokenUsageStore, TokenUsageStore
except ImportError:
    from llm_client import OpenAICompatibleChatClient
    from patients.service import estimate_tokens
    from token_usage import InMemoryTokenUsageStore, TokenUsageStore


class DoctorLLMService:
    """Proxy doctor-agent LLM calls through the contest service model config."""

    def __init__(
        self,
        *,
        usage_store: TokenUsageStore | None = None,
        model_key_pool: Any | None = None,
        model_api: str | None = None,
        model_name: str | None = None,
        timeout_seconds: int | None = None,
        request_retries: int | None = None,
        default_api_key: str | None = None,
        dataset_key: str = "",
    ):
        self.model_api = model_api or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        self.model_name = model_name or "qwen3.5-flash"
        self.timeout_seconds = timeout_seconds or 120
        self.request_retries = request_retries or 2
        self.default_api_key = str(default_api_key or "").strip()
        self.model_key_pool = model_key_pool
        self.dataset_key = str(dataset_key or "").strip()
        self.usage_store = usage_store or InMemoryTokenUsageStore()

    async def chat_async(
        self,
        payload: dict[str, Any],
        *,
        team_id: str,
        allow_payload_api_key: bool,
    ) -> dict[str, Any]:
        messages = _messages(payload.get("messages"))
        if not messages:
            prompt = _optional_text(payload.get("prompt"))
            if not prompt:
                raise ValueError("messages or prompt is required.")
            messages = []
            system_prompt = _optional_text(payload.get("system_prompt") or payload.get("systemPrompt"))
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

        payload_api_key = ""
        if allow_payload_api_key:
            payload_api_key = _payload_api_key(payload)
        api_key = payload_api_key or self.default_api_key
        if not api_key and self.model_key_pool is None:
            raise ValueError("model api_key is required.")

        temperature = _float(payload.get("temperature"), 0.2)
        extra_body = _extra_body(payload)

        if payload_api_key or self.model_key_pool is None:
            return await self._chat_with_api_key(
                api_key=api_key,
                team_id=team_id,
                messages=messages,
                temperature=temperature,
                extra_body=extra_body,
            )
        return await self.model_key_pool.run_async(
            operation=lambda pool_api_key: self._chat_with_api_key(
                api_key=pool_api_key,
                team_id=team_id,
                messages=messages,
                temperature=temperature,
                extra_body=extra_body,
            ),
            usage_extractor=lambda result: result.get("usage", {}) if isinstance(result, dict) else {},
        )

    async def _chat_with_api_key(
        self,
        *,
        api_key: str,
        team_id: str,
        messages: list[dict[str, Any]],
        temperature: float,
        extra_body: dict[str, Any] | None,
    ) -> dict[str, Any]:
        client = OpenAICompatibleChatClient(
            model_api=self.model_api,
            api_key=api_key,
            model=self.model_name,
            timeout_seconds=self.timeout_seconds,
            retries=self.request_retries,
        )
        result = await client.chat(
            messages=messages,
            temperature=temperature,
            extra_body=extra_body,
        )
        usage_record = self._record_usage(
            team_id=team_id,
            messages=messages,
            answer=result.content,
            llm_usage=result.usage,
        )
        return {
            "content": result.content,
            "usage": result.usage,
            "usage_record": usage_record,
            "model": self.model_name,
        }

    def _record_usage(
        self,
        *,
        team_id: str,
        messages: list[dict[str, str]],
        answer: str,
        llm_usage: dict[str, int],
    ) -> dict[str, int]:
        input_tokens = int(llm_usage.get("prompt_tokens") or estimate_tokens(messages))
        output_tokens = int(llm_usage.get("completion_tokens") or estimate_tokens(answer))
        total_tokens = int(llm_usage.get("total_tokens") or input_tokens + output_tokens)
        return self.usage_store.increment_usage(
            dataset_key=self.dataset_key,
            team_id=team_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
        )


def _messages(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("messages must be a JSON array.")
    messages: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError(f"messages[{index}] must be a JSON object.")
        role = _optional_text(item.get("role"))
        content = item.get("content")
        if isinstance(content, list):
            # Keep cache text blocks intact through held-out doctor calls.
            if not content or any(not isinstance(block, dict) or block.get("type") != "text"
                                  or not isinstance(block.get("text"), str) or not block["text"].strip()
                                  or set(block) - {"type", "text", "cache_control"}
                                  or ("cache_control" in block and block["cache_control"] != {"type": "ephemeral"})
                                  for block in content):
                raise ValueError(f"messages[{index}].content must contain valid text blocks.")
            content = [dict(block) for block in content]
        else:
            content = _optional_text(content)
        if not role:
            raise ValueError(f"messages[{index}].role is required.")
        if not content:
            raise ValueError(f"messages[{index}].content is required.")
        messages.append({"role": role, "content": content})
    return messages


def _payload_api_key(payload: dict[str, Any]) -> str:
    for key in ("api_key", "apiKey", "model_api_key", "modelApiKey", "model_key", "modelKey"):
        value = _optional_text(payload.get(key))
        if value:
            return value
    return ""


def _extra_body(payload: dict[str, Any]) -> dict[str, Any] | None:
    value = payload.get("extra_body")
    if value is None:
        value = payload.get("extraBody")
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("extra_body must be a JSON object when provided.")
    blocked = sorted(set(value) & _CONTROLLED_EXTRA_BODY_KEYS)
    if blocked:
        raise ValueError("extra_body must not override controlled field(s): %s" % ", ".join(blocked))
    return dict(value)


def _optional_text(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _float(value: Any, default: float) -> float:
    if value in (None, ""):
        return default
    return float(value)


_CONTROLLED_EXTRA_BODY_KEYS = {"messages", "model", "stream"}
