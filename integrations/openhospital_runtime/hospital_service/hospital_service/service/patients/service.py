"""Patient dialogue service backed by the patient-agent prompt pipeline."""

from __future__ import annotations

import json
import re
from typing import Any

try:
    from ..llm_client import OpenAICompatibleChatClient
    from ..repository import ServiceDataRepository
    from ..token_usage import InMemoryTokenUsageStore, TokenUsageStore
except ImportError:
    from llm_client import OpenAICompatibleChatClient
    from repository import ServiceDataRepository
    from token_usage import InMemoryTokenUsageStore, TokenUsageStore

from .patient_data import PatientRecord, build_patient_record
from .prompt_builder import (
    PATIENT_REPLY_SYSTEM_PROMPT,
    RETRIEVAL_PLAN_SYSTEM_PROMPT,
    PatientPromptBuilder,
)


class PatientDialogueService:
    """Patient-agent compatible service for one dataset."""

    def __init__(
        self,
        repository: ServiceDataRepository,
        usage_store: TokenUsageStore | None = None,
        model_key_pool: Any | None = None,
        *,
        model_api: str | None = None,
        model_name: str | None = None,
        timeout_seconds: int | None = None,
        request_retries: int | None = None,
        default_api_key: str | None = None,
    ):
        self.repository = repository
        self.prompt_builder = PatientPromptBuilder()
        self.model_api = model_api or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        self.model_name = model_name or "qwen3.5-flash"
        self.timeout_seconds = timeout_seconds or 120
        self.request_retries = request_retries or 2
        self.default_api_key = str(default_api_key or "").strip()
        self.model_key_pool = model_key_pool
        self._records = {
            patient_id: build_patient_record(profile)
            for patient_id, profile in self.repository.profiles.items()
        }
        self.usage_store = usage_store or InMemoryTokenUsageStore()

    async def invoke_async(self, payload: dict[str, Any]) -> dict[str, Any]:
        team_id = _required_text(payload, "team_id")
        patient_id = _required_text(payload, "patient_id")
        payload_api_key = _payload_api_key(payload)
        api_key = payload_api_key or self.default_api_key
        if not api_key and self.model_key_pool is None:
            raise ValueError("api_key is required.")
        input_data = payload.get("input")
        if not isinstance(input_data, dict):
            raise ValueError("input must be a JSON object.")

        patient_record = self._records.get(patient_id)
        if patient_record is None:
            raise KeyError(f"Unknown patient_id: {patient_id}")

        if payload_api_key or self.model_key_pool is None:
            result = await self._invoke_with_api_key(
                api_key=api_key,
                team_id=team_id,
                patient_id=patient_id,
                input_data=input_data,
                patient_record=patient_record,
            )
            return result
        return await self.model_key_pool.run_async(
            operation=lambda pool_api_key: self._invoke_with_api_key(
                api_key=pool_api_key,
                team_id=team_id,
                patient_id=patient_id,
                input_data=input_data,
                patient_record=patient_record,
            ),
            usage_extractor=lambda result: result.get("usage_delta", {}) if isinstance(result, dict) else {},
        )

    async def _invoke_with_api_key(
        self,
        *,
        api_key: str,
        team_id: str,
        patient_id: str,
        input_data: dict[str, Any],
        patient_record: PatientRecord,
    ) -> dict[str, Any]:
        client = OpenAICompatibleChatClient(
            model_api=self.model_api,
            api_key=api_key,
            model=self.model_name,
            timeout_seconds=self.timeout_seconds,
            retries=self.request_retries,
        )
        retrieval_plan = await self._generate_retrieval_plan(
            client=client,
            patient_record=patient_record,
            user_input=input_data,
        )
        relevant_history = self.prompt_builder.select_history_entries(
            patient_record=patient_record,
            history_types=retrieval_plan.get("history_types", []) or [],
            field_names=retrieval_plan.get("field_names", []) or [],
        )
        if not relevant_history:
            relevant_history = self.prompt_builder.select_chief_complaint(patient_record)

        answer, usage = await self._generate_patient_answer(
            client=client,
            patient_record=patient_record,
            relevant_history=relevant_history,
            user_input=input_data,
        )
        usage_record = self._record_usage(
            team_id=team_id,
            input_data=input_data,
            answer=answer,
            llm_usage=usage,
        )
        return {
            "answer": answer,
            "patient_id": patient_id,
            "team_id": team_id,
            "usage": usage_record,
            "usage_delta": normalize_llm_usage_for_pool(usage, input_data, answer),
        }

    def usage(self, team_id: str | None = None) -> dict[str, Any]:
        if team_id:
            return {
                "team_id": team_id,
                "usage": self.usage_store.get_usage(
                    dataset_key=self.repository.dataset_key,
                    team_id=team_id,
                ),
            }
        return {
            "usage_by_team_id": self.usage_store.list_usage(
                dataset_key=self.repository.dataset_key,
            ),
        }

    async def _generate_retrieval_plan(
        self,
        *,
        client: OpenAICompatibleChatClient,
        patient_record: PatientRecord,
        user_input: dict[str, Any],
    ) -> dict[str, Any]:
        user_prompt = self.prompt_builder.build_retrieval_prompt(
            patient_record=patient_record,
            user_input=user_input,
        )
        result = await client.chat(
            messages=[
                {"role": "system", "content": RETRIEVAL_PLAN_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
        )
        return self.prompt_builder.parse_retrieval_plan(
            raw_response=result.content,
            patient_record=patient_record,
        )

    async def _generate_patient_answer(
        self,
        *,
        client: OpenAICompatibleChatClient,
        patient_record: PatientRecord,
        relevant_history: list[dict[str, str]],
        user_input: dict[str, Any],
    ) -> tuple[str, dict[str, int]]:
        user_prompt = self.prompt_builder.build_reply_prompt(
            patient_record=patient_record,
            user_input=user_input,
            relevant_history=relevant_history,
        )

        result = await client.chat(
            messages=[
                {"role": "system", "content": PATIENT_REPLY_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
        )
        return result.content, result.usage

    def _record_usage(
        self,
        *,
        team_id: str,
        input_data: dict[str, Any],
        answer: str,
        llm_usage: dict[str, int],
    ) -> dict[str, int]:
        input_tokens = int(llm_usage.get("prompt_tokens") or estimate_tokens(input_data))
        output_tokens = int(llm_usage.get("completion_tokens") or estimate_tokens(answer))
        total_tokens = int(llm_usage.get("total_tokens") or input_tokens + output_tokens)
        return self.usage_store.increment_usage(
            dataset_key=self.repository.dataset_key,
            team_id=team_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
        )


def _required_text(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    text = str(value).strip() if value is not None else ""
    if not text:
        raise ValueError(f"{key} is required.")
    return text


def _optional_text(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _payload_api_key(payload: dict[str, Any]) -> str:
    for key in ("api_key", "apiKey", "model_api_key", "modelApiKey", "model_key", "modelKey"):
        value = _optional_text(payload.get(key))
        if value:
            return value
    return ""


def estimate_tokens(value: Any) -> int:
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True)
    text = value.strip()
    if not text:
        return 0
    ascii_words = re.findall(r"[A-Za-z0-9_]+", text)
    non_ascii_chars = re.findall(r"[^\x00-\x7F\s]", text)
    punctuation_chunks = re.findall(r"[^\w\s]", text, flags=re.ASCII)
    return max(1, len(ascii_words) + len(non_ascii_chars) + len(punctuation_chunks))


def normalize_llm_usage_for_pool(usage: dict[str, int], input_data: Any, answer: Any) -> dict[str, int]:
    input_tokens = int(usage.get("prompt_tokens") or estimate_tokens(input_data))
    output_tokens = int(usage.get("completion_tokens") or estimate_tokens(answer))
    total_tokens = int(usage.get("total_tokens") or input_tokens + output_tokens)
    return {
        "prompt_tokens": input_tokens,
        "completion_tokens": output_tokens,
        "total_tokens": total_tokens,
    }
