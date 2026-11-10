"""Strict LLM generation of missing patient examination results."""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Callable, Literal

try:
    from ..llm_client import OpenAICompatibleChatClient
except ImportError:
    from llm_client import OpenAICompatibleChatClient

from .catalog import ExaminationIndicator

SYSTEM_PROMPT = """你负责为患者生成一项缺失检查的完整结果。请严格遵循以下规则：

1. 信息源限制
   只能使用 `patient_profile` 和 `known_examination_results` 中的患者事实，不得引入外部知识或假设。

2. 指标完整性
   必须返回 `required_indicators` 中列出的全部指标，每个指标恰好出现一次，不得改名、遗漏或增补。

3. 结果范围限定
   每个指标的 `value` 必须严格限于该指标自身检查所覆盖的器官、部位或系统。
   - 可依据相关症状、病史和诊断作为推断背景，但最终结论只能描述当前指标范围内的所见。
   - 不得写入其他指标、器官、部位或系统的表现，例如：描述外阴时不得描述上唇、胸部、下腹部等。
   - 若无法根据现有信息形成可靠判断，则按 `normal_reference` 生成正常结果。

4. 结果表述要求
   - `value` 必须为非空、明确且确定的表述。
   - 定量指标须给出合理的数值及单位，且同一检查内各数值之间须自洽。
   - 不得包含证据原文、推理过程或病因分析。
   - 禁止使用模糊词汇（如“可能”“疑似”“不排除”“如果……则……”），也禁止编造“患者主诉”。

5. 状态判定
   - `status` 只能为 `"normal"` 或 `"abnormal"`，且须与 `value` 一致。
   - 当 `value` 与 `normal_reference` 含义相同或相似，或数值处于参考范围内时，必须判为 `"normal"`；不得仅凭孤立词语主观判定异常。

输出格式
仅返回如下结构的 JSON，不得附加任何解释或额外文本：
{
  "examination": "检查名称",
  "indicators": [
    {"name": "指标名称", "value": "非空检查值", "status": "normal 或 abnormal"}
  ]
}
"""

AMBIGUOUS_VALUE_PATTERN = re.compile(
    r"(?:可能|或许|也许|疑似|不排除|通常|倾向于|一般(?:可|会)|"
    r"如果.+?(?:则|就)|若.+?(?:则|就)|视.+而定)"
)


@dataclass(frozen=True)
class _GeneratedIndicator:
    value: Any
    status: Literal["normal", "abnormal"]


@dataclass(frozen=True)
class GeneratedExaminationResult:
    result: dict[str, Any]
    indicator_statuses: dict[str, Literal["normal", "abnormal"]]
    status: Literal["normal", "abnormal"]
    abnormal_indicators: tuple[str, ...]
    model_name: str
    usage: dict[str, int]


@dataclass(frozen=True)
class ExaminationModelChannel:
    model_api: str
    api_key: str
    model_name: str
    timeout_seconds: int
    request_retries: int


class ExaminationResultGenerator:
    def __init__(
        self,
        *,
        model_api: str = "",
        model_names: tuple[str, ...] = (),
        timeout_seconds: int = 120,
        request_retries: int = 2,
        api_key: str | None = None,
        model_channels: tuple[ExaminationModelChannel, ...] | None = None,
        client_factory: Callable[..., Any] = OpenAICompatibleChatClient,
    ):
        legacy_model_names = tuple(dict.fromkeys(
            name
            for value in model_names
            if (name := str(value).strip())
        ))
        legacy_channels = tuple(
            ExaminationModelChannel(
                model_api=str(model_api).strip(),
                api_key=str(api_key or "").strip(),
                model_name=model_name,
                timeout_seconds=int(timeout_seconds),
                request_retries=max(1, int(request_retries)),
            )
            for model_name in legacy_model_names
        )
        self.model_channels = tuple(
            ExaminationModelChannel(
                model_api=str(channel.model_api or "").strip(),
                api_key=str(channel.api_key or "").strip(),
                model_name=str(channel.model_name).strip(),
                timeout_seconds=int(channel.timeout_seconds),
                request_retries=max(1, int(channel.request_retries)),
            )
            for channel in (model_channels or legacy_channels)
            if str(channel.model_name).strip()
        )
        self.client_factory = client_factory

    async def generate(
        self,
        *,
        examination: str,
        indicators: tuple[ExaminationIndicator, ...],
        profile: dict[str, Any],
        known_results: dict[str, Any],
        normal_reference: Any | None = None,
    ) -> GeneratedExaminationResult:
        if not indicators:
            raise ValueError("At least one examination indicator is required")

        if not self.model_channels:
            raise LookupError("no examination generation model configured")
        last_error: Exception | None = None
        for channel in self.model_channels:
            if not channel.api_key:
                last_error = LookupError(
                    f"no examination generation API key configured for model {channel.model_name}"
                )
                continue
            try:
                return await self._generate_with_model(
                    channel=channel,
                    examination=examination,
                    indicators=indicators,
                    profile=profile,
                    known_results=known_results,
                    normal_reference=normal_reference,
                )
            except Exception as error:  # Try the next configured model before failing the request.
                last_error = error
        if last_error is not None:
            raise RuntimeError(
                f"No configured examination generation model succeeded: {last_error}"
            ) from last_error
        raise LookupError("no examination generation model configured")

    async def _generate_with_model(
        self,
        *,
        channel: ExaminationModelChannel,
        examination: str,
        indicators: tuple[ExaminationIndicator, ...],
        profile: dict[str, Any],
        known_results: dict[str, Any],
        normal_reference: Any | None,
    ) -> GeneratedExaminationResult:
        model_name = channel.model_name
        client = self.client_factory(
            model_api=channel.model_api,
            api_key=channel.api_key,
            model=model_name,
            timeout_seconds=channel.timeout_seconds,
            retries=channel.request_retries,
        )
        indicator_requests = _indicator_requests(indicators, normal_reference)
        request_payload = {
            "patient_profile": profile,
            "known_examination_results": known_results,
            "requested_examination": examination,
            "required_indicators": indicator_requests,
        }
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(request_payload, ensure_ascii=False)},
        ]
        expected_names = tuple(indicator.name for indicator in indicators)
        requests_by_name = {row["name"]: row for row in indicator_requests}
        accepted: dict[str, _GeneratedIndicator] = {}
        requested_names = expected_names
        last_error: ValueError | None = None
        for attempt in range(channel.request_retries):
            response = await client.chat(
                messages=messages,
                temperature=0.1,
                extra_body={"response_format": {"type": "json_object"}},
            )
            try:
                parsed = parse_generated_result(
                    raw=response.content,
                    examination=examination,
                    expected_names=expected_names,
                    model_name=model_name,
                    usage=response.usage,
                )
                return _finalize_generated_result(parsed)
            except ValueError as error:
                last_error = error
            accepted.update(
                _extract_valid_indicator_rows(response.content, frozenset(requested_names))
            )
            remaining = tuple(name for name in expected_names if name not in accepted)
            if not remaining:
                parsed = _build_generated_result(
                    examination=examination,
                    expected_names=expected_names,
                    rows=accepted,
                    model_name=model_name,
                    usage=response.usage,
                )
                return _finalize_generated_result(parsed)
            if attempt + 1 >= channel.request_retries:
                break

            requested_names = remaining
            messages.append({"role": "assistant", "content": response.content})
            messages.append({
                "role": "user",
                "content": json.dumps({
                    "task": "修复无效指标",
                    "validation_error": str(last_error),
                    "instruction": (
                        "只返回 required_indicators 中的指标，name 必须完全一致，value 必须非空。"
                        "value 只能描述当前指标范围内可直接观察或测量的结果，不得夹带无关内容。"
                        "没有直接异常依据时，按 normal_reference 返回正常值。"
                        "normal_reference 是通用参考描述，可能含有‘通常’‘可能’等词；"
                        "不得照抄其中的模糊措辞，必须根据患者信息改写为本次检查的确定所见。"
                        "仍须遵守最初的全部规则，不得更改指标名称或省略任何待修复指标。"
                    ),
                    "requested_examination": examination,
                    "required_indicators": [requests_by_name[name] for name in remaining],
                }, ensure_ascii=False),
            })

        missing = ", ".join(name for name in expected_names if name not in accepted)
        detail = f"missing or invalid indicators: {missing}"
        if last_error is not None:
            detail += f"; last validation error: {last_error}"
        raise ValueError(f"Invalid examination model output after retries: {detail}") from last_error


def _canonical_indicator_name(name: str, expected_names) -> str:
    """Accept presentation-only spacing/bracket changes when unambiguous.

    Preserve specimen, time, posture, units and all other clinical characters.
    Unknown names and colliding catalog entries remain subject to strict checks.
    """
    if name in expected_names:
        return name
    def key(text):
        return "".join(text.split()).replace("（", "(").replace("）", ")")
    matches = [expected for expected in expected_names if key(expected) == key(name)]
    return matches[0] if len(matches) == 1 else name


def parse_generated_result(
    *,
    raw: str,
    examination: str,
    expected_names: tuple[str, ...],
    model_name: str,
    usage: dict[str, int] | None = None,
) -> GeneratedExaminationResult:
    try:
        payload = json.loads(str(raw).strip())
    except json.JSONDecodeError as error:
        raise ValueError(f"response is not valid JSON: {error.msg}") from error
    if not isinstance(payload, dict):
        raise ValueError("response must be a JSON object")
    unexpected_payload_fields = sorted(set(payload) - {"examination", "indicators"})
    if unexpected_payload_fields:
        raise ValueError(
            f"unexpected response fields: {', '.join(unexpected_payload_fields)}"
        )
    if str(payload.get("examination") or "").strip() != examination:
        raise ValueError(f"examination must be {examination!r}")
    rows = payload.get("indicators")
    if not isinstance(rows, list):
        raise ValueError("indicators must be a JSON array")

    by_name: dict[str, _GeneratedIndicator] = {}
    duplicate_names: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"indicators[{index}] must be an object")
        unexpected_fields = sorted(set(row) - {"name", "value", "status"})
        if unexpected_fields:
            raise ValueError(
                f"unexpected fields for indicators[{index}]: {', '.join(unexpected_fields)}"
            )
        name = str(row.get("name") or "").strip()
        if not name:
            raise ValueError(f"indicators[{index}].name is required")
        name = _canonical_indicator_name(name, expected_names)
        if name in by_name:
            duplicate_names.add(name)
            continue
        value = row.get("value")
        if _is_blank_value(value):
            raise ValueError(f"indicator {name!r} has a blank value")
        if _is_ambiguous_value(value):
            raise ValueError(f"indicator {name!r} has an ambiguous value")
        status = str(row.get("status") or "").strip()
        if status not in {"normal", "abnormal"}:
            raise ValueError(f"indicator {name!r} has invalid status {status!r}")
        by_name[name] = _GeneratedIndicator(
            value=value,
            status=status,  # type: ignore[arg-type]
        )
    if duplicate_names:
        raise ValueError(f"duplicate indicators: {', '.join(sorted(duplicate_names))}")

    expected = set(expected_names)
    actual = set(by_name)
    missing = [name for name in expected_names if name not in actual]
    additional = sorted(actual - expected)
    errors = []
    if missing:
        errors.append(f"missing indicators: {', '.join(missing)}")
    if additional:
        errors.append(f"additional indicators: {', '.join(additional)}")
    if errors:
        raise ValueError("; ".join(errors))

    result: dict[str, Any] = {}
    statuses: dict[str, Literal["normal", "abnormal"]] = {}
    abnormal: list[str] = []
    for name in expected_names:
        indicator = by_name[name]
        result[name] = indicator.value
        statuses[name] = indicator.status
        if indicator.status == "abnormal":
            abnormal.append(name)
    return GeneratedExaminationResult(
        result=result,
        indicator_statuses=statuses,
        status="abnormal" if abnormal else "normal",
        abnormal_indicators=tuple(abnormal),
        model_name=model_name,
        usage=dict(usage or {}),
    )


def _indicator_requests(
    indicators: tuple[ExaminationIndicator, ...],
    normal_reference: Any | None,
) -> list[dict[str, Any]]:
    reference_values = normal_reference if isinstance(normal_reference, dict) else {}
    return [
        {
            "name": indicator.name,
            "normal_reference": reference_values.get(indicator.name),
            "abnormal_case_only": indicator.abnormal_case_only,
        }
        for indicator in indicators
    ]


def _extract_valid_indicator_rows(
    raw: str,
    allowed_names: frozenset[str],
) -> dict[str, _GeneratedIndicator]:
    try:
        payload = json.loads(str(raw).strip())
    except (json.JSONDecodeError, TypeError):
        return {}
    if not isinstance(payload, dict) or not isinstance(payload.get("indicators"), list):
        return {}

    rows: dict[str, _GeneratedIndicator] = {}
    duplicates: set[str] = set()
    for row in payload["indicators"]:
        if not isinstance(row, dict) or set(row) - {"name", "value", "status"}:
            continue
        name = str(row.get("name") or "").strip()
        name = _canonical_indicator_name(name, allowed_names)
        status = str(row.get("status") or "").strip()
        value = row.get("value")
        if (
            name not in allowed_names
            or _is_blank_value(value)
            or _is_ambiguous_value(value)
            or status not in {"normal", "abnormal"}
        ):
            continue
        if name in rows:
            duplicates.add(name)
            continue
        rows[name] = _GeneratedIndicator(
            value=value,
            status=status,  # type: ignore[arg-type]
        )
    for name in duplicates:
        rows.pop(name, None)
    return rows


def _build_generated_result(
    *,
    examination: str,
    expected_names: tuple[str, ...],
    rows: dict[str, _GeneratedIndicator],
    model_name: str,
    usage: dict[str, int] | None,
) -> GeneratedExaminationResult:
    payload = {
        "examination": examination,
        "indicators": [
            {
                "name": name,
                "value": rows[name].value,
                "status": rows[name].status,
            }
            for name in expected_names
        ],
    }
    return parse_generated_result(
        raw=json.dumps(payload, ensure_ascii=False),
        examination=examination,
        expected_names=expected_names,
        model_name=model_name,
        usage=usage,
    )


def _finalize_generated_result(
    generated: GeneratedExaminationResult,
) -> GeneratedExaminationResult:
    values = dict(generated.result)
    _cohere_derived_measurements(values)
    return GeneratedExaminationResult(
        result=values,
        indicator_statuses=generated.indicator_statuses,
        status=generated.status,
        abnormal_indicators=generated.abnormal_indicators,
        model_name=generated.model_name,
        usage=generated.usage,
    )


def _cohere_derived_measurements(values: dict[str, Any]) -> None:
    height_cm = _single_measurement(values.get("身高"), "cm")
    weight_kg = _single_measurement(values.get("体重"), "kg")
    if height_cm and weight_kg and "体重指数" in values:
        values["体重指数"] = f"{weight_kg / ((height_cm / 100) ** 2):.1f} kg/m²"

    waist_cm = _single_measurement(values.get("腰围"), "cm")
    hip_cm = _single_measurement(values.get("臀围"), "cm")
    if waist_cm and hip_cm and "腰臀比" in values:
        values["腰臀比"] = f"{waist_cm / hip_cm:.2f}"


def _single_measurement(value: Any, unit: str) -> float | None:
    text = _normalize_text(value)
    if unit.lower() not in text.lower():
        return None
    numbers = re.findall(r"[-+]?\d+(?:\.\d+)?", text)
    if len(numbers) != 1:
        return None
    measured = float(numbers[0])
    return measured if measured > 0 else None


def _normalize_text(value: Any) -> str:
    return unicodedata.normalize("NFKC", str(value if value is not None else "")).strip()


def _is_ambiguous_value(value: Any) -> bool:
    return isinstance(value, str) and AMBIGUOUS_VALUE_PATTERN.search(value) is not None


def _is_blank_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, dict, tuple, set)):
        return not value
    return False
