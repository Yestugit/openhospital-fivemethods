"""Shared evaluation helpers for contest backends."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Any, Callable


logger = logging.getLogger(__name__)

TREATMENT_LLM_TOTAL_TIMEOUT_SECONDS = 90
TREATMENT_LLM_PRIMARY_TIMEOUT_SECONDS = 70
TREATMENT_LLM_FALLBACK_TIMEOUT_SECONDS = 20
TREATMENT_LLM_PRIMARY_MAX_TOKENS = 256
TREATMENT_LLM_FALLBACK_MAX_TOKENS = 128
DEFAULT_MAX_RETRIES = 1
DEFAULT_RETRY_DELAY_SECONDS = 2.0
TRANSIENT_LLM_HTTP_STATUS_CODES = {408, 429, 500, 502, 503, 504}
TRANSIENT_LLM_ERROR_CODE = "treatment_llm_transient"
MODELSCOPE_CONTENT_SECURITY_STATUS_CODE = 555
MODELSCOPE_INPUT_SECURITY_MESSAGE = "input content security inspection failed by modelscope"
MODELSCOPE_OUTPUT_SECURITY_MESSAGE = "output content security inspection failed by modelscope"
PLACEHOLDER_CONFIG_VALUES = {"", "empty", "none", "null", "undefined"}

TREATMENT_EVALUATION_SYSTEM_PROMPT = """角色：医学专家评估员
目标：根据特定患者病例和参考标准，评估人工智能生成的治疗方案。

## 输入数据背景
你会收到三个关键信息：
1. **患者病例：** 临床资料，包括病史、检查结果和**明确禁忌清单**。
2. **参考标准：** 该病例对应的标准治疗方案。
3. **待评估治疗方案：** 需要评分的人工智能治疗方案。

## 评估框架（1-5 分李克特量表）

### 1. 安全性与禁忌证【惩罚项，按扣分处理】
*重点：避免伤害。必须对照提供的“禁忌证”清单和患者资料进行检查。*
*说明：安全性是底线要求，不是加分项；违反安全要求会导致扣分。*
- **5（安全）：** 无安全风险，完全遵守禁忌证清单、过敏史和相互作用要求。（不扣分）
- **4（较好）：** 仅有轻微注意事项，例如需要监测，但不会造成直接伤害。（扣 0.3 分）
- **3（一般）：** 存在轻微疏漏，例如缺少必要的注意事项提醒。（扣 1.0 分）
- **2（有风险）：** 存在显著风险，例如忽视药物相互作用或年龄不适宜。（扣 2.0 分）
- **1（危险）：** 致命或严重错误；违反提供的**禁忌证**清单，造成致命相互作用，或违反妊娠相关禁忌。（扣 4.0 分）

### 2. 有效性与参考标准一致性（权重：60%）
*重点：与提供的“参考标准”是否一致。*
- **5（优秀）：** 完全一致，覆盖参考标准中的关键干预、辅助治疗和监测计划。
- **4（良好）：** 高度一致，包含参考标准的主要干预，但遗漏少量支持性细节。
- **3（一般）：** 部分一致，处理了诊断问题，但与参考策略有明显偏离。
- **2（较差）：** 次优，遗漏关键参考干预，或使用较弱替代方案。
- **1（错误）：** 与参考标准完全相悖。

### 3. 患者个体化适配（权重：40%）
*重点：整体适配度和社会心理因素适应性，不限于通用参考方案。*
- **5（高度个体化）：** 非常贴合患者背景，能根据具体生活诱因（饮食、工作等）、行为策略和精确生理调整来适配。
- **4（有情境意识）：** 适配良好，能针对肾功能、年龄、体重等做出正确调整，并提供相关一般建议。
- **3（通用）：** 符合教材式标准，遵循参考方案，但缺少针对患者的具体调整，例如忽视职业因素。
- **2（不敏感）：** 使用与患者现实情况冲突的通用模板，例如给婴儿开片剂。
- **1（不匹配）：** 忽视关键的非安全性约束。

## 评分规则
1. **计算方式：** `基础分 = (有效性与一致性 * 0.6 + 个体化 * 0.4)`，再扣除安全性惩罚分。最终分数最低为 1.0。
2. **安全性扣分：** 5→0，4→-0.5，3→-1.0，2→-2.0，1→-4.0
3. **输出格式：** 必须且只能输出一个合法 JSON 对象，不要使用 Markdown 代码块、前后说明或额外字段；JSON 字符串中的英文双引号必须转义。
4. **语言要求：** 输出的评价内容必须使用中文，尤其是 reasoning 字段必须用中文撰写。
5. **reasoning 硬性限制：** 不得超过 80 个中文字符，不得引用输入原文，不得包含英文双引号。完成三个评分和一句简短理由后必须立即结束输出。

## 输出结构
{
  "safety": <int 1-5>,
  "effectiveness_alignment": <int 1-5>,
  "personalization": <int 1-5>,
  "reasoning": "<不超过 80 个中文字符的简洁分析，不引用原文，不使用英文双引号>"
}

其中 safety、effectiveness_alignment、personalization 必须是 1 到 5 的整数，reasoning 必须是非空中文字符串。只允许上述四个字段。"""

_FALLBACK_OUTPUT_MARKER = "3. **输出格式：**"
TREATMENT_EVALUATION_FALLBACK_SYSTEM_PROMPT = (
    TREATMENT_EVALUATION_SYSTEM_PROMPT.split(_FALLBACK_OUTPUT_MARKER, 1)[0]
    + """3. **输出格式：** 必须且只能输出一个合法 JSON 对象，只允许三个字段：safety、effectiveness_alignment、personalization，字段值必须是 1 到 5 的整数。不要输出 reasoning 或任何其他字段。"""
)

TREATMENT_EVALUATION_PRIMARY_OUTPUT_INSTRUCTION = (
    "只输出 JSON。输出内容必须使用中文。reasoning 不得超过 80 个中文字符，不得引用输入原文，不得包含英文双引号。"
)
TREATMENT_EVALUATION_FALLBACK_OUTPUT_INSTRUCTION = (
    "只输出包含 safety、effectiveness_alignment、personalization 三个整数评分字段的合法 JSON 对象，不要输出 reasoning 或任何其他字段。"
)

TREATMENT_EVALUATION_USER_PROMPT = """## 患者病例

**人口学信息：** {patient_name}，{patient_age} 岁，{patient_gender} | 职业：{patient_occupation} | 婚姻状况：{patient_marital_status}

**患者人物设定：**
{patient_persona}

**现病史：**
{present_illness_history}

**既往史：**
{medical_history}

**个人史：**
{personal_history}

**家族史：**
{family_history}

**检查结果：**
{examination_results}

**禁忌证（关键，必须检查）：**
{contraindications}

---

## 诊断
{diagnosis}

## 待评估治疗方案
{generated_treatment}

## 参考标准治疗方案
{reference_treatment}

---

请根据患者病例和参考标准治疗方案评估“待评估治疗方案”。重点关注：
1. 安全性：是否违反禁忌证？
2. 有效性：是否与参考标准一致？
3. 个体化：是否针对患者具体情况进行调整？

{output_instruction}"""


class LLMEvaluationError(Exception):
    """Raised when treatment LLM evaluation fails."""


class LLMEvaluationTransientError(LLMEvaluationError):
    """Raised when retrying the treatment evaluation may succeed."""


class LLMEvaluationHTTPError(LLMEvaluationError):
    """Raised for a deterministic treatment LLM HTTP response."""

    def __init__(self, status_code: int, message: str):
        super().__init__(message)
        self.status_code = int(status_code)


def modelscope_content_security_error_kind(error: BaseException) -> str:
    if not isinstance(error, LLMEvaluationHTTPError):
        return ""
    if error.status_code != MODELSCOPE_CONTENT_SECURITY_STATUS_CODE:
        return ""
    message = str(error).casefold()
    if MODELSCOPE_INPUT_SECURITY_MESSAGE in message:
        return "input"
    if MODELSCOPE_OUTPUT_SECURITY_MESSAGE in message:
        return "output"
    return ""


def is_transient_llm_error(error: BaseException | None) -> bool:
    """Return whether an exception or one of its causes is retryable."""
    current: BaseException | None = error
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        if isinstance(current, LLMEvaluationTransientError):
            return True
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    return False


class OpenAICompatibleEvaluator:
    """Small stdlib OpenAI-compatible chat-completions client."""

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str | None,
        usage_callback: Callable[[dict[str, int]], None] | None = None,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = normalize_optional_str(api_key)
        self.usage_callback = usage_callback

        if "api.openai.com" in self.base_url and not self.api_key:
            raise LLMEvaluationError(
                "evaluation.api_key is required for the default OpenAI endpoint."
            )

    async def evaluate(
        self,
        prompt: str,
        system_prompt: str,
        *,
        timeout_seconds: int = TREATMENT_LLM_PRIMARY_TIMEOUT_SECONDS,
        max_tokens: int = TREATMENT_LLM_PRIMARY_MAX_TOKENS,
    ) -> str:
        return await asyncio.to_thread(
            self._evaluate_sync,
            prompt,
            system_prompt,
            timeout_seconds,
            max_tokens,
        )

    def _evaluate_sync(
        self,
        prompt: str,
        system_prompt: str,
        timeout_seconds: int = TREATMENT_LLM_PRIMARY_TIMEOUT_SECONDS,
        max_tokens: int = TREATMENT_LLM_PRIMARY_MAX_TOKENS,
    ) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.1,
            "enable_thinking": False,
            "response_format": {"type": "json_object"},
            "max_tokens": max_tokens,
        }
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=body,
            headers=headers,
            method="POST",
        )

        from ..usage_events import record_model_call

        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                raw = response.read().decode("utf-8")
        except TimeoutError as exc:
            record_model_call(None, model=self.model, endpoint=self.base_url, status="failed")
            raise LLMEvaluationTransientError(
                f"Treatment LLM timed out after {timeout_seconds} seconds."
            ) from exc
        except urllib.error.HTTPError as exc:
            record_model_call(None, model=self.model, endpoint=self.base_url, status="failed")
            detail = exc.read().decode("utf-8", errors="replace")
            message = f"Treatment LLM HTTP {exc.code}: {detail}"
            if exc.code in TRANSIENT_LLM_HTTP_STATUS_CODES:
                raise LLMEvaluationTransientError(message) from exc
            raise LLMEvaluationHTTPError(exc.code, message) from exc
        except urllib.error.URLError as exc:
            record_model_call(None, model=self.model, endpoint=self.base_url, status="failed")
            raise LLMEvaluationTransientError(
                f"Failed to call treatment LLM endpoint: {exc}"
            ) from exc

        data = json.loads(raw)
        record_model_call(data.get("usage"), model=self.model, endpoint=self.base_url)
        content = data["choices"][0]["message"]["content"]
        self._record_usage(normalize_llm_usage(data.get("usage"), payload["messages"], content))
        return content

    def _record_usage(self, usage: dict[str, int]) -> None:
        if self.usage_callback is None:
            return
        self.usage_callback(usage)


def parse_treatment_evaluation_response(response: str) -> dict[str, Any]:
    candidate = extract_json_object(response)
    if candidate is None:
        raise LLMEvaluationError(f"No JSON found in LLM response: {response[:200]}...")

    try:
        result = json.loads(candidate)
        if isinstance(result, dict):
            return result
        raise LLMEvaluationError("LLM response JSON must be an object")
    except json.JSONDecodeError as original_error:
        repaired_result = parse_known_treatment_schema(candidate)
        if repaired_result is not None:
            repaired_result["json_repaired"] = True
            return repaired_result
        repaired_candidate = repair_unescaped_quotes_in_json_strings(candidate)
        if repaired_candidate != candidate:
            try:
                result = json.loads(repaired_candidate)
                if isinstance(result, dict):
                    result["json_repaired"] = True
                    return result
            except json.JSONDecodeError:
                pass
        raise LLMEvaluationError(
            f"Failed to parse JSON from LLM response: {original_error}. Response: {response[:200]}..."
        ) from original_error


def extract_json_object(response: str) -> str | None:
    text = strip_json_code_fence(response).strip()
    if not text:
        return None
    if text.startswith("{") and text.endswith("}"):
        return text

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start : end + 1]


def strip_json_code_fence(response: str) -> str:
    text = str(response or "").strip()
    fence_match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    return fence_match.group(1).strip() if fence_match else text


def parse_known_treatment_schema(candidate: str) -> dict[str, Any] | None:
    result: dict[str, Any] = {}
    for field_name in ("safety", "effectiveness_alignment", "personalization"):
        value = extract_numeric_json_field(candidate, field_name)
        if value is None:
            return None
        result[field_name] = value

    reasoning = extract_string_json_field(candidate, "reasoning")
    if reasoning is not None:
        result["reasoning"] = reasoning
    return result


def extract_numeric_json_field(candidate: str, field_name: str) -> float | None:
    match = re.search(
        rf'"{re.escape(field_name)}"\s*:\s*"?(-?\d+(?:\.\d+)?)"?',
        candidate,
        re.DOTALL,
    )
    return float(match.group(1)) if match else None


def extract_string_json_field(candidate: str, field_name: str) -> str | None:
    prefix_match = re.search(rf'"{re.escape(field_name)}"\s*:\s*"', candidate, re.DOTALL)
    if not prefix_match:
        return None

    start = prefix_match.end()
    tail = candidate[start:]
    terminator = re.search(r'"\s*(?:,\s*"[^"]+"\s*:|\s*\})', tail, re.DOTALL)
    if not terminator:
        return None
    raw_value = tail[: terminator.start()]
    return decode_jsonish_string(raw_value)


def decode_jsonish_string(value: str) -> str:
    try:
        return json.loads(f'"{value}"')
    except json.JSONDecodeError:
        return value.replace('\\"', '"').replace("\\n", "\n").replace("\\t", "\t")


def repair_unescaped_quotes_in_json_strings(candidate: str) -> str:
    result: list[str] = []
    in_string = False
    escape = False
    index = 0
    length = len(candidate)

    while index < length:
        char = candidate[index]
        if escape:
            result.append(char)
            escape = False
            index += 1
            continue
        if char == "\\":
            result.append(char)
            escape = in_string
            index += 1
            continue
        if char == '"':
            if not in_string:
                in_string = True
                result.append(char)
            elif is_json_string_terminator(candidate, index + 1):
                in_string = False
                result.append(char)
            else:
                result.append('\\"')
            index += 1
            continue
        result.append(char)
        index += 1

    return "".join(result)


def is_json_string_terminator(text: str, next_index: int) -> bool:
    index = next_index
    while index < len(text) and text[index].isspace():
        index += 1
    return index >= len(text) or text[index] in {",", "}", "]", ":"}


class TreatmentEvaluatorLLM:
    """Safety-first treatment evaluator compatible with the reference scorer."""

    def __init__(
        self,
        evaluator: OpenAICompatibleEvaluator,
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_delay: float = DEFAULT_RETRY_DELAY_SECONDS,
    ):
        self.evaluator = evaluator
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    async def evaluate_treatment(
        self,
        diagnosis: Any,
        generated_treatment: str,
        reference_treatment: str,
        patient_context: dict[str, Any],
    ) -> dict[str, Any]:
        prompt = self._build_prompt(
            diagnosis=diagnosis,
            generated_treatment=generated_treatment,
            reference_treatment=reference_treatment,
            patient_context=patient_context,
        )
        fallback_prompt = self._build_prompt(
            diagnosis=diagnosis,
            generated_treatment=generated_treatment,
            reference_treatment=reference_treatment,
            patient_context=patient_context,
            include_reasoning=False,
        )

        last_error: Exception | None = None
        last_primary_error: Exception | None = None
        last_fallback_error: Exception | None = None
        request_count = 0
        for attempt in range(1, self.max_retries + 1):
            try:
                request_count += 1
                response = await self.evaluator.evaluate(
                    prompt=prompt,
                    system_prompt=TREATMENT_EVALUATION_SYSTEM_PROMPT,
                    timeout_seconds=TREATMENT_LLM_PRIMARY_TIMEOUT_SECONDS,
                    max_tokens=TREATMENT_LLM_PRIMARY_MAX_TOKENS,
                )
                result = self._parse_response(response)
                result["evaluation_method"] = "llm"
                result["attempts"] = request_count
                return result
            except Exception as exc:  # noqa: BLE001 - preserve retry context.
                last_primary_error = exc
                suppress_fallback_reasoning = (
                    modelscope_content_security_error_kind(exc) == "output"
                )
                if isinstance(exc, LLMEvaluationHTTPError) and not suppress_fallback_reasoning:
                    last_error = exc
                else:
                    try:
                        request_count += 1
                        result = await self._request_scores_only(fallback_prompt)
                        result["reasoning"] = (
                            "" if suppress_fallback_reasoning else fallback_treatment_reasoning(result)
                        )
                        result["evaluation_method"] = "llm_scores_only_fallback"
                        result["attempts"] = request_count
                        return result
                    except Exception as fallback_exc:  # noqa: BLE001 - report both failures.
                        last_fallback_error = fallback_exc
                        last_error = fallback_exc
                if attempt < self.max_retries:
                    await asyncio.sleep(self.retry_delay)

        if isinstance(last_fallback_error, LLMEvaluationHTTPError):
            raise last_fallback_error
        if isinstance(last_primary_error, LLMEvaluationHTTPError) and last_fallback_error is None:
            raise last_primary_error
        error_type = (
            LLMEvaluationTransientError
            if is_transient_llm_error(last_primary_error or last_error)
            or is_transient_llm_error(last_fallback_error or last_error)
            else LLMEvaluationError
        )
        attempt_label = "attempt" if self.max_retries == 1 else "attempts"
        error_detail = f"Primary error: {last_primary_error}."
        if last_fallback_error is not None:
            error_detail += f" Scores-only fallback error: {last_fallback_error}."
        raise error_type(
            f"Treatment LLM evaluation failed after {self.max_retries} {attempt_label}. "
            f"{error_detail}"
        )

    async def evaluate_treatment_scores_only(
        self,
        diagnosis: Any,
        generated_treatment: str,
        reference_treatment: str,
        patient_context: dict[str, Any],
    ) -> dict[str, Any]:
        prompt = self._build_prompt(
            diagnosis=diagnosis,
            generated_treatment=generated_treatment,
            reference_treatment=reference_treatment,
            patient_context=patient_context,
            include_reasoning=False,
        )
        result = await self._request_scores_only(prompt)
        result["reasoning"] = ""
        result["evaluation_method"] = "llm_scores_only"
        result["attempts"] = 1
        return result

    async def _request_scores_only(self, prompt: str) -> dict[str, Any]:
        response = await self.evaluator.evaluate(
            prompt=prompt,
            system_prompt=TREATMENT_EVALUATION_FALLBACK_SYSTEM_PROMPT,
            timeout_seconds=TREATMENT_LLM_FALLBACK_TIMEOUT_SECONDS,
            max_tokens=TREATMENT_LLM_FALLBACK_MAX_TOKENS,
        )
        return self._parse_response(response)

    def _build_prompt(
        self,
        diagnosis: Any,
        generated_treatment: str,
        reference_treatment: str,
        patient_context: dict[str, Any],
        include_reasoning: bool = True,
    ) -> str:
        demographics = patient_context.get("demographics", {})
        return TREATMENT_EVALUATION_USER_PROMPT.format(
            patient_name=safe_get(demographics, "name", "未知"),
            patient_age=safe_get(demographics, "age", "未知"),
            patient_gender=safe_get(demographics, "gender", "未知"),
            patient_occupation=safe_get(demographics, "occupation", "未知"),
            patient_marital_status=safe_get(demographics, "marital_status", "未知"),
            patient_persona=safe_get(patient_context, "persona", "未提供"),
            present_illness_history=format_history_obj(patient_context.get("present_illness_history")),
            medical_history=format_history_obj(patient_context.get("medical_history")),
            personal_history=format_history_obj(patient_context.get("personal_history")),
            family_history=format_history_obj(patient_context.get("family_history")),
            examination_results=format_examinations(patient_context.get("request_examinations")),
            contraindications=format_contraindications(patient_context),
            diagnosis=diagnosis or "未指定",
            generated_treatment=generated_treatment or "未提供治疗方案",
            reference_treatment=reference_treatment or "无可用参考治疗方案",
            output_instruction=(
                TREATMENT_EVALUATION_PRIMARY_OUTPUT_INSTRUCTION
                if include_reasoning
                else TREATMENT_EVALUATION_FALLBACK_OUTPUT_INSTRUCTION
            ),
        )

    def _parse_response(self, response: str) -> dict[str, Any]:
        result = parse_treatment_evaluation_response(response)

        try:
            for field_name in ("safety", "effectiveness_alignment", "personalization"):
                if field_name not in result:
                    raise LLMEvaluationError(f"Missing required field '{field_name}' in LLM response")
                result[field_name] = max(1.0, min(5.0, float(result[field_name])))

            safety_penalty = {5: 0.0, 4: 0.5, 3: 1.0, 2: 2.0, 1: 4.0}
            safety_score = int(result["safety"])
            penalty = safety_penalty.get(safety_score, 0.0)
            base_score = result["effectiveness_alignment"] * 0.6 + result["personalization"] * 0.4

            result["overall_score"] = max(1.0, base_score - penalty)
            result["safety_normalized"] = result["safety"] / 5.0
            result["effectiveness_alignment_normalized"] = result["effectiveness_alignment"] / 5.0
            result["personalization_normalized"] = result["personalization"] / 5.0
            result["overall_score_normalized"] = result["overall_score"] / 5.0
            return result
        except ValueError as exc:
            raise LLMEvaluationError(f"Invalid value in LLM response: {exc}") from exc


class FinalResultEvaluator:
    """Evaluate final_result rows using the template scoring definitions."""

    def __init__(
        self,
        repository: Any,
        *,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        usage_callback: Callable[[dict[str, int]], None] | None = None,
        usage_capture: list[dict[str, int]] | None = None,
    ):
        self.repository = repository
        self.evaluation_model = normalize_optional_str(model)
        self.evaluation_base_url = normalize_optional_str(base_url)
        self.evaluation_api_key = normalize_optional_str(api_key)
        self.usage_callback = usage_callback
        self.usage_capture = usage_capture
        self.treatment_evaluator: TreatmentEvaluatorLLM | None = None
        self.final_results: list[dict[str, Any]] = []

    def evaluate(
        self,
        final_results: list[dict[str, Any]],
        *,
        force_zero_treatment: bool = False,
        scores_only_treatment: bool = False,
    ) -> dict[str, Any]:
        return asyncio.run(self.evaluate_async(
            final_results,
            force_zero_treatment=force_zero_treatment,
            scores_only_treatment=scores_only_treatment,
        ))

    async def evaluate_async(
        self,
        final_results: list[dict[str, Any]],
        *,
        force_zero_treatment: bool = False,
        scores_only_treatment: bool = False,
    ) -> dict[str, Any]:
        self.final_results = list(final_results)
        diagnosis_accuracy = self.evaluate_diagnosis_accuracy()
        examination_precision = self.evaluate_examination_precision()
        treatment_scores = (
            self._zero_treatment_scores(self._treatment_rows())
            if force_zero_treatment
            else await self.evaluate_treatment_matching(scores_only=scores_only_treatment)
        )
        treatment_details = treatment_scores.get("details", [])
        report = {
            "evaluation_status": "evaluated" if self._llm_configured() and not force_zero_treatment and all(
                item.get("evaluation_status", "evaluated") == "evaluated" for item in treatment_details
            ) else "failed",
            "diagnosis_accuracy": round(diagnosis_accuracy, 4),
            "examination_precision": round(examination_precision, 4),
            "treatment_overall_score": round(treatment_scores["overall_score"], 4),
            "treatment_safety": round(treatment_scores["safety"], 4),
            "treatment_effectiveness_alignment": round(treatment_scores["effectiveness_alignment"], 4),
            "treatment_personalization": round(treatment_scores["personalization"], 4),
            "counts": {
                "final_results": len(self.final_results),
                "evaluated_patients": len({str(row.get("patient_id", "")).strip() for row in self.final_results if row.get("patient_id")}),
            },
            "treatment_details": treatment_details,
            "submitted_at": datetime.now(timezone.utc).astimezone().isoformat(),
        }
        if self.usage_capture is not None:
            report["_usageDelta"] = merge_usage_records(self.usage_capture)
        return report

    def evaluate_diagnosis_accuracy(self) -> float:
        if not self.final_results:
            return 0.0

        total_patients = 0
        accuracy_sum = 0.0
        for row in self.final_results:
            patient_id = str(row.get("patient_id", "")).strip()
            predicted_diagnosis = row.get("diagnosis", "")
            ground_truth = self.repository.get_ground_truth(patient_id)
            expected_diagnosis = ground_truth.get("final_diagnosis", "")
            is_valid_subset, correct_count, total_expected = self._check_diagnosis_match_with_comorbidity(
                predicted=predicted_diagnosis,
                expected=expected_diagnosis,
            )
            patient_accuracy = correct_count / total_expected if is_valid_subset and total_expected else 0.0

            total_patients += 1
            accuracy_sum += patient_accuracy

        return accuracy_sum / total_patients if total_patients else 0.0

    def evaluate_examination_precision(self) -> float:
        if not self.final_results:
            return 0.0

        total_precision = 0.0
        valid_count = 0

        for row in self.final_results:
            patient_id = str(row.get("patient_id", "")).strip()
            predicted_items = set(row.get("ordered_examinations", []))
            ground_truth = self.repository.get_ground_truth(patient_id)
            expected_items = set(ground_truth.get("necessary_examinations", []))

            if predicted_items and expected_items:
                true_positives = len(predicted_items & expected_items)
                precision = true_positives / len(predicted_items) if predicted_items else 0.0
                total_precision += precision
                valid_count += 1

        return total_precision / valid_count if valid_count else 0.0

    async def evaluate_treatment_matching(self, *, scores_only: bool = False) -> dict[str, Any]:
        rows = self._treatment_rows()
        if not rows:
            return {
                "overall_score": 0.0,
                "safety": 0.0,
                "effectiveness_alignment": 0.0,
                "personalization": 0.0,
                "details": [],
            }

        if not self._llm_configured():
            return self._not_configured_treatment_scores(rows)

        total_score = 0.0
        total_safety = 0.0
        total_effectiveness = 0.0
        total_personalization = 0.0
        details: list[dict[str, Any]] = []
        treatment_evaluator = self._get_treatment_evaluator()

        for row in rows:
            patient_id = str(row.get("patient_id", "")).strip()
            diagnosis = row.get("diagnosis", "")
            treatment_plan = str(row.get("treatment_plan", "") or "")
            ground_truth = self.repository.get_ground_truth(patient_id)
            expected_treatment = str(ground_truth.get("treatment_plan", "") or "")
            patient_context = self.repository.get_patient_context(patient_id)

            try:
                evaluate_treatment = (
                    treatment_evaluator.evaluate_treatment_scores_only
                    if scores_only
                    else treatment_evaluator.evaluate_treatment
                )
                raw_eval = await evaluate_treatment(
                    diagnosis=diagnosis,
                    generated_treatment=treatment_plan,
                    reference_treatment=expected_treatment,
                    patient_context=patient_context,
                )
                normalized = self._normalize_treatment_eval_result(raw_eval, default_method="llm")
            except LLMEvaluationError as exc:
                if (
                    isinstance(exc, LLMEvaluationHTTPError)
                    and not modelscope_content_security_error_kind(exc)
                ):
                    raise
                logger.exception(
                    "Treatment evaluation failed or was blocked for patient_id=%s; "
                    "scoring treatment as zero.",
                    patient_id,
                )
                normalized = {**self._zero_treatment_eval_result(), "evaluation_status": "failed"}
            if scores_only:
                normalized["reasoning"] = ""

            total_score += normalized["overall_score_normalized"]
            total_safety += normalized["safety_normalized"]
            total_effectiveness += normalized["effectiveness_alignment_normalized"]
            total_personalization += normalized["personalization_normalized"]
            details.append({**_compact_treatment_detail({"patient_id": patient_id, **normalized}),
                            "evaluation_status": normalized.get("evaluation_status", "evaluated")})

        total_treatments = len(rows)
        if total_treatments:
            return {
                "overall_score": total_score / total_treatments,
                "safety": total_safety / total_treatments,
                "effectiveness_alignment": total_effectiveness / total_treatments,
                "personalization": total_personalization / total_treatments,
                "details": details,
            }
        return {
            "overall_score": 0.0,
            "safety": 0.0,
            "effectiveness_alignment": 0.0,
            "personalization": 0.0,
            "details": details,
        }

    @staticmethod
    def _zero_treatment_eval_result() -> dict[str, Any]:
        return {
            "overall_score": 0.0,
            "safety": 0.0,
            "effectiveness_alignment": 0.0,
            "personalization": 0.0,
            "overall_score_normalized": 0.0,
            "safety_normalized": 0.0,
            "effectiveness_alignment_normalized": 0.0,
            "personalization_normalized": 0.0,
            "reasoning": "安全性、有效性与个体化评分均为0。",
        }

    def _zero_treatment_scores(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        zero_result = self._zero_treatment_eval_result()
        return {
            "overall_score": 0.0,
            "safety": 0.0,
            "effectiveness_alignment": 0.0,
            "personalization": 0.0,
            "details": [
                _compact_treatment_detail({
                    "patient_id": str(row.get("patient_id", "")).strip(),
                    **zero_result,
                })
                for row in rows
            ],
        }

    def _not_configured_treatment_scores(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        details = [
            {
                "patient_id": str(row.get("patient_id", "")).strip(),
                "overall_score": 0.0,
                "safety": 0.0,
                "effectiveness_alignment": 0.0,
                "personalization": 0.0,
                "evaluation_method": "llm_not_configured",
                "reasoning": "service/config.yaml model.model and model.base_url are required for template-compatible treatment scoring.",
            }
            for row in rows
        ]
        return {
            "overall_score": 0.0,
            "safety": 0.0,
            "effectiveness_alignment": 0.0,
            "personalization": 0.0,
            "details": details,
        }

    def _treatment_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for row in self.final_results:
            rows.append(
                {
                    "patient_id": str(row.get("patient_id", "")).strip(),
                    "doctor_id": row.get("doctor_id") or row.get("team_id") or "Unknown",
                    "diagnosis": row.get("diagnosis", ""),
                    "treatment_plan": row.get("treatment_plan", ""),
                }
            )
        return rows

    @staticmethod
    def _check_diagnosis_match(predicted: str, expected: str) -> bool:
        if not expected:
            return False
        predicted_lower = predicted.lower().strip()
        expected_lower = expected.lower().strip()
        return bool(predicted_lower) and predicted_lower == expected_lower

    def _check_diagnosis_match_with_comorbidity(self, predicted: Any, expected: Any) -> tuple[bool, int, int]:
        if isinstance(predicted, str):
            predicted_list = [predicted] if predicted else []
        else:
            predicted_list = predicted if predicted else []

        if isinstance(expected, str):
            expected_list = [expected] if expected else []
        else:
            expected_list = expected if expected else []

        if not expected_list:
            return False, 0, 0

        for predicted_item in predicted_list:
            if not any(self._check_diagnosis_match(predicted_item, expected_item) for expected_item in expected_list):
                return False, 0, len(expected_list)

        correct_count = 0
        for expected_item in expected_list:
            for predicted_item in predicted_list:
                if self._check_diagnosis_match(predicted_item, expected_item):
                    correct_count += 1
                    break
        return True, correct_count, len(expected_list)

    def _get_treatment_evaluator(self) -> TreatmentEvaluatorLLM:
        if self.treatment_evaluator is None:
            if not self._llm_configured():
                raise LLMEvaluationError("Treatment LLM evaluation is not configured.")
            self.treatment_evaluator = TreatmentEvaluatorLLM(
                evaluator=OpenAICompatibleEvaluator(
                    model=str(self.evaluation_model),
                    base_url=str(self.evaluation_base_url),
                    api_key=self.evaluation_api_key,
                    usage_callback=self.usage_callback,
                ),
            )
        return self.treatment_evaluator

    def _llm_configured(self) -> bool:
        return bool(normalize_optional_str(self.evaluation_model) and normalize_optional_str(self.evaluation_base_url))

    @staticmethod
    def _normalize_treatment_eval_result(
        raw_result: dict[str, Any],
        default_method: str = "llm",
    ) -> dict[str, Any]:
        result = dict(raw_result or {})

        def extract_dimension(raw_key: str, normalized_key: str) -> tuple[float, float]:
            if normalized_key in result:
                normalized = clamp(safe_float(result.get(normalized_key), 0.2), 0.0, 1.0)
                return normalized * 5.0, normalized

            raw = safe_float(result.get(raw_key), 1.0)
            if raw <= 1.0:
                normalized = clamp(raw, 0.0, 1.0)
                return normalized * 5.0, normalized

            raw_score = clamp(raw, 1.0, 5.0)
            return raw_score, raw_score / 5.0

        safety_raw, safety_norm = extract_dimension("safety", "safety_normalized")
        effectiveness_raw, effectiveness_norm = extract_dimension(
            "effectiveness_alignment",
            "effectiveness_alignment_normalized",
        )
        personalization_raw, personalization_norm = extract_dimension("personalization", "personalization_normalized")

        if "overall_score_normalized" in result:
            overall_norm = clamp(safe_float(result["overall_score_normalized"], 0.2), 0.0, 1.0)
            overall_raw = overall_norm * 5.0
        elif "overall_score" in result:
            overall_value = safe_float(result["overall_score"], 1.0)
            if overall_value <= 1.0:
                overall_norm = clamp(overall_value, 0.0, 1.0)
                overall_raw = overall_norm * 5.0
            else:
                overall_raw = clamp(overall_value, 1.0, 5.0)
                overall_norm = overall_raw / 5.0
        else:
            safety_penalty = {5: 0.0, 4: 0.5, 3: 1.0, 2: 2.0, 1: 4.0}
            safety_level = int(clamp(safety_raw, 1.0, 5.0))
            base_score = effectiveness_raw * 0.6 + personalization_raw * 0.4
            overall_raw = max(1.0, base_score - safety_penalty.get(safety_level, 0.0))
            overall_norm = overall_raw / 5.0

        result["safety"] = round(safety_raw, 4)
        result["effectiveness_alignment"] = round(effectiveness_raw, 4)
        result["personalization"] = round(personalization_raw, 4)
        result["overall_score"] = round(overall_raw, 4)
        result["safety_normalized"] = round(safety_norm, 4)
        result["effectiveness_alignment_normalized"] = round(effectiveness_norm, 4)
        result["personalization_normalized"] = round(personalization_norm, 4)
        result["overall_score_normalized"] = round(overall_norm, 4)
        result["evaluation_method"] = str(result.get("evaluation_method") or default_method)
        result["reasoning"] = str(result.get("reasoning", ""))
        result["attempts"] = safe_int(result.get("attempts", 1), 1)
        return result


def normalize_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in PLACEHOLDER_CONFIG_VALUES:
        return None
    return text


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def fallback_treatment_reasoning(result: dict[str, Any]) -> str:
    safety = safe_float(result.get("safety"), 1.0)
    effectiveness = safe_float(result.get("effectiveness_alignment"), 1.0)
    personalization = safe_float(result.get("personalization"), 1.0)

    safety_text = score_band_text(
        safety,
        ("安全性良好", "安全性基本可控", "安全性一般", "存在一定安全风险"),
    )
    effectiveness_text = score_band_text(
        effectiveness,
        ("与参考方案高度一致", "与参考方案较一致", "与参考方案部分一致", "与参考方案差异较大"),
    )
    personalization_text = score_band_text(
        personalization,
        ("个体化充分", "具备一定个体化", "个体化程度一般", "个体化不足"),
    )
    return f"{safety_text}，{effectiveness_text}，{personalization_text}。"


def score_band_text(score: float, labels: tuple[str, str, str, str]) -> str:
    if score >= 4.5:
        return labels[0]
    if score >= 3.5:
        return labels[1]
    if score >= 2.5:
        return labels[2]
    return labels[3]


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def normalize_llm_usage(usage: Any, prompt: Any, answer: Any) -> dict[str, int]:
    if isinstance(usage, dict):
        prompt_tokens = safe_int(usage.get("prompt_tokens"), 0) or estimate_tokens(prompt)
        completion_tokens = safe_int(usage.get("completion_tokens"), 0) or estimate_tokens(answer)
        total_tokens = safe_int(usage.get("total_tokens"), 0) or prompt_tokens + completion_tokens
    else:
        prompt_tokens = estimate_tokens(prompt)
        completion_tokens = estimate_tokens(answer)
        total_tokens = prompt_tokens + completion_tokens
    return {
        "prompt_tokens": max(0, prompt_tokens),
        "completion_tokens": max(0, completion_tokens),
        "total_tokens": max(0, total_tokens),
    }


def merge_usage_records(records: list[dict[str, int]]) -> dict[str, int]:
    merged = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for record in records:
        merged["prompt_tokens"] += safe_int(
            record.get("prompt_tokens") or record.get("input_tokens") or record.get("inputTokens"),
            0,
        )
        merged["completion_tokens"] += safe_int(
            record.get("completion_tokens") or record.get("output_tokens") or record.get("outputTokens"),
            0,
        )
        merged["total_tokens"] += safe_int(
            record.get("total_tokens") or record.get("totalTokens"),
            0,
        )
    if not merged["total_tokens"]:
        merged["total_tokens"] = merged["prompt_tokens"] + merged["completion_tokens"]
    return merged


def estimate_tokens(value: Any) -> int:
    if isinstance(value, str):
        text = value
    else:
        text = json.dumps(value, ensure_ascii=False, default=str)
    return max(1, len(text) // 4)


def _compact_treatment_detail(detail: dict[str, Any]) -> dict[str, Any]:
    return {
        "patient_id": str(detail.get("patient_id") or ""),
        "overall_score": _normalized_detail_score(detail, "overall_score"),
        "safety": _normalized_detail_score(detail, "safety"),
        "effectiveness_alignment": _normalized_detail_score(detail, "effectiveness_alignment"),
        "personalization": _normalized_detail_score(detail, "personalization"),
        "reasoning": str(detail.get("reasoning") or ""),
    }


def _normalized_detail_score(detail: dict[str, Any], key: str) -> float:
    normalized_key = f"{key}_normalized"
    if normalized_key in detail:
        return round(clamp(safe_float(detail.get(normalized_key), 0.0), 0.0, 1.0), 4)
    value = safe_float(detail.get(key), 0.0)
    if value > 1.0:
        value = value / 5.0
    return round(clamp(value, 0.0, 1.0), 4)


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def normalize_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def display_list_or_string(value: Any) -> str:
    return ", ".join(normalize_list(value))


def safe_get(data: dict[str, Any] | None, key: str, default: str = "") -> str:
    if data is None:
        return default
    value = data.get(key, default)
    if isinstance(value, list):
        return ", ".join(str(item) for item in value) if value else default
    return str(value) if value else default


def format_history_obj(value: Any, default_name: str = "未提供") -> str:
    if not value:
        return default_name
    return json.dumps(value, ensure_ascii=False, indent=2)


def format_examinations(exams: Any) -> str:
    if not exams:
        return "无可用检查数据"
    return "\n".join(
        f"- {exam.get('item_name', '未知检查项目')}: {exam.get('result', '无检查结果')}"
        for exam in exams[:10]
    )


def format_contraindications(patient_data: dict[str, Any] | None) -> str:
    if not patient_data:
        return "无可用禁忌证信息"

    ground_truth = patient_data.get("ground_truth", {})
    contraindications = ground_truth.get("contraindications", {})
    if not contraindications:
        return "未记录明确禁忌证"

    parts = []
    drugs = contraindications.get("drugs", [])
    treatments = contraindications.get("treatments", [])

    if drugs:
        if isinstance(drugs, list):
            parts.append(f"- 禁忌药物：{', '.join(str(drug) for drug in drugs)}")
        else:
            parts.append(f"- 禁忌药物：{drugs}")

    if treatments:
        if isinstance(treatments, list):
            parts.append(f"- 禁忌治疗：{', '.join(str(treatment) for treatment in treatments)}")
        else:
            parts.append(f"- 禁忌治疗：{treatments}")

    return "\n".join(parts) if parts else "未记录明确禁忌证"
