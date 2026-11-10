"""Evaluation wrapper for the unified contest service."""

from __future__ import annotations

from typing import Any

try:
    from ..repository import ServiceDataRepository
    from ..token_usage import InMemoryTokenUsageStore, TokenUsageStore
except ImportError:
    from repository import ServiceDataRepository
    from token_usage import InMemoryTokenUsageStore, TokenUsageStore

try:
    from .evaluator import FinalResultEvaluator
except ImportError:
    from evaluator import FinalResultEvaluator


class EvaluationService:
    """Evaluate submitted final results against service ground truth."""

    def __init__(
        self,
        repository: ServiceDataRepository,
        *,
        model_settings: Any | None = None,
        usage_store: TokenUsageStore | None = None,
        model_key_pool: Any | None = None,
    ):
        self.repository = repository
        self.model_settings = model_settings
        self.usage_store = usage_store or InMemoryTokenUsageStore()
        self.model_key_pool = model_key_pool

    def evaluate(self, payload: dict[str, Any], *, include_reference_answers: bool = False) -> dict[str, Any]:
        final_results = _ensure_list(payload.get("final_result"), "final_result")
        treatment_evaluation_mode = _treatment_evaluation_mode(payload)
        report = self._evaluate_with_optional_pool(
            final_results=final_results,
            api_key=_request_api_key(payload),
            team_id=str(payload.get("team_id") or "").strip(),
            treatment_evaluation_mode=treatment_evaluation_mode,
        )
        _attach_timing(report, payload)
        return report if include_reference_answers else _public_batch_report(report)

    def evaluate_case(self, payload: dict[str, Any], *, include_reference_answers: bool = False) -> dict[str, Any]:
        final_result = payload.get("final_result")
        if not isinstance(final_result, dict):
            raise ValueError("final_result must be a JSON object.")
        treatment_evaluation_mode = _treatment_evaluation_mode(payload)
        report = self._evaluate_with_optional_pool(
            final_results=[final_result],
            api_key=_request_api_key(payload),
            team_id=str(payload.get("team_id") or "").strip(),
            treatment_evaluation_mode=treatment_evaluation_mode,
        )
        return self._case_report(final_result, report, include_reference_answers=include_reference_answers)

    async def evaluate_async(self, payload: dict[str, Any], *, include_reference_answers: bool = False) -> dict[str, Any]:
        final_results = _ensure_list(payload.get("final_result"), "final_result")
        treatment_evaluation_mode = _treatment_evaluation_mode(payload)
        report = await self._evaluate_with_optional_pool_async(
            final_results=final_results,
            api_key=_request_api_key(payload),
            team_id=str(payload.get("team_id") or "").strip(),
            treatment_evaluation_mode=treatment_evaluation_mode,
        )
        _attach_timing(report, payload)
        return report if include_reference_answers else _public_batch_report(report)

    async def evaluate_case_async(self, payload: dict[str, Any], *, include_reference_answers: bool = False) -> dict[str, Any]:
        final_result = payload.get("final_result")
        if not isinstance(final_result, dict):
            raise ValueError("final_result must be a JSON object.")
        treatment_evaluation_mode = _treatment_evaluation_mode(payload)
        report = await self._evaluate_with_optional_pool_async(
            final_results=[final_result],
            api_key=_request_api_key(payload),
            team_id=str(payload.get("team_id") or "").strip(),
            treatment_evaluation_mode=treatment_evaluation_mode,
        )
        return self._case_report(final_result, report, include_reference_answers=include_reference_answers)

    def _evaluate_with_optional_pool(
        self,
        *,
        final_results: list[dict[str, Any]],
        api_key: str | None,
        team_id: str,
        treatment_evaluation_mode: str,
    ) -> dict[str, Any]:
        force_zero_treatment = treatment_evaluation_mode == "zero"
        scores_only_treatment = treatment_evaluation_mode == "scores_only"
        if force_zero_treatment or api_key or self.model_key_pool is None:
            return self._build_evaluator(api_key=api_key, team_id=team_id).evaluate(
                final_results,
                force_zero_treatment=force_zero_treatment,
                scores_only_treatment=scores_only_treatment,
            )
        return self.model_key_pool.run(
            operation=lambda pool_api_key: self._build_evaluator(
                api_key=pool_api_key,
                team_id=team_id,
                record_usage_for_explicit_key=True,
            ).evaluate(
                final_results,
                force_zero_treatment=False,
                scores_only_treatment=scores_only_treatment,
            ),
            usage_extractor=lambda report: report.get("_usageDelta", {}) if isinstance(report, dict) else {},
        )

    async def _evaluate_with_optional_pool_async(
        self,
        *,
        final_results: list[dict[str, Any]],
        api_key: str | None,
        team_id: str,
        treatment_evaluation_mode: str,
    ) -> dict[str, Any]:
        force_zero_treatment = treatment_evaluation_mode == "zero"
        scores_only_treatment = treatment_evaluation_mode == "scores_only"
        if force_zero_treatment or api_key or self.model_key_pool is None:
            return await self._build_evaluator(api_key=api_key, team_id=team_id).evaluate_async(
                final_results,
                force_zero_treatment=force_zero_treatment,
                scores_only_treatment=scores_only_treatment,
            )
        return await self.model_key_pool.run_async(
            operation=lambda pool_api_key: self._build_evaluator(
                api_key=pool_api_key,
                team_id=team_id,
                record_usage_for_explicit_key=True,
            ).evaluate_async(
                final_results,
                force_zero_treatment=False,
                scores_only_treatment=scores_only_treatment,
            ),
            usage_extractor=lambda report: report.get("_usageDelta", {}) if isinstance(report, dict) else {},
        )

    def _build_evaluator(
        self,
        *,
        api_key: str | None = None,
        team_id: str = "",
        record_usage_for_explicit_key: bool = False,
    ) -> FinalResultEvaluator:
        model_kwargs: dict[str, Any] = {}
        if self.model_settings is not None:
            model_kwargs = {
                "model": getattr(self.model_settings, "model", None),
                "base_url": getattr(self.model_settings, "base_url", None),
                "api_key": api_key or getattr(self.model_settings, "api_key", None),
            }
        usage_callback = None
        using_default_api_key = not str(api_key or "").strip()
        captured_usage: list[dict[str, int]] = []
        if team_id and (using_default_api_key or record_usage_for_explicit_key):
            def usage_callback(usage: dict[str, int]) -> None:
                usage_delta = _normalize_usage_delta(usage)
                captured_usage.append(usage_delta)
                self._record_usage(team_id=team_id, usage=usage_delta)
        return FinalResultEvaluator(
            repository=self.repository,
            usage_callback=usage_callback,
            usage_capture=captured_usage,
            **model_kwargs,
        )

    def _record_usage(self, *, team_id: str, usage: dict[str, int]) -> dict[str, int]:
        return self.usage_store.increment_usage(
            dataset_key=self.repository.dataset_key,
            team_id=team_id,
            input_tokens=_safe_int(usage.get("prompt_tokens")),
            output_tokens=_safe_int(usage.get("completion_tokens")),
            total_tokens=_safe_int(usage.get("total_tokens")),
        )

    def _case_report(
        self,
        final_result: dict[str, Any],
        report: dict[str, Any],
        *,
        include_reference_answers: bool = False,
    ) -> dict[str, Any]:
        patient_id = str(final_result.get("patient_id") or "")
        ground_truth = self.repository.get_ground_truth(patient_id)
        treatment_detail = _build_treatment_detail(
            report,
            patient_id,
            final_result,
            ground_truth,
            include_reference_answers=include_reference_answers,
        )
        diagnosis_detail = _build_diagnosis_detail(final_result, ground_truth)
        examination_detail = _build_examination_detail(final_result, ground_truth)
        if not include_reference_answers:
            diagnosis_detail = _public_diagnosis_detail(diagnosis_detail)
            examination_detail = _public_examination_detail(examination_detail)
        case_report = {
            "patientId": patient_id,
            "status": report.get("evaluation_status", "evaluated"),
            "evaluationStatus": report.get("evaluation_status", "evaluated"),
            "diagnosisAccuracy": report["diagnosis_accuracy"],
            "examinationPrecision": report["examination_precision"],
            "treatmentOverallScore": report["treatment_overall_score"],
            "treatmentSafety": report["treatment_safety"],
            "treatmentEffectivenessAlignment": report["treatment_effectiveness_alignment"],
            "treatmentPersonalization": report["treatment_personalization"],
            "diagnosisDetail": diagnosis_detail,
            "examinationDetail": examination_detail,
            "treatmentDetail": treatment_detail,
        }
        if include_reference_answers:
            case_report["ground_truth"] = ground_truth
        return case_report


def _public_batch_report(report: dict[str, Any]) -> dict[str, Any]:
    public = dict(report)
    if isinstance(public.get("treatment_details"), list):
        public["treatment_details"] = [
            _public_treatment_summary(item)
            for item in public["treatment_details"]
            if isinstance(item, dict)
        ]
    return public


def _public_treatment_summary(detail: dict[str, Any]) -> dict[str, Any]:
    return {
        "patient_id": str(detail.get("patient_id") or ""),
        "overall_score": _safe_float(detail.get("overall_score"), 0.0),
        "safety": _safe_float(detail.get("safety"), 0.0),
        "effectiveness_alignment": _safe_float(detail.get("effectiveness_alignment"), 0.0),
        "personalization": _safe_float(detail.get("personalization"), 0.0),
        "reasoning": str(detail.get("reasoning") or ""),
    }


def _ensure_list(value: Any, field_name: str) -> list[dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a JSON array.")
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(value):
        if not isinstance(row, dict):
            raise ValueError(f"{field_name}[{index}] must be a JSON object.")
        rows.append(row)
    return rows


def _request_api_key(payload: dict[str, Any]) -> str | None:
    value = (
        payload.get("api_key")
        or payload.get("apiKey")
        or payload.get("model_api_key")
        or payload.get("modelApiKey")
        or payload.get("model_key")
        or payload.get("modelKey")
    )
    text = str(value).strip() if value is not None else ""
    return text or None


def _treatment_evaluation_mode(payload: dict[str, Any]) -> str:
    mode = str(payload.get("_treatment_evaluation_mode") or "").strip().lower()
    return mode if mode in {"zero", "scores_only"} else "normal"


def _normalize_usage_delta(usage: dict[str, Any]) -> dict[str, int]:
    prompt_tokens = _safe_int(usage.get("prompt_tokens") or usage.get("input_tokens") or usage.get("inputTokens"))
    completion_tokens = _safe_int(
        usage.get("completion_tokens") or usage.get("output_tokens") or usage.get("outputTokens"),
    )
    total_tokens = _safe_int(usage.get("total_tokens") or usage.get("totalTokens"))
    if not total_tokens:
        total_tokens = prompt_tokens + completion_tokens
    return {
        "prompt_tokens": max(0, prompt_tokens),
        "completion_tokens": max(0, completion_tokens),
        "total_tokens": max(0, total_tokens),
    }


def _attach_timing(report: dict[str, Any], payload: dict[str, Any]) -> None:
    elapsed_seconds = _optional_int(payload.get("elapsed_seconds"))
    if elapsed_seconds is None:
        return
    started_at = payload.get("started_at")
    submitted_at = payload.get("submitted_at")
    report["elapsed_seconds"] = elapsed_seconds
    if started_at is not None:
        report["test_started_at"] = str(started_at)
    if submitted_at is not None:
        report["test_submitted_at"] = str(submitted_at)
    report["timing"] = {
        "elapsedSeconds": elapsed_seconds,
        "startedAt": str(started_at) if started_at is not None else "",
        "submittedAt": str(submitted_at) if submitted_at is not None else "",
    }


def _optional_int(value: Any) -> int | None:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return max(0, number)


def _build_diagnosis_detail(final_result: dict[str, Any], ground_truth: dict[str, Any]) -> dict[str, Any]:
    submitted = _normalize_text_list(final_result.get("diagnosis"))
    expected = _normalize_text_list(ground_truth.get("final_diagnosis"))
    matched = []
    for expected_item in expected:
        if any(_diagnosis_matches(submitted_item, expected_item) for submitted_item in submitted):
            matched.append(expected_item)

    correct_count = len(matched)
    total_expected = len(expected)
    submitted_is_subset = all(
        any(_diagnosis_matches(submitted_item, expected_item) for expected_item in expected)
        for submitted_item in submitted
    )
    if total_expected == 0:
        status = "unknown"
        accuracy = 0.0
    elif not submitted_is_subset:
        status = "wrong"
        accuracy = 0.0
    elif correct_count == total_expected:
        status = "correct"
        accuracy = 1.0
    elif correct_count == 0:
        status = "wrong"
        accuracy = 0.0
    else:
        status = "partial"
        accuracy = correct_count / total_expected

    return {
        "submitted": submitted,
        "expected": expected,
        "matched": matched,
        "accuracy": accuracy,
        "status": status,
    }


def _build_examination_detail(final_result: dict[str, Any], ground_truth: dict[str, Any]) -> dict[str, Any]:
    ordered = _unique_preserve_order(_normalize_text_list(final_result.get("ordered_examinations")))
    expected = _unique_preserve_order(_normalize_text_list(ground_truth.get("necessary_examinations")))
    ordered_set = set(ordered)
    expected_set = set(expected)
    matched = [item for item in ordered if item in expected_set]
    precision = len(matched) / len(ordered_set) if ordered_set and expected_set else 0.0
    coverage = len(set(matched)) / len(expected_set) if expected_set else 0.0
    return {
        "ordered": ordered,
        "expected": expected,
        "matched": matched,
        "precision": precision,
        "coverage": coverage,
        "precisionEligible": bool(ordered_set and expected_set),
    }


def _build_treatment_detail(
    report: dict[str, Any],
    patient_id: str,
    final_result: dict[str, Any],
    ground_truth: dict[str, Any],
    *,
    include_reference_answers: bool = False,
) -> dict[str, Any]:
    detail = {}
    details = report.get("treatment_details")
    if isinstance(details, list):
        detail = next(
            (
                item
                for item in details
                if isinstance(item, dict) and str(item.get("patient_id") or "") == patient_id
            ),
            {},
        )
        if not detail and details and isinstance(details[0], dict):
            detail = details[0]

    treatment_detail = {
        "submitted": str(final_result.get("treatment_plan") or ""),
        "overallScore": _first_score(detail, report, "overall_score", "treatment_overall_score"),
        "safety": _first_score(detail, report, "safety", "treatment_safety"),
        "effectivenessAlignment": _first_score(
            detail,
            report,
            "effectiveness_alignment",
            "treatment_effectiveness_alignment",
        ),
        "personalization": _first_score(detail, report, "personalization", "treatment_personalization"),
        "reasoning": str(detail.get("reasoning") or ""),
    }
    if include_reference_answers:
        treatment_detail["reference"] = str(ground_truth.get("treatment_plan") or "")
    return treatment_detail


def _public_diagnosis_detail(detail: dict[str, Any]) -> dict[str, Any]:
    return {
        "submitted": detail.get("submitted") or [],
        "accuracy": detail.get("accuracy") or 0.0,
        "status": detail.get("status") or "unknown",
    }


def _public_examination_detail(detail: dict[str, Any]) -> dict[str, Any]:
    return {
        "ordered": detail.get("ordered") or [],
        "precision": detail.get("precision") or 0.0,
        "coverage": detail.get("coverage") or 0.0,
        "precisionEligible": bool(detail.get("precisionEligible")),
    }


def _first_score(primary: dict[str, Any], fallback: dict[str, Any], primary_key: str, fallback_key: str) -> float:
    if primary_key in primary:
        return _safe_float(primary.get(primary_key), 0.0)
    return _safe_float(fallback.get(fallback_key), 0.0)


def _diagnosis_matches(predicted: str, expected: str) -> bool:
    predicted_lower = predicted.lower().strip()
    expected_lower = expected.lower().strip()
    if not predicted_lower or not expected_lower:
        return False
    return predicted_lower == expected_lower


def _normalize_text_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def _unique_preserve_order(values: list[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
