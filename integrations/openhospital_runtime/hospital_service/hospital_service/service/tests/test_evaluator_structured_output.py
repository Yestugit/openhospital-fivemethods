from __future__ import annotations

import asyncio
import json

import pytest

from hospital_service.service.evaluation import evaluator


class CaseEvaluationRepository:
    def get_ground_truth(self, patient_id):
        return {
            "final_diagnosis": f"diagnosis-{patient_id}",
            "necessary_examinations": [f"exam-{patient_id}"],
            "treatment_plan": f"reference-{patient_id}",
        }

    def get_patient_context(self, patient_id):
        return {"patient_id": patient_id}


def final_result(patient_id):
    return {
        "patient_id": patient_id,
        "diagnosis": [f"diagnosis-{patient_id}"],
        "ordered_examinations": [f"exam-{patient_id}"],
        "treatment_plan": f"generated-{patient_id}",
    }


def test_openai_compatible_evaluator_requests_json_object_output(monkeypatch):
    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def read(self):
            return json.dumps({
                "choices": [{
                    "message": {
                        "content": json.dumps({
                            "safety": 5,
                            "effectiveness_alignment": 4,
                            "personalization": 4,
                            "reasoning": "治疗方案安全且与参考方案一致。",
                        }, ensure_ascii=False),
                    },
                }],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
            }, ensure_ascii=False).encode("utf-8")

    def fake_urlopen(request, timeout):
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(evaluator.urllib.request, "urlopen", fake_urlopen)
    client = evaluator.OpenAICompatibleEvaluator(
        model="qwen-test",
        base_url="https://model.test/v1",
        api_key="secret",
    )

    response = client._evaluate_sync("user prompt", "system prompt")

    assert json.loads(response)["safety"] == 5
    assert captured["payload"]["response_format"] == {"type": "json_object"}
    assert captured["payload"]["enable_thinking"] is False
    assert captured["payload"]["max_tokens"] == evaluator.TREATMENT_LLM_PRIMARY_MAX_TOKENS == 256
    assert captured["timeout"] == evaluator.TREATMENT_LLM_PRIMARY_TIMEOUT_SECONDS == 70
    assert (
        evaluator.TREATMENT_LLM_PRIMARY_TIMEOUT_SECONDS
        + evaluator.TREATMENT_LLM_FALLBACK_TIMEOUT_SECONDS
        == evaluator.TREATMENT_LLM_TOTAL_TIMEOUT_SECONDS
        == 90
    )


def test_treatment_evaluator_uses_scores_only_fallback_after_timeout():
    class FailingEvaluator:
        calls = []

        async def evaluate(self, prompt, system_prompt, *, timeout_seconds, max_tokens):
            self.calls.append({
                "prompt": prompt,
                "system_prompt": system_prompt,
                "timeout_seconds": timeout_seconds,
                "max_tokens": max_tokens,
            })
            if len(self.calls) == 1:
                raise evaluator.LLMEvaluationTransientError("timed out")
            return json.dumps({
                "safety": 5,
                "effectiveness_alignment": 4,
                "personalization": 3,
            })

    failing_evaluator = FailingEvaluator()
    treatment_evaluator = evaluator.TreatmentEvaluatorLLM(
        evaluator=failing_evaluator,
        retry_delay=0,
    )

    result = asyncio.run(treatment_evaluator.evaluate_treatment(
        diagnosis="diagnosis",
        generated_treatment="generated treatment",
        reference_treatment="reference treatment",
        patient_context={},
    ))

    assert result["reasoning"] == "安全性良好，与参考方案较一致，个体化程度一般。"
    assert "降级" not in result["reasoning"]
    assert result["evaluation_method"] == "llm_scores_only_fallback"
    assert result["attempts"] == 2
    assert len(failing_evaluator.calls) == 2
    assert failing_evaluator.calls[0]["timeout_seconds"] == 70
    assert failing_evaluator.calls[0]["max_tokens"] == 256
    assert failing_evaluator.calls[1]["timeout_seconds"] == 20
    assert failing_evaluator.calls[1]["max_tokens"] == 128
    assert "不要输出 reasoning" in failing_evaluator.calls[1]["system_prompt"]
    assert "不要输出 reasoning" in failing_evaluator.calls[1]["prompt"]
    assert "80 个中文字符" not in failing_evaluator.calls[1]["prompt"]


def test_treatment_evaluator_falls_back_after_invalid_primary_json():
    class InvalidThenValidEvaluator:
        calls = 0

        async def evaluate(self, prompt, system_prompt, *, timeout_seconds, max_tokens):
            self.calls += 1
            if self.calls == 1:
                return '{"safety": 5, "reasoning": "unterminated}'
            return '{"safety": 5, "effectiveness_alignment": 4, "personalization": 3}'

    model = InvalidThenValidEvaluator()
    result = asyncio.run(evaluator.TreatmentEvaluatorLLM(model).evaluate_treatment(
        diagnosis="diagnosis",
        generated_treatment="generated treatment",
        reference_treatment="reference treatment",
        patient_context={},
    ))

    assert model.calls == 2
    assert result["evaluation_method"] == "llm_scores_only_fallback"
    assert result["reasoning"] == "安全性良好，与参考方案较一致，个体化程度一般。"


def test_treatment_evaluator_output_security_fallback_omits_reasoning():
    class OutputBlockedThenScoresEvaluator:
        calls = []

        async def evaluate(self, prompt, system_prompt, *, timeout_seconds, max_tokens):
            self.calls.append({
                "prompt": prompt,
                "system_prompt": system_prompt,
                "timeout_seconds": timeout_seconds,
                "max_tokens": max_tokens,
            })
            if len(self.calls) == 1:
                raise evaluator.LLMEvaluationHTTPError(
                    555,
                    "output content security inspection failed by Modelscope.",
                )
            return '{"safety": 5, "effectiveness_alignment": 4, "personalization": 3}'

    model = OutputBlockedThenScoresEvaluator()
    result = asyncio.run(evaluator.TreatmentEvaluatorLLM(model).evaluate_treatment(
        diagnosis="diagnosis",
        generated_treatment="generated treatment",
        reference_treatment="reference treatment",
        patient_context={},
    ))

    assert len(model.calls) == 2
    assert model.calls[1]["timeout_seconds"] == 20
    assert model.calls[1]["max_tokens"] == 128
    assert "不要输出 reasoning" in model.calls[1]["prompt"]
    assert result["safety"] == 5
    assert result["effectiveness_alignment"] == 4
    assert result["personalization"] == 3
    assert result["reasoning"] == ""
    assert result["evaluation_method"] == "llm_scores_only_fallback"


def test_treatment_evaluator_does_not_fallback_on_deterministic_http_error():
    class UnauthorizedEvaluator:
        calls = 0

        async def evaluate(self, prompt, system_prompt, *, timeout_seconds, max_tokens):
            self.calls += 1
            raise evaluator.LLMEvaluationHTTPError(401, "unauthorized")

    model = UnauthorizedEvaluator()
    try:
        asyncio.run(evaluator.TreatmentEvaluatorLLM(model).evaluate_treatment(
            diagnosis="diagnosis",
            generated_treatment="generated treatment",
            reference_treatment="reference treatment",
            patient_context={},
        ))
    except evaluator.LLMEvaluationHTTPError as exc:
        assert exc.status_code == 401
    else:
        raise AssertionError("Expected the deterministic HTTP error to be preserved.")

    assert model.calls == 1


def test_treatment_evaluator_preserves_deterministic_fallback_http_error():
    class InvalidThenForbiddenEvaluator:
        calls = 0

        async def evaluate(self, prompt, system_prompt, *, timeout_seconds, max_tokens):
            self.calls += 1
            if self.calls == 1:
                return '{"safety": 5}'
            raise evaluator.LLMEvaluationHTTPError(403, "forbidden")

    model = InvalidThenForbiddenEvaluator()
    with pytest.raises(evaluator.LLMEvaluationHTTPError) as error:
        asyncio.run(evaluator.TreatmentEvaluatorLLM(model).evaluate_treatment(
            diagnosis="diagnosis",
            generated_treatment="generated treatment",
            reference_treatment="reference treatment",
            patient_context={},
        ))

    assert error.value.status_code == 403
    assert model.calls == 2


def test_openai_compatible_evaluator_classifies_timeout_as_transient(monkeypatch):
    def fake_urlopen(request, timeout):
        raise TimeoutError("read timed out")

    monkeypatch.setattr(evaluator.urllib.request, "urlopen", fake_urlopen)
    client = evaluator.OpenAICompatibleEvaluator(
        model="qwen-test",
        base_url="https://model.test/v1",
        api_key="secret",
    )

    try:
        client._evaluate_sync("user prompt", "system prompt")
    except evaluator.LLMEvaluationTransientError as exc:
        assert "timed out after 70 seconds" in str(exc)
    else:
        raise AssertionError("Expected the model timeout to be classified as transient.")


def test_transient_error_detection_follows_public_key_pool_error_chain():
    transient_error = evaluator.LLMEvaluationTransientError("timed out")
    wrapped_error = RuntimeError("All public model API keys failed")
    wrapped_error.__cause__ = transient_error

    assert evaluator.is_transient_llm_error(wrapped_error) is True
    assert evaluator.is_transient_llm_error(RuntimeError("deterministic failure")) is False


def test_final_result_evaluator_zeroes_failed_treatment_and_continues():
    class FirstPatientFails:
        calls = []

        async def evaluate_treatment(self, *, patient_context, **kwargs):
            patient_id = patient_context["patient_id"]
            self.calls.append(patient_id)
            if patient_id == "Patient_001":
                raise evaluator.LLMEvaluationTransientError(
                    "primary timed out; scores-only fallback timed out"
                )
            return {
                "overall_score": 5,
                "safety": 5,
                "effectiveness_alignment": 5,
                "personalization": 5,
                "reasoning": "评分正常。",
            }

    treatment_evaluator = FirstPatientFails()
    final_evaluator = evaluator.FinalResultEvaluator(
        CaseEvaluationRepository(),
        model="model",
        base_url="https://model.test/v1",
        api_key="secret",
    )
    final_evaluator.treatment_evaluator = treatment_evaluator

    report = asyncio.run(final_evaluator.evaluate_async([
        final_result("Patient_001"),
        final_result("Patient_002"),
    ]))

    assert treatment_evaluator.calls == ["Patient_001", "Patient_002"]
    assert report["diagnosis_accuracy"] == 1.0
    assert report["examination_precision"] == 1.0
    assert report["treatment_overall_score"] == 0.5
    assert report["treatment_safety"] == 0.5
    assert report["treatment_effectiveness_alignment"] == 0.5
    assert report["treatment_personalization"] == 0.5
    assert report["treatment_details"][0] == {
        "evaluation_status": "failed",
        "patient_id": "Patient_001",
        "overall_score": 0.0,
        "safety": 0.0,
        "effectiveness_alignment": 0.0,
        "personalization": 0.0,
        "reasoning": "安全性、有效性与个体化评分均为0。",
    }
    assert report["treatment_details"][1]["overall_score"] == 1.0


def test_final_result_evaluator_single_failed_treatment_preserves_other_scores():
    class AlwaysFails:
        async def evaluate_treatment(self, **kwargs):
            raise evaluator.LLMEvaluationError("invalid model response containing private detail")

    final_evaluator = evaluator.FinalResultEvaluator(
        CaseEvaluationRepository(),
        model="model",
        base_url="https://model.test/v1",
        api_key="secret",
    )
    final_evaluator.treatment_evaluator = AlwaysFails()

    report = asyncio.run(final_evaluator.evaluate_async([final_result("Patient_001")]))

    assert report["diagnosis_accuracy"] == 1.0
    assert report["examination_precision"] == 1.0
    for field in (
        "treatment_overall_score",
        "treatment_safety",
        "treatment_effectiveness_alignment",
        "treatment_personalization",
    ):
        assert report[field] == 0.0
    public_text = json.dumps(report, ensure_ascii=False)
    assert "invalid model response" not in public_text
    assert "降级" not in public_text
    assert "fallback" not in public_text
    assert "evaluation_method" not in public_text


def test_final_result_evaluator_force_zero_treatment_does_not_call_model():
    class UnexpectedTreatmentCall:
        async def evaluate_treatment(self, **kwargs):
            raise AssertionError("Treatment model must not be called in forced-zero mode.")

    final_evaluator = evaluator.FinalResultEvaluator(
        CaseEvaluationRepository(),
        model="model",
        base_url="https://model.test/v1",
        api_key="secret",
    )
    final_evaluator.treatment_evaluator = UnexpectedTreatmentCall()

    report = asyncio.run(final_evaluator.evaluate_async(
        [final_result("Patient_001")],
        force_zero_treatment=True,
    ))

    assert report["diagnosis_accuracy"] == 1.0
    assert report["examination_precision"] == 1.0
    assert report["treatment_overall_score"] == 0.0
    assert report["treatment_safety"] == 0.0
    assert report["treatment_effectiveness_alignment"] == 0.0
    assert report["treatment_personalization"] == 0.0
    assert report["treatment_details"][0]["reasoning"] == "安全性、有效性与个体化评分均为0。"


def test_final_result_evaluator_scores_only_treatment_preserves_scores_without_reasoning():
    class ScoresOnlyTreatmentEvaluator:
        normal_calls = 0
        scores_only_calls = 0

        async def evaluate_treatment(self, **kwargs):
            self.normal_calls += 1
            raise AssertionError("Normal treatment evaluation must not be called.")

        async def evaluate_treatment_scores_only(self, **kwargs):
            self.scores_only_calls += 1
            return {
                "overall_score": 3.6,
                "safety": 5,
                "effectiveness_alignment": 4,
                "personalization": 3,
                "reasoning": "这段文字不得进入最终结果。",
            }

    treatment_evaluator = ScoresOnlyTreatmentEvaluator()
    final_evaluator = evaluator.FinalResultEvaluator(
        CaseEvaluationRepository(),
        model="model",
        base_url="https://model.test/v1",
        api_key="secret",
    )
    final_evaluator.treatment_evaluator = treatment_evaluator

    report = asyncio.run(final_evaluator.evaluate_async(
        [final_result("Patient_001")],
        scores_only_treatment=True,
    ))

    assert treatment_evaluator.normal_calls == 0
    assert treatment_evaluator.scores_only_calls == 1
    assert report["diagnosis_accuracy"] == 1.0
    assert report["examination_precision"] == 1.0
    assert report["treatment_overall_score"] == 0.72
    assert report["treatment_safety"] == 1.0
    assert report["treatment_effectiveness_alignment"] == 0.8
    assert report["treatment_personalization"] == 0.6
    assert report["treatment_details"][0]["reasoning"] == ""


def test_final_result_evaluator_zeroes_modelscope_input_security_treatment_error():
    class ModelScopeInputSecurityFailure:
        async def evaluate_treatment(self, **kwargs):
            raise evaluator.LLMEvaluationHTTPError(
                555,
                "input content security inspection failed by Modelscope.",
            )

    final_evaluator = evaluator.FinalResultEvaluator(
        CaseEvaluationRepository(),
        model="model",
        base_url="https://model.test/v1",
        api_key="secret",
    )
    final_evaluator.treatment_evaluator = ModelScopeInputSecurityFailure()

    report = asyncio.run(final_evaluator.evaluate_async([final_result("Patient_001")]))

    assert report["diagnosis_accuracy"] == 1.0
    assert report["examination_precision"] == 1.0
    assert report["treatment_overall_score"] == 0.0
    assert report["treatment_safety"] == 0.0
    assert report["treatment_effectiveness_alignment"] == 0.0
    assert report["treatment_personalization"] == 0.0


def test_final_result_evaluator_preserves_deterministic_http_errors():
    class Unauthorized:
        async def evaluate_treatment(self, **kwargs):
            raise evaluator.LLMEvaluationHTTPError(401, "unauthorized")

    final_evaluator = evaluator.FinalResultEvaluator(
        CaseEvaluationRepository(),
        model="model",
        base_url="https://model.test/v1",
        api_key="secret",
    )
    final_evaluator.treatment_evaluator = Unauthorized()

    with pytest.raises(evaluator.LLMEvaluationHTTPError) as error:
        asyncio.run(final_evaluator.evaluate_async([final_result("Patient_001")]))

    assert error.value.status_code == 401
