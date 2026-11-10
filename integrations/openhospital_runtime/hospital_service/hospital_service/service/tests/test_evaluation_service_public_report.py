from __future__ import annotations

import asyncio
from typing import Any

from hospital_service.service.evaluation.evaluator import FinalResultEvaluator
from hospital_service.service.evaluation.service import EvaluationService


class FakeRepository:
    dataset_key = "test"

    def get_ground_truth(self, patient_id: str) -> dict[str, Any]:
        assert patient_id == "Patient_001"
        return {
            "final_diagnosis": "secret diagnosis",
            "necessary_examinations": ["secret exam"],
            "treatment_plan": "secret treatment",
            "contraindications": {"drugs": ["secret drug"]},
        }


class DiagnosisRepository:
    dataset_key = "test"

    def get_ground_truth(self, patient_id: str) -> dict[str, Any]:
        return {
            "Patient_001": {
                "final_diagnosis": "重症肺炎",
                "necessary_examinations": [],
                "treatment_plan": "",
            },
            "Patient_002": {
                "final_diagnosis": "肺炎",
                "necessary_examinations": [],
                "treatment_plan": "",
            },
            "Patient_003": {
                "final_diagnosis": "Secret Diagnosis",
                "necessary_examinations": [],
                "treatment_plan": "",
            },
            "Patient_004": {
                "final_diagnosis": ["糖尿病", "高血压"],
                "necessary_examinations": [],
                "treatment_plan": "",
            },
            "Patient_005": {
                "final_diagnosis": ["慢性鼻窦炎"],
                "necessary_examinations": [],
                "treatment_plan": "",
            },
        }[patient_id]


class CumulativeUsageStore:
    def __init__(self):
        self.calls: list[dict[str, int | str]] = []
        self.total_tokens = 1000

    def get_usage(self, *, dataset_key: str, team_id: str) -> dict[str, int]:
        return {"requests": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": self.total_tokens}

    def list_usage(self, *, dataset_key: str) -> dict[str, dict[str, int]]:
        return {}

    def increment_usage(
        self,
        *,
        dataset_key: str,
        team_id: str,
        input_tokens: int,
        output_tokens: int,
        total_tokens: int,
    ) -> dict[str, int]:
        self.calls.append({
            "dataset_key": dataset_key,
            "team_id": team_id,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
        })
        self.total_tokens += total_tokens
        return {
            "requests": len(self.calls),
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": self.total_tokens,
        }


class UnexpectedModelKeyPool:
    async def run_async(self, **kwargs):
        raise AssertionError("Forced-zero treatment evaluation must bypass the model key pool.")


def test_diagnosis_accuracy_requires_exact_normalized_match():
    evaluator = FinalResultEvaluator(DiagnosisRepository())  # type: ignore[arg-type]

    report = evaluator.evaluate([
        {
            "patient_id": "Patient_001",
            "diagnosis": ["肺炎"],
            "ordered_examinations": [],
            "treatment_plan": "",
        },
        {
            "patient_id": "Patient_002",
            "diagnosis": ["重症肺炎"],
            "ordered_examinations": [],
            "treatment_plan": "",
        },
        {
            "patient_id": "Patient_003",
            "diagnosis": [" secret diagnosis "],
            "ordered_examinations": [],
            "treatment_plan": "",
        },
    ])

    assert report["diagnosis_accuracy"] == 0.3333


def test_evaluation_service_accepts_internal_zero_treatment_mode():
    service = EvaluationService(
        FakeRepository(),  # type: ignore[arg-type]
        model_key_pool=UnexpectedModelKeyPool(),  # type: ignore[arg-type]
    )

    report = asyncio.run(service.evaluate_async({
        "team_id": "team-a",
        "_treatment_evaluation_mode": "zero",
        "final_result": [{
            "patient_id": "Patient_001",
            "diagnosis": ["secret diagnosis"],
            "ordered_examinations": ["secret exam"],
        }],
    }))

    assert report["diagnosis_accuracy"] == 1.0
    assert report["examination_precision"] == 1.0
    assert report["treatment_overall_score"] == 0.0
    assert report["treatment_safety"] == 0.0
    assert report["treatment_effectiveness_alignment"] == 0.0
    assert report["treatment_personalization"] == 0.0
    assert "_treatment_evaluation_mode" not in str(report)


def test_evaluation_service_routes_internal_scores_only_treatment_mode():
    captured = {}

    class CapturingEvaluator:
        async def evaluate_async(
            self,
            final_results,
            *,
            force_zero_treatment=False,
            scores_only_treatment=False,
        ):
            captured["final_results"] = final_results
            captured["force_zero_treatment"] = force_zero_treatment
            captured["scores_only_treatment"] = scores_only_treatment
            return {
                "diagnosis_accuracy": 1,
                "examination_precision": 1,
                "treatment_overall_score": 0.72,
                "treatment_safety": 1,
                "treatment_effectiveness_alignment": 0.8,
                "treatment_personalization": 0.6,
                "treatment_details": [{
                    "patient_id": "Patient_001",
                    "overall_score": 0.72,
                    "safety": 1,
                    "effectiveness_alignment": 0.8,
                    "personalization": 0.6,
                    "reasoning": "",
                }],
            }

    service = EvaluationService(FakeRepository())  # type: ignore[arg-type]
    service._build_evaluator = lambda **kwargs: CapturingEvaluator()  # type: ignore[method-assign]
    final_results = [{
        "patient_id": "Patient_001",
        "diagnosis": ["secret diagnosis"],
        "ordered_examinations": ["secret exam"],
        "treatment_plan": "submitted treatment",
    }]

    report = asyncio.run(service.evaluate_async({
        "team_id": "team-a",
        "_treatment_evaluation_mode": "scores_only",
        "final_result": final_results,
    }))

    assert captured == {
        "final_results": final_results,
        "force_zero_treatment": False,
        "scores_only_treatment": True,
    }
    assert report["treatment_overall_score"] == 0.72
    assert report["treatment_details"][0]["reasoning"] == ""
    assert "_treatment_evaluation_mode" not in str(report)


def test_diagnosis_accuracy_requires_predictions_to_be_subset_of_expected_diagnoses():
    evaluator = FinalResultEvaluator(DiagnosisRepository())  # type: ignore[arg-type]

    report = evaluator.evaluate([
        {
            "patient_id": "Patient_004",
            "diagnosis": ["糖尿病"],
            "ordered_examinations": [],
            "treatment_plan": "",
        },
        {
            "patient_id": "Patient_004",
            "diagnosis": ["糖尿病", "高血压"],
            "ordered_examinations": [],
            "treatment_plan": "",
        },
        {
            "patient_id": "Patient_005",
            "diagnosis": ["慢性鼻窦炎", "过敏性鼻炎"],
            "ordered_examinations": [],
            "treatment_plan": "",
        },
    ])

    assert report["diagnosis_accuracy"] == 0.5


def test_evaluation_usage_capture_keeps_llm_delta_not_cumulative_store_snapshot():
    usage_store = CumulativeUsageStore()
    service = EvaluationService(FakeRepository(), usage_store=usage_store)  # type: ignore[arg-type]
    evaluator = service._build_evaluator(team_id="team_001", record_usage_for_explicit_key=True)  # noqa: SLF001

    assert evaluator.usage_callback is not None
    evaluator.usage_callback({"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15})
    evaluator.usage_callback({"prompt_tokens": 20, "completion_tokens": 7, "total_tokens": 27})

    assert evaluator.usage_capture == [
        {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        {"prompt_tokens": 20, "completion_tokens": 7, "total_tokens": 27},
    ]
    assert [call["total_tokens"] for call in usage_store.calls] == [15, 27]
    assert usage_store.total_tokens == 1042


def test_case_report_diagnosis_detail_requires_exact_normalized_match():
    service = EvaluationService(FakeRepository())  # type: ignore[arg-type]
    report = {
        "diagnosis_accuracy": 0,
        "examination_precision": 0,
        "treatment_overall_score": 0,
        "treatment_safety": 0,
        "treatment_effectiveness_alignment": 0,
        "treatment_personalization": 0,
        "treatment_details": [],
    }

    case_report = service._case_report({  # noqa: SLF001
        "patient_id": "Patient_001",
        "diagnosis": ["secret"],
        "ordered_examinations": [],
        "treatment_plan": "",
    }, report, include_reference_answers=True)

    assert case_report["diagnosisDetail"]["matched"] == []
    assert case_report["diagnosisDetail"]["accuracy"] == 0.0
    assert case_report["diagnosisDetail"]["status"] == "wrong"


def test_case_report_diagnosis_detail_marks_extra_diagnoses_wrong():
    service = EvaluationService(DiagnosisRepository())  # type: ignore[arg-type]
    report = {
        "diagnosis_accuracy": 0,
        "examination_precision": 0,
        "treatment_overall_score": 0,
        "treatment_safety": 0,
        "treatment_effectiveness_alignment": 0,
        "treatment_personalization": 0,
        "treatment_details": [],
    }

    case_report = service._case_report({  # noqa: SLF001
        "patient_id": "Patient_005",
        "diagnosis": ["慢性鼻窦炎", "过敏性鼻炎"],
        "ordered_examinations": [],
        "treatment_plan": "",
    }, report, include_reference_answers=True)

    assert case_report["diagnosisDetail"]["matched"] == ["慢性鼻窦炎"]
    assert case_report["diagnosisDetail"]["accuracy"] == 0.0
    assert case_report["diagnosisDetail"]["status"] == "wrong"


def test_test_case_report_does_not_expose_ground_truth_fields():
    service = EvaluationService(FakeRepository())  # type: ignore[arg-type]
    report = {
        "diagnosis_accuracy": 1,
        "examination_precision": 1,
        "treatment_overall_score": 1,
        "treatment_safety": 1,
        "treatment_effectiveness_alignment": 1,
        "treatment_personalization": 1,
        "treatment_details": [{
            "patient_id": "Patient_001",
            "overall_score": 1,
            "safety": 1,
            "effectiveness_alignment": 1,
            "personalization": 1,
            "reasoning": "score explanation",
        }],
    }

    case_report = service._case_report({  # noqa: SLF001
        "patient_id": "Patient_001",
        "diagnosis": ["secret diagnosis"],
        "ordered_examinations": ["secret exam"],
        "treatment_plan": "submitted treatment",
    }, report)
    text = str(case_report)

    assert "ground_truth" not in case_report
    assert "secret treatment" not in text
    assert "secret drug" not in text
    assert "expected" not in text
    assert "matched" not in text
    assert "reference" not in text
    assert case_report["diagnosisDetail"] == {
        "submitted": ["secret diagnosis"],
        "accuracy": 1.0,
        "status": "correct",
    }
    assert case_report["examinationDetail"] == {
        "precisionEligible": True,
        "ordered": ["secret exam"],
        "precision": 1.0,
        "coverage": 1.0,
    }


def test_train_case_report_keeps_reference_answer_fields():
    service = EvaluationService(FakeRepository())  # type: ignore[arg-type]
    report = {
        "diagnosis_accuracy": 1,
        "examination_precision": 1,
        "treatment_overall_score": 1,
        "treatment_safety": 1,
        "treatment_effectiveness_alignment": 1,
        "treatment_personalization": 1,
        "treatment_details": [{
            "patient_id": "Patient_001",
            "overall_score": 1,
            "safety": 1,
            "effectiveness_alignment": 1,
            "personalization": 1,
            "reasoning": "score explanation",
        }],
    }

    case_report = service._case_report(  # noqa: SLF001
        {
            "patient_id": "Patient_001",
            "diagnosis": ["secret diagnosis"],
            "ordered_examinations": ["secret exam"],
            "treatment_plan": "submitted treatment",
        },
        report,
        include_reference_answers=True,
    )

    assert case_report["ground_truth"]["treatment_plan"] == "secret treatment"
    assert case_report["diagnosisDetail"]["expected"] == ["secret diagnosis"]
    assert case_report["diagnosisDetail"]["matched"] == ["secret diagnosis"]
    assert case_report["examinationDetail"]["expected"] == ["secret exam"]
    assert case_report["examinationDetail"]["matched"] == ["secret exam"]
    assert case_report["treatmentDetail"]["reference"] == "secret treatment"
